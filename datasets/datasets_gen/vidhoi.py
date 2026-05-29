# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import json
import random
import time
import numpy as np

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
from ModifiedCLIP import clip
from datasets.vidhoi_text_label import vidhoi_text_label


class VIDHOI(torch.utils.data.Dataset):
    def __init__(
        self,
        img_set,
        img_folder,
        anno_file,
        transforms,
        train_ratio,
        num_queries,
        args,
    ):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, "r") as f:
            self.annotations = json.load(f)

        self._transforms = transforms
        self.num_queries = num_queries
        self.train_ratio = train_ratio
        self.annotations = self.annotations[: int(len(self.annotations) * self.train_ratio)]
        self._valid_verb_ids = list(range(args.num_verb_classes))  # [0, 1, ..., 49]

        # HOI text label dict: (verb_id, obj_cat_id) -> text  (same structure as hico_text_label)
        self.text_label_ids = list(vidhoi_text_label.keys())

        _, self.clip_preprocess = clip.load(args.clip_model)

        print(f"{self.img_set} set: totally {len(self.annotations)} items, sample_mode: frame")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]
        img = open_with_retries(self.img_folder / img_anno["file_name"])
        w, h = img.size

        if self.img_set == "train" and len(img_anno["annotations"]) > self.num_queries:
            img_anno["annotations"] = img_anno["annotations"][: self.num_queries]

        boxes = [obj["bbox"] for obj in img_anno["annotations"]]
        classes = [obj["category_id"] for obj in img_anno["annotations"]]

        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        if self.img_set == "train":
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target["boxes"] = boxes
            target["labels"] = classes
            target["iscrowd"] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target = self._transforms[1](img_0, target_0)

            clip_inputs = self.clip_preprocess(img_0)
            target["clip_inputs"] = clip_inputs
            target["filename"] = img_anno["file_name"]

            obj_labels, verb_labels, hoi_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                if sub_id > len(target["labels"]) - 1 or obj_id > len(target["labels"]) - 1:
                    continue

                # (verb_id, obj_category) pair must exist in the text label dict
                verb_obj_pair = (hoi["category_id"], int(target["labels"][obj_id]))
                if verb_obj_pair not in self.text_label_ids:
                    continue
                hoi_class_idx = self.text_label_ids.index(verb_obj_pair)

                sub_obj_pair = (sub_id, obj_id)
                if sub_obj_pair in sub_obj_pairs:
                    idx_pair = sub_obj_pairs.index(sub_obj_pair)
                    verb_labels[idx_pair][hoi["category_id"]] = 1
                    hoi_labels[idx_pair][hoi_class_idx] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target["labels"][obj_id])
                    verb_label = [0] * len(self._valid_verb_ids)
                    verb_label[hoi["category_id"]] = 1
                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[hoi_class_idx] = 1
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(target["boxes"][sub_id])
                    obj_boxes.append(target["boxes"][obj_id])

            if len(sub_obj_pairs) == 0:
                target["obj_labels"] = torch.zeros((0,), dtype=torch.int64)
                target["verb_labels"] = torch.zeros(
                    (0, len(self._valid_verb_ids)), dtype=torch.float32
                )
                target["hoi_labels"] = torch.zeros(
                    (0, len(self.text_label_ids)), dtype=torch.float32
                )
                target["sub_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["obj_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target["obj_labels"] = torch.stack(obj_labels)
                target["verb_labels"] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target["hoi_labels"] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target["sub_boxes"] = torch.stack(sub_boxes)
                target["obj_boxes"] = torch.stack(obj_boxes)
        else:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.tensor(classes, dtype=torch.int64)
            target["id"] = idx
            target["filename"] = img_anno["file_name"]

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img, _ = self._transforms[1](img_0, None)

            clip_inputs = self.clip_preprocess(img_0)
            target["clip_inputs"] = clip_inputs

            hois = []
            for hoi in img_anno["hoi_annotation"]:
                hois.append(
                    (
                        hoi["subject_id"],
                        hoi["object_id"],
                        hoi["category_id"],
                    )
                )
            target["hois"] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


def make_vcoco_transforms(image_set):
    """Returns a list [augment_transform, normalize_transform] for two-stage application."""
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return [
            T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.4, 0.4, 0.4),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=1333),
                        T.Compose(
                            [
                                T.RandomResize([400, 500, 600]),
                                T.RandomSizeCrop(384, 600),
                                T.RandomResize(scales, max_size=1333),
                            ]
                        ),
                    ),
                ]
            ),
            normalize,
        ]

    if image_set == "val" or image_set == "test":
        return [
            T.Compose([T.RandomResize([800], max_size=1333)]),
            normalize,
        ]

    raise ValueError(f"unknown {image_set}")


def open_with_retries(path, retries=6, base_delay=0.05):
    last = None
    for attempt in range(retries):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except (OSError, UnidentifiedImageError, IOError) as e:
            last = e
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, base_delay))
    raise OSError(f"Open image failed after {retries+1} retries: {path} ({last})")


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f"provided HOI path {root} does not exist"
    PATHS = {
        "train": (
            root / "images",
            root / "VidHOI_annotation" / "train_frame_hoia.json",
        ),
        "val": (
            root / "images",
            root / "VidHOI_annotation" / "val_frame_hoia.json",
        ),
    }

    img_folder, anno_file = PATHS[image_set]
    dataset = VIDHOI(
        image_set,
        img_folder,
        anno_file,
        transforms=make_vcoco_transforms(image_set),
        train_ratio=args.train_ratio,
        num_queries=args.num_queries,
        args=args,
    )
    return dataset
