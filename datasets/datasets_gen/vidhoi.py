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
        self._valid_verb_ids = list(range(args.num_verb_classes))

        self.text_label_ids = list(vidhoi_text_label.keys())
        # O(1) lookup: (verb_id, obj_class) -> HOI class index
        self.text_label_to_idx = {pair: i for i, pair in enumerate(self.text_label_ids)}

        _, self.clip_preprocess = clip.load(args.clip_model)

        valid_ids = []
        for idx, img_anno in enumerate(self.annotations):
            new_hoi_anno = []
            for hoi in img_anno['hoi_annotation']:
                # Invalidate entire image if any subject/object id is out of bounds
                if (hoi['subject_id'] >= len(img_anno['annotations']) or
                        hoi['object_id'] >= len(img_anno['annotations'])):
                    new_hoi_anno = []
                    break
                new_hoi_anno.append(hoi)
            if len(new_hoi_anno) > 0:
                valid_ids.append(idx)
                img_anno['hoi_annotation'] = new_hoi_anno  # drop stale entries in-place (init only)

        # Apply train_ratio after validity filtering so the fraction is over clean data
        valid_ids = valid_ids[: int(len(valid_ids) * self.train_ratio)]

        # Compact to release memory for filtered-out images.
        # image_ids retains each sample's original JSON line number for stable
        # cross-rank deduplication in engine.py's np.unique step.
        self.image_ids = valid_ids
        self.annotations = [self.annotations[i] for i in valid_ids]
        self.ids = list(range(len(self.annotations)))  # trivial 0..N-1 after compaction

        print(f"{self.img_set} set: {len(self.ids)} valid images, sample_mode: frame")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Shallow-copy the annotation dict so that any key reassignment below
        # (e.g. truncating 'annotations') does not mutate the shared self.annotations list.
        img_anno = dict(self.annotations[self.ids[idx]])
        img = open_with_retries(self.img_folder / img_anno["file_name"])
        w, h = img.size

        if self.img_set == "train" and len(img_anno["annotations"]) > self.num_queries:
            img_anno["annotations"] = img_anno["annotations"][: self.num_queries]

        boxes = [obj["bbox"] for obj in img_anno["annotations"]]
        # Plain class-id list; train branch will override with (orig_idx, class_id) tuples.
        classes = [obj["category_id"] for obj in img_anno["annotations"]]

        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        if self.img_set == "train":
            # Store (original_annotation_idx, class_id) so we can track which boxes
            # survive the keep-mask and random-crop transforms.
            classes = [(i, obj["category_id"]) for i, obj in enumerate(img_anno["annotations"])]
            classes = torch.tensor(classes, dtype=torch.int64)

            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]  # still (orig_idx, class_id) pairs

            target["boxes"] = boxes
            target["labels"] = classes  # shape (N, 2)
            target["iscrowd"] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target = self._transforms[1](img_0, target_0)
            else:
                img_0 = img  # fallback so clip_preprocess always has a valid source

            clip_inputs = self.clip_preprocess(img_0)
            target["clip_inputs"] = clip_inputs
            target["filename"] = img_anno["file_name"]

            # Original annotation indices that survived the keep-mask + random-crop
            kept_box_indices = [int(label[0]) for label in target["labels"]]
            target["labels"] = target["labels"][:, 1]  # drop index, keep class_id

            obj_labels, verb_labels, hoi_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs = []
            sub_obj_pair_to_idx = {}  # O(1) duplicate-pair lookup
            for hoi in img_anno["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                # Skip if subject or object bbox was removed by transforms
                if sub_id not in kept_box_indices or obj_id not in kept_box_indices:
                    continue

                obj_class = int(target["labels"][kept_box_indices.index(obj_id)])
                verb_obj_pair = (hoi["category_id"], obj_class)
                hoi_class_idx = self.text_label_to_idx.get(verb_obj_pair)
                if hoi_class_idx is None:
                    continue

                sub_obj_pair = (sub_id, obj_id)
                if sub_obj_pair in sub_obj_pair_to_idx:
                    i = sub_obj_pair_to_idx[sub_obj_pair]
                    verb_labels[i][hoi["category_id"]] = 1
                    hoi_labels[i][hoi_class_idx] = 1
                else:
                    i = len(sub_obj_pairs)
                    sub_obj_pair_to_idx[sub_obj_pair] = i
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target["labels"][kept_box_indices.index(obj_id)])
                    verb_label = [0] * len(self._valid_verb_ids)
                    verb_label[hoi["category_id"]] = 1
                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[hoi_class_idx] = 1
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(target["boxes"][kept_box_indices.index(sub_id)])
                    obj_boxes.append(target["boxes"][kept_box_indices.index(obj_id)])

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
            target["filename"] = img_anno["file_name"]
            # Use the original JSON index as the sample id so that DistributedSampler
            # padding duplicates are correctly deduplicated by engine.py's np.unique.
            target["id"] = self.image_ids[idx]

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img, _ = self._transforms[1](img_0, None)
            else:
                img_0 = img  # fallback so clip_preprocess always has a valid source

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
    raise OSError(f"Open image failed after {retries} retries: {path} ({last})")


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
