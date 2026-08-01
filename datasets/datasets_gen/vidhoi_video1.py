# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import json
import random
import copy
import time
import numpy as np

import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import torchvision

import datasets.transforms as Tf
from ModifiedCLIP import clip
from datasets.vidhoi_text_label import vidhoi_text_label


class VidHOI(torch.utils.data.Dataset):
    def __init__(
        self,
        img_set,
        img_folder,
        anno_file,
        transforms,
        train_ratio,
        num_queries,
    ):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, "r") as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.num_queries = num_queries
        self._valid_verb_ids=range(50)
        self.train_ratio = train_ratio

        self.annotations = self.annotations[: int(len(self.annotations) * self.train_ratio)]

        self.object_text = load_vidhoi_text(anno_file.parent / "obj_categories.json")
        self.verb_text = load_vidhoi_text(anno_file.parent / "pred_categories.json")   
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
            # guard against no boxes via resizing: when no boxes, also return tensor(0,4)
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
            target['obj_classes'] = self.object_text
            target['verb_classes'] = self.verb_text

            if self._transforms is not None:
                img, target = self._transforms(img, target)
            """
            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]
            """
            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for i,hoi in enumerate(img_anno["hoi_annotation"]):
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                if sub_id > len(target["labels"])-1 or obj_id > len(target["labels"])-1:
                    continue

                sub_obj_pair = (sub_id, obj_id)
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][
                        self._valid_verb_ids.index(hoi["category_id"])
                    ] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target["labels"][obj_id])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi["category_id"])] = 1
                    verb_labels.append(verb_label)
                    sub_boxes.append(target["boxes"][sub_id])
                    obj_boxes.append(target["boxes"][obj_id])

            if len(sub_obj_pairs) == 0:
                target["obj_labels"] = torch.zeros((0,), dtype=torch.int64)
                target["sub_labels"] = torch.zeros((0,), dtype=torch.int64)
                target["verb_labels"] = torch.zeros(
                    (0, len(self._valid_verb_ids)), dtype=torch.float32
                )
                target["sub_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["obj_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target["obj_labels"] = torch.stack(obj_labels)
                target['sub_labels'] = torch.zeros((len(obj_labels),), dtype=torch.int64)
                target["verb_labels"] = torch.as_tensor(
                    verb_labels, dtype=torch.float32
                )
                target["sub_boxes"] = torch.stack(sub_boxes)
                target["obj_boxes"] = torch.stack(obj_boxes)
        else:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.tensor(classes, dtype=torch.int64)
            target["id"] = idx
            #target["img_id"] = int(img_anno["file_name"].rstrip(".jpg").split("/")[2])

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

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


class VidHOI_Clip(torch.utils.data.Dataset):
    """Window/clip-sampled VidHOI dataset with HOICLIP-compatible single-frame output.

    Each item is the end frame of a sampled window.  The window structure
    controls temporal coverage during training; HOICLIP processes each frame
    independently (no temporal modelling changes required).
    """

    def __init__(
        self,
        img_set,
        img_folder,
        anno_file,
        meta_file,
        transforms,
        train_ratio,
        val_ratio,
        sample_mode,
        min_length,
        max_length,
        min_future,
        num_queries,
        args,
        anticipation=False
    ):
        self.img_set = img_set
        self.img_folder = img_folder
        self.sample_mode = sample_mode
        self.min_length = min_length
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self._transforms = transforms
        self.num_queries = num_queries
        self._valid_verb_ids = list(range(args.num_verb_classes))
        # clip 模式下，__getitem__ 返回 T 帧序列 + 未来帧标注
        self.anticipation = anticipation
        self.futures = [1, 3, 5, 7]

        # HOICLIP: HOI text-label lookup (same as VIDHOI)
        self.text_label_ids = list(vidhoi_text_label.keys())
        self.text_label_to_idx = {pair: i for i, pair in enumerate(self.text_label_ids)}
        _, self.clip_preprocess = clip.load(args.clip_model)

        with open(anno_file, "r") as f:
            self.annotations = json.load(f)
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        if self.img_set == "train":
            meta_split = metadata['train']
            self.output_list = []
            # window start frames and video last frames for clip/anticipation mode
            self.output_start_list = []
            self.output_video_end_list = []
            if self.sample_mode == "clip":
                # Non-overlapping clips; output_list holds 1-indexed end frame IDs.
                for meta in meta_split:
                    frame_ids = meta["frame_ids"]
                    num_windows = meta["video_length"] // self.max_length
                    if num_windows > 0:
                        residue = meta["video_length"] % self.max_length
                        if residue > num_windows:
                            attached_length = residue // num_windows
                            residue = residue - attached_length * num_windows
                        else:
                            attached_length = 0
                        clip_length = [self.max_length + attached_length] * (num_windows - 1) + \
                                      [self.max_length + attached_length + residue]
                        clip_indices = np.cumsum([0] + clip_length) + frame_ids[0]
                        assert clip_indices[-1] - 1 == frame_ids[-1], \
                            f"Invalid clip indices at video {meta['video_id']}"
                        end_frames = (clip_indices[1:] - 1).tolist()
                        start_frames = clip_indices[:-1].tolist()
                        self.output_list.extend(end_frames)
                        self.output_start_list.extend(start_frames)
                        self.output_video_end_list.extend([frame_ids[-1]] * len(end_frames))
            elif self.sample_mode == "window":
                # Fixed-length non-overlapping windows; take the end frame of each window.
                for meta in meta_split:
                    frame_ids = meta["frame_ids"]
                    num_windows = meta["video_length"] // self.max_length
                    for w in range(num_windows):
                        start_frame = frame_ids[0] + w * self.max_length
                        end_frame   = frame_ids[0] + (w + 1) * self.max_length - 1
                        self.output_list.append(end_frame)
                        self.output_start_list.append(start_frame)
                        self.output_video_end_list.append(frame_ids[-1])
            else:
                raise ValueError(f"unknown sample mode {self.sample_mode}")
            self.output_list            = self.output_list[:int(len(self.output_list) * self.train_ratio)]
            self.output_start_list      = self.output_start_list[:len(self.output_list)]
            self.output_video_end_list  = self.output_video_end_list[:len(self.output_list)]

        else:  # val
            meta_split = metadata['val']
            self.output_list = []
            self.output_start_list = []
            self.output_video_end_list = []
            for meta in meta_split:
                if meta["video_length"] >= self.min_length:
                    # Every frame that can be the end of a min_length window
                    frames = meta["frame_ids"][self.min_length - 1: meta["video_length"]]
                    self.output_list.extend(frames)
                    # start = end - min_length + 1 for each frame
                    self.output_start_list.extend(
                        [f - self.min_length + 1 for f in frames]
                    )
                    self.output_video_end_list.extend([meta["frame_ids"][-1]] * len(frames))
            self.output_list            = self.output_list[:int(len(self.output_list) * self.val_ratio)]
            self.output_start_list      = self.output_start_list[:len(self.output_list)]
            self.output_video_end_list  = self.output_video_end_list[:len(self.output_list)]

        print(f"{self.img_set} set: {len(self.output_list)} valid items, sample_mode: {self.sample_mode}")
    
    def __len__(self):
        return len(self.output_list)

    def _load_single_frame(self, frame_id):
        """加载并预处理单帧，返回 (img_tensor, img_0_pil, target) 供 __getitem__ 复用。"""
        img_anno = dict(self.annotations[frame_id - 1])
        img = open_with_retries(self.img_folder / img_anno["file_name"])
        w, h = img.size

        if self.img_set == "train" and len(img_anno["annotations"]) > self.num_queries:
            img_anno["annotations"] = img_anno["annotations"][:self.num_queries]

        boxes   = [obj["bbox"] for obj in img_anno["annotations"]]
        target  = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"]      = torch.as_tensor([int(h), int(w)])

        if self.img_set == "train":
            classes = [(i, obj["category_id"]) for i, obj in enumerate(img_anno["annotations"])]
            classes = torch.tensor(classes, dtype=torch.int64)
            boxes   = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep    = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes   = boxes[keep]
            classes = classes[keep]
            target["boxes"]   = boxes
            target["labels"]  = classes
            target["iscrowd"] = torch.tensor([0] * boxes.shape[0])
            target["area"]    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img_t, target   = self._transforms[1](img_0, target_0)
            else:
                img_0 = img
                img_t = img

            target["clip_inputs"] = self.clip_preprocess(img_0)
            target["filename"]    = img_anno["file_name"]

            kept_box_indices = [int(label[0]) for label in target["labels"]]
            target["labels"] = target["labels"][:, 1]

            obj_labels, verb_labels, hoi_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs     = []
            sub_obj_pair_to_idx = {}
            for hoi in img_anno["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                if sub_id not in kept_box_indices or obj_id not in kept_box_indices:
                    continue
                obj_class    = int(target["labels"][kept_box_indices.index(obj_id)])
                hoi_class_idx = self.text_label_to_idx.get((hoi["category_id"], obj_class))
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
                    vl = [0] * len(self._valid_verb_ids)
                    vl[hoi["category_id"]] = 1
                    hl = [0] * len(self.text_label_ids)
                    hl[hoi_class_idx] = 1
                    verb_labels.append(vl)
                    hoi_labels.append(hl)
                    sub_boxes.append(target["boxes"][kept_box_indices.index(sub_id)])
                    obj_boxes.append(target["boxes"][kept_box_indices.index(obj_id)])

            if len(sub_obj_pairs) == 0:
                target["obj_labels"]  = torch.zeros((0,), dtype=torch.int64)
                target["verb_labels"] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target["hoi_labels"]  = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target["sub_boxes"]   = torch.zeros((0, 4), dtype=torch.float32)
                target["obj_boxes"]   = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target["obj_labels"]  = torch.stack(obj_labels)
                target["verb_labels"] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target["hoi_labels"]  = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target["sub_boxes"]   = torch.stack(sub_boxes)
                target["obj_boxes"]   = torch.stack(obj_boxes)
        else:  # val
            classes = [obj["category_id"] for obj in img_anno["annotations"]]
            target["boxes"]  = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.tensor(classes, dtype=torch.int64)
            target["filename"] = img_anno["file_name"]
            target["id"]       = frame_id - 1

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img_t, _ = self._transforms[1](img_0, None)
            else:
                img_0 = img
                img_t = img

            target["clip_inputs"] = self.clip_preprocess(img_0)
            hois = [(hoi["subject_id"], hoi["object_id"], hoi["category_id"])
                    for hoi in img_anno["hoi_annotation"]]
            target["hois"] = torch.as_tensor(hois, dtype=torch.int64)

        return img_t, target, img_anno

    def _load_frame_clip(self, frame_ann, pair_list, anticipation=False):
        """按照 vidhoi_video_copy.py 的 _load_one_frame 模式加载单帧。

        用 obj["id"]（track ID）维护跨帧 pair_list，
        输出 hoi_labels（N_hoi 维）替代 copy.py 中的 verb_labels（50 维）。
        anticipation=True 时只处理标注，不加载图像。
        返回 (img_tensor_or_None, target, pair_list)。
        """
        if not anticipation:
            img = open_with_retries(self.img_folder / frame_ann["file_name"])
            w, h = img.size
        else:
            img = None

        boxes, classes, tracks = [], [], []
        for obj in frame_ann["annotations"]:
            boxes.append(obj["bbox"])
            classes.append(obj["category_id"])
            tracks.append(obj["id"])  # VidHOI track ID

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {}

        if self.img_set == "train":
            if len(frame_ann["annotations"]) > self.num_queries:
                boxes   = boxes[:self.num_queries]
                classes = classes[:self.num_queries]
                tracks  = tracks[:self.num_queries]

            hoi_labels_list = []
            sub_obj_pairs   = []
            pair_indices    = []

            for hoi in frame_ann["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                if sub_id >= len(tracks) or obj_id >= len(tracks):
                    continue
                obj_cls  = classes[obj_id]
                hoi_idx  = self.text_label_to_idx.get((hoi["category_id"], obj_cls))
                if hoi_idx is None:
                    continue
                sub_obj_pair = (tracks[sub_id], tracks[obj_id])
                if sub_obj_pair in sub_obj_pairs:
                    hoi_labels_list[sub_obj_pairs.index(sub_obj_pair)][hoi_idx] = 1
                else:
                    if sub_obj_pair not in pair_list:
                        if anticipation:
                            continue
                        pair_list.append(sub_obj_pair)
                    hl = [0] * len(self.text_label_ids)
                    hl[hoi_idx] = 1
                    hoi_labels_list.append(hl)
                    sub_obj_pairs.append(sub_obj_pair)
                    pair_indices.append(pair_list.index(sub_obj_pair))

            if not anticipation:
                target_inner = {
                    "boxes":   boxes,
                    "labels":  torch.as_tensor([(i, c) for i, c in enumerate(classes)],
                                               dtype=torch.int64),
                    "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                    "area":    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                }
                if self._transforms is not None:
                    # Backbone input: fixed resize so all T frames share the same H×W.
                    # RandomResize with a single scale is deterministic per resolution.
                    _fixed_resize = Tf.RandomResize([800], max_size=1333)
                    img_resized, target_inner = _fixed_resize(img, target_inner)
                    img_t, target_inner = self._transforms[1](img_resized, target_inner)
                    # CLIP preprocess uses the random augmentation for better generalisation.
                    # Pass empty dict — no boxes to track for CLIP.
                    img_0, _ = self._transforms[0](img, {})
                else:
                    img_0 = img_resized = img
                    img_t = img
                target["boxes"]       = target_inner["boxes"]
                target["clip_inputs"] = self.clip_preprocess(img_0)
            else:
                img_t = None
                target["boxes"] = boxes
                target["clip_inputs"] = None

            target["labels"]      = torch.as_tensor(classes, dtype=torch.int64)
            target["tracks"]      = torch.as_tensor(tracks,  dtype=torch.int64)
            target["pair_indices"] = (torch.as_tensor(pair_indices, dtype=torch.int64)
                                      if pair_indices else torch.zeros(0, dtype=torch.int64))
            target["hoi_labels"]   = (torch.as_tensor(hoi_labels_list, dtype=torch.float32)
                                      if hoi_labels_list
                                      else torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32))

        else:  # val
            hois        = []
            pair_indices = []
            for hoi in frame_ann["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                if sub_id >= len(tracks) or obj_id >= len(tracks):
                    continue
                sub_obj_pair = (tracks[sub_id], tracks[obj_id])
                if anticipation:
                    if sub_obj_pair in pair_list:
                        hois.append({"subject_id": pair_list.index(sub_obj_pair),
                                     "object_id":  pair_list.index(sub_obj_pair),
                                     "category_id": hoi["category_id"]})
                else:
                    if sub_obj_pair not in pair_list:
                        pair_list.append(sub_obj_pair)
                    pair_indices.append(pair_list.index(sub_obj_pair))
                    hois.append(hoi)

            if not anticipation:
                if self._transforms is not None:
                    img_0, _ = self._transforms[0](img, None)
                    img_t, _ = self._transforms[1](img_0, None)
                else:
                    img_0 = img
                    img_t = img
                target["clip_inputs"] = self.clip_preprocess(img_0)
            else:
                img_t = None
                target["clip_inputs"] = None

            target["boxes"]       = boxes
            target["labels"]      = torch.as_tensor(classes, dtype=torch.int64)
            target["tracks"]      = tracks
            target["pair_indices"] = pair_indices
            target["hois"]         = hois

        return img_t, target, pair_list

    def __getitem__(self, idx):
        # output_list holds 1-indexed frame IDs; annotation list is 0-indexed.
        frame_id = self.output_list[idx]

        # --- clip 模式：仿照 vidhoi_video_copy.py 的 __getitem__ ---
        if self.anticipation:
            start_frame = self.output_start_list[idx]
            end_frame   = self.output_list[idx]
            video_end   = self.output_video_end_list[idx]

            pair_list = []
            imgs      = []

            if self.img_set == "train":
                raw = {"boxes": [], "labels": [], "tracks": [], "hoi_labels": [],
                       "clip_inputs": [], "pair_indices": [], "file_names": []}

                for f_id in range(start_frame, end_frame + 1):
                    img_t, ft, pair_list = self._load_frame_clip(
                        self.annotations[f_id - 1], pair_list
                    )
                    imgs.append(img_t)
                    raw["boxes"].append(ft["boxes"])
                    raw["labels"].append(ft["labels"])
                    raw["tracks"].append(ft["tracks"])
                    raw["hoi_labels"].append(ft["hoi_labels"])
                    raw["clip_inputs"].append(ft["clip_inputs"])
                    raw["pair_indices"].append(ft["pair_indices"])
                    raw["file_names"].append(self.annotations[f_id - 1]["file_name"])

                # 拼帧（参照 copy.py line 344）
                imgs = torch.cat(imgs, dim=0)  # (T*3, H, W)

                num_pairs = len(pair_list)
                if num_pairs == 0:
                    # 无有效 pair：返回空标注
                    T = end_frame - start_frame + 1
                    N = len(self.text_label_ids)
                    targets = {
                        "sub_boxes": torch.zeros((0, 4)), "obj_boxes": torch.zeros((0, 4)),
                        "obj_labels": torch.zeros(0, dtype=torch.int64),
                        "hoi_labels": torch.zeros((0, N)),
                        "valid": torch.zeros((T, 0), dtype=torch.int64),
                        "clip_inputs": torch.stack(raw["clip_inputs"]),
                        "orig_size": raw["boxes"][0].new_zeros(2),
                        "id": idx,
                    }
                    return imgs, targets

                ho_pairs      = torch.as_tensor(pair_list, dtype=torch.int64)
                hois_zero     = torch.zeros((num_pairs, len(self.text_label_ids)))
                valid_zero    = torch.zeros(num_pairs, dtype=torch.int64)
                box_invalid   = torch.zeros((1, 4))
                label_invalid = torch.full((1,), 78, dtype=torch.int64)

                sub_boxes_list, obj_boxes_list, obj_labels_list = [], [], []
                hoi_labels_list, valid_list = [], []

                for box, label, track, hoi_lab, pair_idx in zip(
                    raw["boxes"], raw["labels"], raw["tracks"],
                    raw["hoi_labels"], raw["pair_indices"]
                ):
                    # track_map: track_id → position in current frame's box list
                    track_map = torch.full((int(ho_pairs.max().item()) + 1,),
                                           -1, dtype=torch.int64)
                    try:
                        track_map[track] = torch.arange(len(track))
                    except IndexError:
                        pass
                    s_idx = track_map[ho_pairs[:, 0]]
                    o_idx = track_map[ho_pairs[:, 1]]
                    box_aug   = torch.cat((box, box_invalid), dim=0)
                    label_aug = torch.cat((label, label_invalid), dim=0)
                    sub_boxes_list.append(box_aug[s_idx])
                    obj_boxes_list.append(box_aug[o_idx])
                    obj_labels_list.append(label_aug[o_idx])
                    hoi_labels_list.append(
                        hois_zero.index_copy(0, pair_idx, hoi_lab) if len(pair_idx) > 0 else hois_zero
                    )
                    valid_list.append(
                        valid_zero.index_copy(0, pair_idx,
                                              torch.ones_like(pair_idx, dtype=torch.int64))
                        if len(pair_idx) > 0 else valid_zero
                    )

                obj_l = torch.stack(obj_labels_list, dim=1)   # (num_pairs, T)
                for i, row in enumerate(obj_l):
                    obj_l[i] = torch.full_like(row, row.min())

                targets = {
                    "sub_boxes":  torch.cat(sub_boxes_list),
                    "obj_boxes":  torch.cat(obj_boxes_list),
                    "obj_labels": torch.flatten(obj_l.transpose(0, 1)),
                    "hoi_labels": torch.cat(hoi_labels_list),
                    "valid":      torch.stack(valid_list),
                    "clip_inputs": torch.stack(raw["clip_inputs"]),  # (T, 3, 224, 224)
                    "filename":   raw["file_names"][-1],  # engine.py expects singular key
                    "file_names": raw["file_names"],
                    "orig_size":  raw["boxes"][-1].new_zeros(2),
                    "id":         idx,
                }

                # 未来帧标注（参照 copy.py line 441-465）
                anticipations = {"future": [], "hoi_labels_future": [], "valid_future": []}
                for h in self.futures:
                    f_future = end_frame + h
                    if f_future > video_end:
                        break
                    _, fut, _ = self._load_frame_clip(
                        self.annotations[f_future - 1], pair_list, anticipation=True
                    )
                    pair_idx = fut["pair_indices"]
                    anticipations["future"].append(h)
                    if len(pair_idx) > 0:
                        anticipations["hoi_labels_future"].append(
                            hois_zero.index_copy(0, pair_idx, fut["hoi_labels"])
                        )
                        anticipations["valid_future"].append(
                            valid_zero.index_copy(0, pair_idx,
                                                  torch.ones_like(pair_idx, dtype=torch.int64))
                        )
                    else:
                        anticipations["hoi_labels_future"].append(hois_zero)
                        anticipations["valid_future"].append(valid_zero)

                if anticipations["future"]:
                    anticipations["hoi_labels_future"] = torch.cat(anticipations["hoi_labels_future"])
                    anticipations["valid_future"]      = torch.stack(anticipations["valid_future"])
                    targets.update(anticipations)

            else:  # val
                raw = {"boxes": [], "labels": [], "tracks": [], "hois": [],
                       "clip_inputs": [], "pair_indices": [], "file_names": []}

                for f_id in range(start_frame, end_frame + 1):
                    img_t, ft, pair_list = self._load_frame_clip(
                        self.annotations[f_id - 1], pair_list
                    )
                    imgs.append(img_t)
                    raw["boxes"].append(ft["boxes"])
                    raw["labels"].append(ft["labels"])
                    raw["tracks"].append(ft["tracks"])
                    raw["hois"].append(ft["hois"])
                    raw["clip_inputs"].append(ft["clip_inputs"])
                    raw["pair_indices"].append(ft["pair_indices"])
                    raw["file_names"].append(self.annotations[f_id - 1]["file_name"])

                imgs = torch.cat(imgs, dim=0)  # (T*3, H, W)

                valid = torch.as_tensor(
                    [[1 if i in pi else 0 for i in range(len(pair_list))]
                     for pi in raw["pair_indices"]],
                    dtype=torch.int64
                )
                # Evaluator uses last-frame boxes/labels/hois for per-frame mAP.
                last_hois = raw["hois"][-1]
                if isinstance(last_hois, list) and last_hois and isinstance(last_hois[0], dict):
                    # normalise to tensor of (sub_id, obj_id, verb_id)
                    last_hois_t = torch.as_tensor(
                        [(h["subject_id"], h["object_id"], h["category_id"]) for h in last_hois],
                        dtype=torch.int64,
                    ) if last_hois else torch.zeros((0, 3), dtype=torch.int64)
                else:
                    last_hois_t = last_hois
                targets = {
                    "boxes":       raw["boxes"][-1],   # last frame for evaluator
                    "labels":      raw["labels"][-1],
                    "tracks":      raw["tracks"],
                    "hois":        last_hois_t,
                    "valid":       valid,
                    "clip_inputs": torch.stack(raw["clip_inputs"]),
                    "filename":    raw["file_names"][-1],
                    "file_names":  raw["file_names"],
                    "orig_size":   raw["boxes"][-1].new_zeros(2),
                    "id":          idx,
                }

                # 未来帧标注（参照 copy.py line 488-517）
                pair_list_last = [pair_list[i] for i in raw["pair_indices"][-1]]
                track_future   = raw["tracks"][-1]
                anticipations  = {"future": [], "hois_future": []}
                for h in self.futures:
                    f_future = end_frame + h
                    if f_future > video_end:
                        break
                    _, fut, _ = self._load_frame_clip(
                        self.annotations[f_future - 1], pair_list_last, anticipation=True
                    )
                    hois_future = [
                        {"subject_id": track_future.index(hoi["subject_id"])
                                       if hoi["subject_id"] in track_future else -1,
                         "object_id":  track_future.index(hoi["object_id"])
                                       if hoi["object_id"] in track_future else -1,
                         "category_id": hoi["category_id"]}
                        for hoi in fut["hois"]
                        if hoi["subject_id"] in track_future and hoi["object_id"] in track_future
                    ]
                    if hois_future:
                        anticipations["future"].append(h)
                        anticipations["hois_future"].append(hois_future)

                if anticipations["future"]:
                    targets.update(anticipations)

            return imgs, targets

        # anticipation=False: 单帧模式，复用 _load_single_frame
        img_t, target, _ = self._load_single_frame(frame_id)
        return img_t, target


def open_with_retries(path, retries=6, base_delay=0.05):
    last = None
    for attempt in range(retries):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except (OSError, UnidentifiedImageError, IOError) as e:
            last = e
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, base_delay))  # Exponential backoff with jitter to mitigate concurrent contention.
    raise OSError(f"Open image failed after {retries+1} retries: {path} ({last})")



def make_vidhoi_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=768),
                    #T.Check(),
                ]),
                T.Compose(transforms=[
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ]), p=1),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            # T.RandomResize([360], max_size=640),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')


# Add color jitter to coco transforms
def video_transforms(image_set):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResize(480),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize(480),
            normalize
        ])

    raise ValueError(f"unknown {image_set}")


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set):
    normalize = Tf.Compose(
        [Tf.ToTensor(), Tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return Tf.Compose(
            [
                Tf.RandomHorizontalFlip(p=0.5),
                Tf.ColorJitter(0.4, 0.4, 0.4),
                Tf.RandomSelect(
                    Tf.RandomResize(scales, max_size=1333),
                    Tf.Compose(
                        [
                            Tf.RandomResize([400, 500, 600]),
                            Tf.RandomSizeCrop(384, 600),
                            Tf.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val" or image_set == "test":
        return Tf.Compose(
            [
                Tf.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")



def load_vidhoi_text(file_path = 'datasets/hico_object_names.txt') -> list:
    with open(file_path, 'r') as f:
        text_list = json.load(f)
    text_list=[(' '.join(t.strip().split(' ')[-1].split('_'))) for t in text_list]
    return text_list



def make_vcoco_transforms_two_stage(image_set):
    """Two-stage transforms for HOICLIP: [augment, normalize].
    Stage 0 output feeds clip_preprocess; stage 1 normalises for the backbone."""
    normalize = Tf.Compose(
        [Tf.ToTensor(), Tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return [
            Tf.Compose([
                Tf.RandomHorizontalFlip(),
                Tf.ColorJitter(0.4, 0.4, 0.4),
                Tf.RandomSelect(
                    Tf.RandomResize(scales, max_size=1333),
                    Tf.Compose([
                        Tf.RandomResize([400, 500, 600]),
                        Tf.RandomSizeCrop(384, 600),
                        Tf.RandomResize(scales, max_size=1333),
                    ]),
                ),
            ]),
            normalize,
        ]
    if image_set in ("val", "test"):
        return [
            Tf.Compose([Tf.RandomResize([800], max_size=1333)]),
            normalize,
        ]
    raise ValueError(f"unknown {image_set}")


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
    if args.sample_mode == "frame":
        dataset = VidHOI(
            image_set,
            img_folder,
            anno_file,
            transforms=make_vcoco_transforms(image_set),
            train_ratio=args.train_ratio,
            num_queries=args.num_queries,
        )
    elif args.sample_mode in ["clip", "window"]:
        meta_file = root / "VidHOI_annotation" / "meta_data.json"
        # HOICLIP_LSTM 需要多帧序列；普通 HOICLIP 用单帧模式
        use_clips = (getattr(args, 'model_name', '') == 'HOICLIP_LSTM')
        dataset = VidHOI_Clip(
            image_set,
            img_folder,
            anno_file,
            meta_file,
            transforms=make_vcoco_transforms_two_stage(image_set),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            sample_mode=args.sample_mode,
            min_length=args.min_length,
            max_length=args.max_length,
            min_future=args.min_future,
            num_queries=args.num_queries,
            args=args,
            anticipation=use_clips,
        )
    return dataset
