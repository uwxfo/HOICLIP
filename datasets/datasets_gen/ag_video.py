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

import datasets.datasets_gen.transforms_video as T
import datasets.transforms as Tf
from ModifiedCLIP import clip
from datasets.ag_text_label import ag_text_label


class AG(torch.utils.data.Dataset):
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
        self.train_ratio = train_ratio

        self.object_text = [
                    'person',
                    'bag',
                    'bed',
                    'blanket',
                    'book',
                    'box',
                    'broom',
                    'chair',
                    'closet cabinet',
                    'clothes',
                    'cup glass bottle',
                    'dish',
                    'door',
                    'doorknob',
                    'doorway',
                    'floor',
                    'food',
                    'groceries',
                    'laptop',
                    'light',
                    'medicine',
                    'mirror',
                    'paper notebook',
                    'phone camera',
                    'picture',
                    'pillow',
                    'refrigerator',
                    'sandwich',
                    'shelf',
                    'shoe',
                    'sofa couch',
                    'table',
                    'television',
                    'towel',
                    'vacuum',
                    'window'
                ]
        self.verb_text = [
                    'looking at',
                    'not looking at',
                    'unsure',
                    'above',
                    'beneath',
                    'in front of',
                    'behind',
                    'on the side of',
                    'in',
                    'carrying',
                    'covered by',
                    'drinking from',
                    'eating',
                    'have it on the back',
                    'holding',
                    'leaning on',
                    'lying on',
                    'not contacting',
                    'other relationship',
                    'sitting on',
                    'standing on',
                    'touching',
                    'twisting',
                    'wearing',
                    'wiping',
                    'writing on'
                ]
        self._valid_verb_ids=range(len(self.verb_text))

        self.annotations = self.annotations[: int(len(self.annotations) * self.train_ratio)]
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


class AG_Clip():
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
        self.min_future = min_future
        self._transforms = transforms
        self.num_queries = num_queries
        self.anticipation = anticipation
        
        with open(anno_file, "r") as f:
            self.annotations = json.load(f)
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        self.object_text = [
                    'person',
                    'bag',
                    'bed',
                    'blanket',
                    'book',
                    'box',
                    'broom',
                    'chair',
                    'closet cabinet',
                    'clothes',
                    'cup glass bottle',
                    'dish',
                    'door',
                    'doorknob',
                    'doorway',
                    'floor',
                    'food',
                    'groceries',
                    'laptop',
                    'light',
                    'medicine',
                    'mirror',
                    'paper notebook',
                    'phone camera',
                    'picture',
                    'pillow',
                    'refrigerator',
                    'sandwich',
                    'shelf',
                    'shoe',
                    'sofa couch',
                    'table',
                    'television',
                    'towel',
                    'vacuum',
                    'window'
                ]
        self.verb_text = [
                    'looking at',
                    'not looking at',
                    'unsure',
                    'above',
                    'beneath',
                    'in front of',
                    'behind',
                    'on the side of',
                    'in',
                    'carrying',
                    'covered by',
                    'drinking from',
                    'eating',
                    'have it on the back',
                    'holding',
                    'leaning on',
                    'lying on',
                    'not contacting',
                    'other relationship',
                    'sitting on',
                    'standing on',
                    'touching',
                    'twisting',
                    'wearing',
                    'wiping',
                    'writing on'
                ]
        self.verb_prompt = { v: f"a person is {v} someone or something" if v in self.verb_text[0:9] else f"a person is {v} something" for v in self.verb_text}
        self.verb_prompt['unsure'] = "it is unclear whether a person is looking at someone or something"
        self.verb_prompt['other relationship'] = "a person has an unspecified interaction with something"
        self.verb_prompt['have it on the back'] = "a person has something on the back of their body"

        self._valid_verb_ids = list(range(args.num_verb_classes))

        _, self.clip_preprocess = clip.load(args.clip_model)
        # HOI-triplet label space; key order matches AG_annotation/triplet_categories_id.json
        self.text_label_ids = list(ag_text_label.keys())
        self.text_label_to_idx = {pair: i for i, pair in enumerate(self.text_label_ids)}

        if self.img_set == "train":  # train set
            metadata=metadata['train']     # 1 for train, 0 for val
            if self.sample_mode == "clip":
                self.output_list, self.output_start_list = [], []
                for i, meta in enumerate(metadata):
                    frame_ids = meta["frame_ids"]
                    #if meta["video_length"] <= self.min_length + self.min_future:
                    #    pass
                    num_windows = meta["video_length"] // self.max_length
                    if num_windows > 0:
                        residue= meta["video_length"] % self.max_length
                        if residue > num_windows:
                            attached_length = (residue) // num_windows
                            residue = residue - attached_length * num_windows
                        else:
                            attached_length = 0
                        clip_length = [self.max_length + attached_length]*(num_windows-1) + [self.max_length + attached_length + residue]
                        clip_indices = np.cumsum([0]+clip_length)+frame_ids[0]
                        assert clip_indices[-1]-1==frame_ids[-1], f"Invalid clip indices at video {meta['video_id']}"
                        self.output_list.extend((clip_indices[1:]-1).tolist())
                        self.output_start_list.extend(clip_indices[:-1].tolist())
                assert len(self.output_list) == len(self.output_start_list), "Inconsistent train clip indices length"
                    
            elif self.sample_mode == "window":
                self.output_list, self.output_start_list, self.track_obj_list = [], [], []
                for i, meta in enumerate(metadata):
                    num_windows = meta["video_length"] // self.max_length
                    if num_windows > 0:
                        self.output_list.extend([meta["frame_ids"][-1]] * num_windows)
                        self.output_start_list.extend([meta["frame_ids"][0]] * num_windows)
                assert len(self.output_list) == len(self.output_start_list), "Inconsistent train clip indices length"
            else:
                raise ValueError(f"unknown sample mode {self.sample_mode}")
            self.output_list = self.output_list[: int(len(self.output_list) * self.train_ratio)]
        
        else:  # val set
            metadata=metadata['test']
            self.output_list, self.output_start_list, self.output_end_list, self.track_obj_list = [], [], [], []
            for meta in metadata:
                if meta["video_length"] >= self.min_length:
                    self.output_list.extend(meta["frame_ids"][self.min_length-1: meta["video_length"]])
                    self.output_start_list.extend([meta["frame_ids"][0]] * (meta["video_length"] - self.min_length + 1))
                    self.output_end_list.extend([meta["frame_ids"][-1]] * (meta["video_length"] - self.min_length + 1))
            assert len(self.output_list) == len(self.output_start_list) == len(self.output_end_list), "Inconsistent val clip indices length"
            self.output_list = self.output_list[: int(len(self.output_list) * self.val_ratio)]
            self.output_start_list = self.output_start_list[: int(len(self.output_start_list) * self.val_ratio)]
            self.output_end_list = self.output_end_list[: int(len(self.output_end_list) * self.val_ratio)]

        print(f"{self.img_set} set: totally {len(self.output_list)} items, sample_mode: {self.sample_mode}")
    
    def __len__(self):
        return len(self.output_list)

    def __getitem__(self, idx):
        '''
        if self.sample_mode == "clip":
            video_id = self.output_list[idx]["video_id"]
            ann_idx_start = self.output_list[idx]["clip_id"]
            ann_idx_end = ann_idx_start + self.max_length
            frame_ann=self.annotations[video_id]
        '''
        #if self.sample_mode == "window":
        if self.img_set == "train":
            video_length=self.output_list[idx]-self.output_start_list[idx]+1
            if self.anticipation and video_length >= self.min_length + self.min_future:
                frame_idx_end = np.random.randint(self.output_start_list[idx] + self.min_length, self.output_list[idx] - self.min_future + 2)  #pick the end frame, with anticipation
            else:
                frame_idx_end = np.random.randint(self.output_start_list[idx] + self.min_length, self.output_list[idx] + 2)  #pick the end frame id(not included), without anticipation, +1:not include end_idx
            frame_idx_start = max(self.output_start_list[idx], frame_idx_end - self.max_length)
            video_idx_end = self.output_list[idx]
        else:
            frame_idx_end = self.output_list[idx] +1
            frame_idx_start = max(self.output_start_list[idx], frame_idx_end - self.max_length)
            video_idx_end = self.output_end_list[idx]
        clip_anns=self.annotations[frame_idx_start-1 : frame_idx_end -1]   #ann idx start from 0, frame idx start from 1
       
        imgs = []
        pair_list = []
        if self.img_set == "train":
            targets = {"boxes":[], "labels":[],"tracks":[], "sub_boxes":[], "obj_boxes":[], "obj_labels":[], "verb_labels":[], "hoi_labels":[], "clip_inputs":[], "file_names":[], "pair_indices":[]}
            for frame_ann in clip_anns:
                img, target, updated_pair_list = self._load_one_frame(frame_ann, pair_list)
                pair_list = updated_pair_list

                imgs.append(img)
                targets["verb_labels"].append(target["verb_labels"])
                targets["hoi_labels"].append(target["hoi_labels"])
                targets["boxes"].append(target["boxes"])
                targets["labels"].append(target["labels"])
                targets["tracks"].append(target["tracks"])
                targets["pair_indices"].append(target["pair_indices"])
                targets["clip_inputs"].append(target["clip_inputs"])
                targets["file_names"].append(frame_ann["file_name"])
                targets['obj_classes'] = self.object_text
                targets['verb_classes'] = self.verb_text
            
            targets['orig_size'] = torch.as_tensor(imgs[0].size[::-1])  #[h,w]
            targets['id'] = idx

            if self._transforms is not None:
                imgs, targets = self._transforms(imgs, targets)
            imgs = torch.cat(imgs, dim=0)
            #targets['size'] =  torch.as_tensor(imgs[0].shape[-2:])

            num_pairs = len(pair_list)
            # Keep a stable 2-D shape even when there is no human-object pair in this clip.
            ho_pairs = torch.as_tensor(pair_list, dtype=torch.int64).reshape(-1, 2)
            '''
            tracks_removed=targets.get("tracks_removed", [])
            if len(tracks_removed)>0:
                h_index=torch.isin(ho_pairs[:,0], tracks_removed)
                o_index=torch.isin(ho_pairs[:,1], tracks_removed)
                idx=(~h_index) & (~o_index)
                
                if len(idx)>0:
                    print(f"Removed {len(idx)} pairs with removed tracks at clip from {targets['file_names'][0]} to {targets['file_names'][-1]}")         
            '''
            box_non_valid=torch.zeros((1,4))
            label_non_valid=torch.full((1,),78, dtype=torch.int64)
            verbs_one_frame=torch.zeros((num_pairs,len(self._valid_verb_ids)))
            hois_one_frame=torch.zeros((num_pairs, len(self.text_label_ids)))
            valid_one_frame=torch.zeros(num_pairs, dtype=torch.int64)
            sub_boxes_list, obj_boxes_list, obj_labels_list, verb_labels_list, hoi_labels_list, valid_list=[], [], [], [], [], []
            for i, (box, label, track, verb_label, hoi_label, pair_idx) in enumerate(zip(targets["boxes"], targets["labels"], targets["tracks"], targets["verb_labels"], targets["hoi_labels"], targets["pair_indices"])):
                # Size track map by the max track id seen either in clip-level pairs or current frame.
                # This also handles empty ho_pairs/track tensors safely.
                max_pair_track = ho_pairs.max().item() if ho_pairs.numel() > 0 else -1
                max_frame_track = track.max().item() if track.numel() > 0 else -1
                track_map_size = max(max_pair_track, max_frame_track) + 1
                track_map = torch.full((track_map_size,), -1, dtype=torch.int64)
                try:
                    track_map[track]=torch.arange(len(track))
                except IndexError:
                    pass
                sub_indices=track_map[ho_pairs[:,0]]
                obj_indices=track_map[ho_pairs[:,1]]
                box=torch.cat((box, box_non_valid), dim=0)
                label=torch.cat((label, label_non_valid), dim=0)
                sub_boxes_list.append(box[sub_indices])
                obj_boxes_list.append(box[obj_indices])
                obj_labels_list.append(label[obj_indices])

                if "tracks_removed" in targets:
                    tracks_removed=targets["tracks_removed"][i]
                    if len(tracks_removed)>0:
                        h_index=torch.isin(ho_pairs[:,0], tracks_removed)
                        o_index=torch.isin(ho_pairs[:,1], tracks_removed)
                        removed_idx=((~h_index) & (~o_index))[:len(pair_idx)]
                        if len(removed_idx)>0:
                            pair_idx=pair_idx[removed_idx]
                            verb_label=verb_label[removed_idx]
                            hoi_label=hoi_label[removed_idx]
                            print(f"Removed {len(pair_list)-removed_idx.sum().item()} / {len(pair_list)} pairs at clip {idx}, frame {targets['file_names'][i]}")
                verb_labels_list.append(verbs_one_frame.index_copy(0, pair_idx, verb_label) if len(pair_idx) > 0 else verbs_one_frame)
                hoi_labels_list.append(hois_one_frame.index_copy(0, pair_idx, hoi_label) if len(pair_idx) > 0 else hois_one_frame)
                valid_list.append(valid_one_frame.index_copy(0, pair_idx, torch.ones_like(pair_idx, dtype=torch.int64)) if len(pair_idx) > 0 else valid_one_frame)

            targets["sub_boxes"]=torch.cat(sub_boxes_list)
            targets["obj_boxes"]=torch.cat(obj_boxes_list)
            targets["verb_labels"]=torch.cat(verb_labels_list)
            targets["hoi_labels"]=torch.cat(hoi_labels_list)
            targets["valid"] = torch.stack(valid_list)
            
            obj_labels_list=torch.stack(obj_labels_list, dim=1)
            for i, label in enumerate(obj_labels_list):
                obj_labels_list[i]=torch.full_like(label, label.min())
            targets["obj_labels"]=torch.flatten(obj_labels_list.transpose(0,1))
            targets['sub_labels'] = torch.zeros_like(targets["obj_labels"], dtype=torch.int64)
            targets["clip_inputs"] = torch.stack(targets["clip_inputs"])  # (T, 3, 224, 224)
            targets["filename"] = targets["file_names"][-1]

        else:       # val set
            targets = {"boxes":[], "labels":[], "tracks":[], "hois":[], "clip_inputs":[], "file_names":[], "pair_indices":[], "valid": []}
            for i, frame_ann in enumerate(clip_anns):
                img, target, updated_pair_list = self._load_one_frame(frame_ann, pair_list)
                pair_list = updated_pair_list

                imgs.append(img)
                targets["boxes"].append(target["boxes"])
                targets["labels"].append(target["labels"])
                targets["tracks"].append(target["tracks"])
                targets["hois"].append(target["hois"])
                targets["clip_inputs"].append(target["clip_inputs"])
                targets["pair_indices"].append(target["pair_indices"])
                targets["file_names"].append(frame_ann["file_name"])
            targets["obj_classes"] = self.object_text
            targets["verb_classes"] = self.verb_text

            targets["orig_size"] = torch.as_tensor(img.size[::-1])     #[h,w]
            targets["id"] = idx
            
            for pair_indices in targets["pair_indices"]:
                targets["valid"].append([1 if i in pair_indices else 0 for i in range(len(pair_list))])
            targets["valid"] = torch.as_tensor(targets["valid"], dtype=torch.int64)
            targets["clip_inputs"] = torch.stack(targets["clip_inputs"])  # (T, 3, 224, 224)
            targets["filename"] = targets["file_names"][-1]

            # Evaluator needs single-frame tensors for the last (current) frame.
            targets["boxes"]  = targets["boxes"][-1]   # (N, 4)
            targets["labels"] = targets["labels"][-1]  # (N,)
            last_hois = targets["hois"][-1]
            if last_hois:
                targets["hois"] = torch.as_tensor(
                    [(h["subject_id"], h["object_id"], h["category_id"]) for h in last_hois],
                    dtype=torch.int64,
                )
            else:
                targets["hois"] = torch.zeros((0, 3), dtype=torch.int64)

            if self._transforms is not None:
                imgs, _ = self._transforms(imgs, targets)
            imgs = torch.cat(imgs, dim=0)

        
        if not self.anticipation:
            return imgs, targets


        # anticipation
        futures=[1, 3, 5, 7]
        if self.img_set == "train":
            anticipations={"future": [], "verb_labels_future": [], "hoi_labels_future": [], "valid_future": []}
            for future in futures:
                frame_idx_future = frame_idx_end+future-1   # frame_id for current future horizon
                if frame_idx_end + future -1 > video_idx_end:
                    break

                _, target, _ = self._load_one_frame(self.annotations[frame_idx_future-1], pair_list, anticipation=True)

                pair_idx=target["pair_indices"]

                anticipations["future"].append(future)
                if len(pair_idx) > 0:
                    anticipations["verb_labels_future"].append(verbs_one_frame.index_copy(0, pair_idx, target["verb_labels"]))
                    anticipations["hoi_labels_future"].append(hois_one_frame.index_copy(0, pair_idx, target["hoi_labels"]))
                    anticipations["valid_future"].append(valid_one_frame.index_copy(0, pair_idx, torch.ones_like(pair_idx, dtype=torch.int64)))
                else:
                    anticipations["verb_labels_future"].append(verbs_one_frame)
                    anticipations["hoi_labels_future"].append(hois_one_frame)
                    anticipations["valid_future"].append(valid_one_frame)

            if anticipations["future"]:
                anticipations["verb_labels_future"]=torch.cat(anticipations["verb_labels_future"])
                anticipations["hoi_labels_future"]=torch.cat(anticipations["hoi_labels_future"])
                anticipations["valid_future"]=torch.stack(anticipations["valid_future"])
                targets.update(anticipations)
        
        else:  #  val set
            '''anticipations = {"future": [], "boxes_future": [], "labels_future": [], "hois_future": []}

            track_future=targets["tracks"][-1]
            track_union=set().union(*[set(t) for t in targets["tracks"]])
            track_diff=list(track_union-set(track_future))
            
            track_map=[torch.tensor(t, dtype=torch.int64) for t in targets["tracks"]]
            track_map=pad_sequence(track_map, batch_first=True, padding_value=-1)  #batch_first: put batch dim first
            
            boxes_future, labels_future = [], []
            for s in track_diff:
                frame_ids, annots_ids=torch.where(track_map==s)
                frame_id, max_id=frame_ids.max(dim=0)
                frame_id=frame_id.item()
                annot_id=annots_ids[max_id].item()
                
                boxes_future.append(torch.zeros((1,4)))
                labels_future.append(targets["labels"][frame_id][[annot_id]])
            track_future = track_future + track_diff'''

            anticipations={"future": [], "hois_future": []}
            pair_indices=targets["pair_indices"][-1]
            pair_list=[pair_list[i] for i in pair_indices]
            track_future=targets["tracks"][-1]
            for future in futures:
                frame_idx_future = frame_idx_end+future-1   # frame_id for current future horizon
                if frame_idx_end + future -1 > video_idx_end:
                    break
                
                _, target, _ = self._load_one_frame(self.annotations[frame_idx_future-1], pair_list, anticipation=True)

                hois_future=[]
                for s,o,v in target["hois"]:
                    if s in track_future and o in track_future:
                        hois_future.append(
                            {
                                "subject_id": track_future.index(s),
                                "category_id": v,
                                "object_id": track_future.index(o),
                        
                            }
                        )
                # Preserve empty-GT horizons so future false positives remain evaluable.
                anticipations["future"].append(future)
                anticipations["hois_future"].append(hois_future)

            if anticipations["future"]:
                targets.update(anticipations)
                

        expected_channels = self.max_length * 3
        assert len(imgs) == expected_channels, (
            f"Invalid clip tensor shape {imgs.shape}; expected {expected_channels} channels "
            f"for max_length={self.max_length} at item {idx}, "
            f"frame {self.annotations[frame_idx_start-1]['file_name']} "
            f"to {self.annotations[frame_idx_end-2]['file_name']}"
        )
        #print(f"{imgs.shape} with idx {idx}")
        return imgs, targets
        

    def _load_one_frame(self, frame_ann, pair_list, anticipation=False):
        if not anticipation:
            img = open_with_retries(self.img_folder / frame_ann["file_name"])
            w, h = img.size
        else:
            img = None
        #print(frame_ann["file_name"])
    
        target = {}
        boxes, classes, tracks = [], [], []
        for obj in frame_ann["annotations"]:
            boxes.append(obj["bbox"])
            classes.append(obj["category_id"])
            tracks.append(obj["category_id"])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)   # guard against no boxes via resizing: when no boxes, also return tensor(0,4)
        assert (boxes[:, 3] > boxes[:, 1]).all() & (boxes[:, 2] > boxes[:, 0]).all(), f"Invalid boxes in frame {frame_ann['file_name']}"
        #boxes[:, 0:4:2].clamp_(min=0, max=w)
        #boxes[:, 1::2].clamp_(min=0, max=h) 
        
        if self.img_set == "train":
            if len(frame_ann["annotations"]) > self.num_queries:
                boxes = boxes[: self.num_queries]
                classes = classes[: self.num_queries]
                tracks = tracks[: self.num_queries]
            
            verb_labels = []
            hoi_labels = []
            sub_obj_pairs = []
            pair_indices = []
            for hoi in frame_ann["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                assert sub_id < len(tracks) and obj_id < len(tracks), f"Invalid HOI annotation in frame {frame_ann['file_name']}"
                sub_obj_pair = (tracks[sub_id], tracks[obj_id])
                # (verb, object) -> triplet class index; None for combinations absent from
                # AG_annotation/triplet_categories_id.json (should not happen: it was built
                # from train+test, but stay defensive rather than crash on an unseen pair).
                hoi_idx = self.text_label_to_idx.get((hoi["category_id"], classes[obj_id]))

                if sub_obj_pair in sub_obj_pairs:
                    i = sub_obj_pairs.index(sub_obj_pair)
                    verb_labels[i][self._valid_verb_ids.index(hoi["category_id"])] = 1
                    if hoi_idx is not None:
                        hoi_labels[i][hoi_idx] = 1
                else:
                    if sub_obj_pair not in pair_list:
                        if anticipation:
                            continue
                        pair_list.append(sub_obj_pair)
                    verb_label = [0] * len(self._valid_verb_ids)
                    verb_label[self._valid_verb_ids.index(hoi["category_id"])] = 1
                    hoi_label = [0] * len(self.text_label_ids)
                    if hoi_idx is not None:
                        hoi_label[hoi_idx] = 1
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_obj_pairs.append(sub_obj_pair)
                    pair_indices.append(pair_list.index(sub_obj_pair))
            target["boxes"] = boxes
            target["labels"] = torch.as_tensor(classes, dtype=torch.int64)
            target["tracks"] = torch.as_tensor(tracks, dtype=torch.int64)
            target["pair_indices"] = torch.as_tensor(pair_indices, dtype=torch.int64) if pair_indices else pair_indices
            target["verb_labels"] = torch.as_tensor(verb_labels, dtype=torch.float32) if verb_labels else verb_labels
            target["hoi_labels"] = (torch.as_tensor(hoi_labels, dtype=torch.float32) if hoi_labels
                                    else torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32))
            # CLIP preprocessing runs on the raw PIL image, before the geometric transforms.
            target["clip_inputs"] = self.clip_preprocess(img) if not anticipation else None

        else:   # val set
            hois = []
            pair_indices = []
            for hoi in frame_ann["hoi_annotation"]:
                sub_id = hoi["subject_id"]
                obj_id = hoi["object_id"]
                assert sub_id < len(tracks) and obj_id < len(tracks), f"Invalid HOI annotation in frame {frame_ann['file_name']}"
                sub_obj_pair = (tracks[sub_id], tracks[obj_id])
                    
                if anticipation:
                    if sub_obj_pair in pair_list:
                        hois.append(
                        (
                            sub_obj_pair[0],
                            sub_obj_pair[1],
                            hoi["category_id"],
                        ))
                else:
                    if sub_obj_pair not in pair_list:
                        pair_list.append(sub_obj_pair)
                    hois.append(hoi)
                    '''(
                        sub_id,
                        obj_id,
                        hoi["category_id"],
                    ))'''
                    pair_indices.append(pair_list.index(sub_obj_pair))
            target["boxes"] = boxes
            target["labels"]= torch.as_tensor(classes, dtype=torch.int64)
            target["tracks"] = tracks
            target["pair_indices"] = pair_indices
            target["hois"] = hois
            target["clip_inputs"] = self.clip_preprocess(img) if not anticipation else None


        return img, target, pair_list


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



def load_vidhoi_text(file_path) -> list:
    text_names = []
    for line in open(file_path,'r'):
        text_names.append(' '.join(line.strip().split('_')))
    return text_names



def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f"provided HOI path {root} does not exist"
    PATHS = {
        "train": (
            root / "images",
            root / "AG_annotation" / "train_frame_hoia.json",
        ),
        "val": (
            root / "images",
            root / "AG_annotation" / "test_frame_hoia.json",
        ),
    }
    img_folder, anno_file = PATHS[image_set]
    if args.sample_mode == "frame":   #image based
        dataset = AG(
            image_set,
            img_folder,
            anno_file,
            transforms=make_vcoco_transforms(image_set),
            train_ratio=args.train_ratio,
            num_queries=args.num_queries,
        )
    elif args.sample_mode in ["clip", "window"]:  #video based
        meta_file = root / "AG_annotation" / "meta_data.json"
        dataset = AG_Clip(
            image_set,
            img_folder,
            anno_file,
            meta_file,
            transforms=make_vidhoi_transforms(image_set),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            sample_mode=args.sample_mode,
            min_length=args.min_length,
            max_length=args.max_length,
            min_future=args.min_future,
            num_queries=args.num_queries,
            args=args,
            anticipation=args.anticipation,
        )
    return dataset
