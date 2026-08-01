# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_vidhoi_triplet_label_ids(hoi_path):
    # VidHOI and AG share this file layout, only the annotation directory differs.
    for dirname in ("VidHOI_annotation", "AG_annotation"):
        triplet_file = os.path.join(hoi_path, dirname, "triplet_categories_id.json")
        if os.path.exists(triplet_file):
            break
    else:
        raise FileNotFoundError(
            f"No <VidHOI|AG>_annotation/triplet_categories_id.json under {hoi_path}")
    with open(triplet_file, "r", encoding="utf-8-sig") as f:
        triplets = json.load(f)
    triplet_ids=[]
    for triplet in triplets:
        triplet_ids.append((triplet[0], triplet[2], triplet[1]))
    return triplet_ids


class VidHOIEvaluator:
    """Triplet-level VidHOI evaluator for the current SL-HOI output contract."""

    def __init__(
        self,
        preds,
        gts,
        hoi_path,
        use_nms_filter=False,
        thres_nms=0.5,
        nms_alpha=1.0,
        nms_beta=0.0,
        output_dir="",
    ):
        self.overlap_iou = 0.5
        self.max_hois = 100
        self.rare_limit = 25
        self.human_top_ks = [10, 20, 50, 100]
        self.human_score_threshold = 0.0
        self.output_dir = Path(output_dir) if output_dir else None

        self.use_nms_filter = use_nms_filter
        self.thres_nms = thres_nms
        self.nms_alpha = nms_alpha
        self.nms_beta = nms_beta
        self.triplet_eval = True
        self.triplet_label_ids = load_vidhoi_triplet_label_ids(hoi_path)
        self.triplet_label_to_idx = {
            triplet: i for i, triplet in enumerate(self.triplet_label_ids)
        }
        # triplet_label_ids[i] = (sub, obj, verb) -> object category each triplet scores
        # against, so obj_scores can be broadcast onto the triplet axis like HICOEvaluator.
        self.hoi_obj_list = [triplet[1] for triplet in self.triplet_label_ids]
        print("Use NMS: ", self.use_nms_filter)

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)

        self.preds = []
        for img_preds in preds:
            img_preds = {
                k: v.to("cpu").numpy() if hasattr(v, "to") else v
                for k, v in img_preds.items()
            }
            bboxes = [
                {"bbox": np.asarray(bbox), "category_id": int(label)}
                for bbox, label in zip(img_preds["boxes"], img_preds["labels"])
            ]
            hoi_scores = np.asarray(img_preds["hoi_scores"])
            if hoi_scores.ndim != 2:
                raise ValueError(
                    f"Expected hoi_scores with shape [queries, triplets], got {hoi_scores.shape}"
                )
            if hoi_scores.shape[1] != len(self.triplet_label_ids):
                raise ValueError(
                    "hoi_scores class dimension does not match "
                    f"triplet_categories_id.json: {hoi_scores.shape[1]} != "
                    f"{len(self.triplet_label_ids)}"
                )
            # Fold object confidence into the triplet score, mirroring HICOEvaluator
            # (datasets/hico_eval_triplet.py): square obj_scores, then broadcast each
            # triplet's target-object column onto the [Q, N_triplet] hoi_scores. This runs
            # after any transition-matrix propagation (evaluate_vidhoi_transition only
            # overwrites the 'hoi_scores' key), so noise-OR always acts on the raw
            # (un-combined) HOI-triplet score.
            if hoi_scores.shape[0] > 0:
                obj_scores = np.asarray(img_preds["obj_scores"])
                obj_scores = obj_scores * obj_scores
                hoi_scores = hoi_scores + obj_scores[:, self.hoi_obj_list]
            if hoi_scores.shape[0] == 0:
                self.preds.append(
                    {"predictions": bboxes, "hoi_prediction": []}
                )
                continue

            flat_scores = hoi_scores.reshape(-1)
            num_to_keep = min(self.max_hois, flat_scores.size)
            if num_to_keep < flat_scores.size:
                top_indices = np.argpartition(
                    flat_scores, -num_to_keep
                )[-num_to_keep:]
                top_indices = top_indices[
                    np.argsort(-flat_scores[top_indices])
                ]
            else:
                top_indices = np.argsort(-flat_scores)
            num_triplets = hoi_scores.shape[1]
            hois = [
                {
                    "subject_id": int(img_preds["sub_ids"][index // num_triplets]),
                    "object_id": int(img_preds["obj_ids"][index // num_triplets]),
                    "category_id": int(index % num_triplets),
                    "score": float(flat_scores[index]),
                }
                for index in top_indices
            ]

            if self.use_nms_filter:
                pred_triplets = {}
                keep_indices = []
                for i, hoi in enumerate(hois):
                    triplet = self.triplet_label_ids[hoi["category_id"]]
                    if triplet not in pred_triplets:
                        pred_triplets[triplet] = {
                            "subs": [],
                            "objs": [],
                            "scores": [],
                            "indexes": [],
                        }
                    pred_triplets[triplet]["subs"].append(
                        bboxes[hoi["subject_id"]]["bbox"]
                    )
                    pred_triplets[triplet]["objs"].append(
                        bboxes[hoi["object_id"]]["bbox"]
                    )
                    pred_triplets[triplet]["scores"].append(hoi["score"])
                    pred_triplets[triplet]["indexes"].append(i)

                for values in pred_triplets.values():
                    indices = self.pairwise_nms(
                        np.asarray(values["subs"]),
                        np.asarray(values["objs"]),
                        np.asarray(values["scores"]),
                    )
                    indices = list(np.asarray(values["indexes"])[indices])
                    keep_indices.extend(indices)
                hois = [hois[i] for i in keep_indices]

            self.preds.append(
                {"predictions": bboxes, "hoi_prediction": hois}
            )

        self.gt_triplets = []
        self.gts = []
        for img_gts in gts:
            boxes = self.to_numpy(img_gts["boxes"])
            labels = self.to_numpy(img_gts["labels"])
            hoi_annotation = self.normalize_hoi_annotation(img_gts.get("hois", []))
            annotations = [
                {"bbox": np.asarray(bbox), "category_id": int(label)}
                for bbox, label in zip(boxes, labels)
            ]
            normalized_hois = []

            for hoi in hoi_annotation:
                sub_id = int(hoi["subject_id"])
                obj_id = int(hoi["object_id"])
                if sub_id < 0 or obj_id < 0 or sub_id >= len(labels) or obj_id >= len(labels):
                    continue
                triplet_key = (
                    int(labels[sub_id]),
                    int(labels[obj_id]),
                    int(hoi["category_id"]),
                )
                if triplet_key not in self.triplet_label_to_idx:
                    continue
                triplet = self.triplet_label_to_idx[triplet_key]
                normalized_hois.append(
                    {
                        "subject_id": sub_id,
                        "object_id": obj_id,
                        "category_id": triplet,
                    }
                )
                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)
                self.sum_gts[triplet] += 1

            self.gts.append(
                {
                    "annotations": annotations,
                    "hoi_annotation": normalized_hois,
                }
            )

        self.set_rare_hois(self.rare_limit)

    @staticmethod
    def to_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().to("cpu").numpy()
        return np.asarray(value)

    def normalize_hoi_annotation(self, hois):
        # Convert frame-mode tensor/list triples and clip-mode dicts into one dict format.
        if hois is None:
            return []
        if len(hois) == 0:
            return []
        normalized = []
        for hoi in hois:
            if hasattr(hoi, 'tolist'):
                hoi = hoi.tolist()
            if isinstance(hoi, dict):
                normalized.append({
                    'subject_id': int(hoi['subject_id']),
                    'object_id': int(hoi['object_id']),
                    'category_id': int(hoi['category_id']),
                })
            elif isinstance(hoi, (list, tuple)) and len(hoi) >= 3:
                normalized.append({
                    'subject_id': int(hoi[0]),
                    'object_id': int(hoi[1]),
                    'category_id': int(hoi[2]),
                })
            else:
                continue
        return normalized
    

    def set_rare_hois(self, rare_limit):
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet in self.gt_triplets:
            num = self.sum_gts[triplet]
            if num < rare_limit:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)
    
    def pairwise_nms(self, subs, objs, scores):
        '''
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter/sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]   # return (inds)

            order = order[inds + 1]
        '''
        order = scores.argsort()[::-1]
        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)
            sub_iou=self.compute_IOU(subs[i], subs[order[1:]])
            obj_iou=self.compute_IOU(objs[i], objs[order[1:]])
            ovr = np.power(sub_iou, self.nms_alpha) * np.power(obj_iou, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]
            order = order[inds + 1]
        return keep_inds


    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) > 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    if self.triplet_eval:
                        triplet = pred_hoi['category_id']
                    else:
                        triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'],
                                   pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        human_centric_recall = self.compute_human_centric_recall_k()
        map.update(human_centric_recall)
        return {key: float(value) for key, value in map.items()}

    def compute_map(self, save_all_APs=True):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        if save_all_APs:
            ap=dict(sorted(ap.items(), key=lambda k: k[1]))
            rare_ap=dict(sorted(rare_ap.items(), key=lambda k: k[1]))
            non_rare_ap=dict(sorted(non_rare_ap.items(), key=lambda k: k[1]))
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                ap_files = {
                    "ap_list.json": ap,
                    "rare_ap_list.json": rare_ap,
                    "non_rare_ap_list.json": non_rare_ap,
                }
                for filename, values in ap_files.items():
                    with (self.output_dir / filename).open("w") as f:
                        json.dump(
                            {str(k): float(v) for k, v in values.items()},
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )
        m_ap = np.mean(list(ap.values())) if ap else 0.0
        m_ap_rare = np.mean(list(rare_ap.values())) if rare_ap else 0.0
        m_ap_non_rare = np.mean(list(non_rare_ap.values())) if non_rare_ap else 0.0
        m_max_recall = np.mean(list(max_recall.values())) if max_recall else 0.0

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP_rare': m_ap_rare, 'mAP_non_rare': m_ap_non_rare, 'mean_max_recall': m_max_recall}

    def voc_ap(self, rec, prec, use_07_metric=True):
        if use_07_metric:              # VOC 2007 11-point
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:                         # VOC 2012 P-R curve method
            # first append sentinel values at the end
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def compute_human_centric_recall_k(self):
        """
        Compute recall@k and recall_h@k in a human-centric way.

        Data structures used in this function:
        - self.preds: list[dict], one item per frame/clip. Each dict has:
          - 'predictions': list[dict], each box dict = {'bbox': [x1, y1, x2, y2], 'category_id': int}
          - 'hoi_prediction': list[dict], each hoi dict =
            {'subject_id': int, 'object_id': int, 'category_id': int, 'score': float}
        - self.gts: list[dict], aligned with self.preds. Each dict has:
          - 'annotations': list[dict], same box dict format as predictions
          - 'hoi_annotation': list[dict], same hoi format without score

        Metrics produced:
        - recall@k: frame-level recall over all GT interactions in a frame/clip.
        - recall_h@k: human-level recall, computed separately for each GT human.

        Matching rule:
        - A predicted pair matches a GT pair when subject and object boxes both match
          GT boxes with IoU >= self.overlap_iou and category ids are consistent
          (enforced by compute_iou_mat category-aware matching).
        """
        # Collect per-sample recalls first, then average at the end.
        recall_frame_top_all = [[] for _ in self.human_top_ks]
        recall_human_top_all = [[] for _ in self.human_top_ks]
        num_gt_human = 0
        num_pred_human = 0
        gt_triplet_set = set(self.gt_triplets)

        for img_preds, img_gts in zip(self.preds, self.gts):
            # Predicted boxes/hois for current sample.
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            # GT HOI annotations for current sample.
            gt_hois = img_gts['hoi_annotation']

            # No GT HOI -> this sample does not contribute to recall metrics.
            if len(gt_hois) == 0:
                continue

            # Category ids are indices into triplet_categories_id.json in triplet mode.
            filtered_pred_hois = []
            for hoi in pred_hois:
                subj_idx = int(hoi['subject_id'])
                obj_idx = int(hoi['object_id'])
                if subj_idx < 0 or obj_idx < 0:
                    continue
                if subj_idx >= len(pred_bboxes) or obj_idx >= len(pred_bboxes):
                    continue
                triplet = int(hoi['category_id'])
                if triplet in gt_triplet_set:
                    filtered_pred_hois.append(hoi)
            pred_hois = filtered_pred_hois

            # Build a dense verb axis [0..num_verbs-1] large enough for both GT and predictions.
            num_verbs = 1 + max(
                [hoi['category_id'] for hoi in gt_hois] +
                [hoi['category_id'] for hoi in pred_hois]
            ) if (len(gt_hois) > 0 or len(pred_hois) > 0) else 0
            if num_verbs == 0:
                continue

            # GT pair list and its multi-hot interaction matrix:
            # pair_idxes_gt: [N_gt_pair, 2], interaction_distribution_gt: [N_gt_pair, V]
            pair_idxes_gt, interaction_distribution_gt = self.build_gt_pair_interactions(
                gt_hois,
                num_verbs,
            )
            if len(pair_idxes_gt) == 0:
                continue

            # Build unique predicted (subject_id, object_id) pair list from hoi_prediction.
            pair_to_idx_pred = {}
            pair_idxes = []
            for hoi in pred_hois:
                subj_idx = int(hoi['subject_id'])
                obj_idx = int(hoi['object_id'])
                # Skip invalid indices.
                if subj_idx < 0 or obj_idx < 0:
                    continue
                if subj_idx >= len(pred_bboxes) or obj_idx >= len(pred_bboxes):
                    continue
                pair = (subj_idx, obj_idx)
                if pair not in pair_to_idx_pred:
                    pair_to_idx_pred[pair] = len(pair_idxes)
                    pair_idxes.append([subj_idx, obj_idx])

            # Convert hoi list into pair-by-verb score matrix:
            # triplet_scores shape [N_pred_pair, V], keep max score per (pair, verb).
            triplet_scores = np.zeros((len(pair_idxes), num_verbs), dtype=np.float32)
            for hoi in pred_hois:
                subj_idx = int(hoi['subject_id'])
                obj_idx = int(hoi['object_id'])
                verb_idx = int(hoi['category_id'])
                score = float(hoi.get('score', 0.0))
                pair = (subj_idx, obj_idx)
                idx_pair = pair_to_idx_pred.get(pair, None)
                if idx_pair is None:
                    continue
                if verb_idx < 0 or verb_idx >= num_verbs:
                    continue
                if score > triplet_scores[idx_pair, verb_idx]:
                    triplet_scores[idx_pair, verb_idx] = score

            pair_idxes = np.asarray(pair_idxes, dtype=np.int64)
            pair_idxes_gt = np.asarray(pair_idxes_gt, dtype=np.int64)
            interaction_distribution_gt = np.asarray(interaction_distribution_gt, dtype=np.float32)

            # Total number of positive GT interactions in this sample (frame-level denominator).
            gt_count_frame = float(np.sum(interaction_distribution_gt))
            if gt_count_frame <= 0:
                continue

            gt_human_idxes = sorted({int(pair_idx_gt[0]) for pair_idx_gt in pair_idxes_gt})
            num_gt_human += len(gt_human_idxes)

            # If no predicted pair exists, frame-level and human-level recall are all zero.
            if len(pair_idxes) == 0:
                for idx_k in range(len(self.human_top_ks)):
                    recall_frame_top_all[idx_k].append(0.0)
                    for _ in gt_human_idxes:
                        recall_human_top_all[idx_k].append(0.0)
                continue

            # Match predicted boxes to GT boxes using category-aware IoU matrix.
            gt_bboxes = img_gts['annotations']
            bbox_pred_gt_map, bbox_pred_gt_overlap = self.compute_iou_mat(gt_bboxes, pred_bboxes)

            # pred_to_gt_overlap_map[pred_box_idx][gt_box_idx] = iou
            pred_to_gt_overlap_map = {}
            for pred_idx, gt_idxes in bbox_pred_gt_map.items():
                pred_to_gt_overlap_map[pred_idx] = {
                    gt_idx: overlap
                    for gt_idx, overlap in zip(gt_idxes, bbox_pred_gt_overlap[pred_idx])
                }

            # interaction_gts has same shape as triplet_scores and stores matched GT labels.
            # pair_to_gt_pair stores one-to-one pair matching index, -1 means unmatched.
            interaction_gts = np.zeros_like(triplet_scores, dtype=np.float32)
            pair_to_gt_pair = np.full(len(pair_idxes), -1, dtype=np.int64)
            matched_gt_pair = set()

            # Greedy pair matching: each GT pair can be claimed by at most one predicted pair.
            for idx_pair, pair_idx_pred in enumerate(pair_idxes):
                subj_idx_pred = int(pair_idx_pred[0])
                obj_idx_pred = int(pair_idx_pred[1])
                if subj_idx_pred not in pred_to_gt_overlap_map or obj_idx_pred not in pred_to_gt_overlap_map:
                    continue

                subj_matches = pred_to_gt_overlap_map[subj_idx_pred]
                obj_matches = pred_to_gt_overlap_map[obj_idx_pred]
                max_idx_pair_gt = -1
                max_iou = 0.0
                for idx_pair_gt, pair_idx_gt in enumerate(pair_idxes_gt):
                    subj_idx_gt = int(pair_idx_gt[0])
                    obj_idx_gt = int(pair_idx_gt[1])
                    subj_iou = subj_matches.get(subj_idx_gt, -1.0)
                    obj_iou = obj_matches.get(obj_idx_gt, -1.0)
                    if subj_iou < 0 or obj_iou < 0:
                        continue
                    min_iou_pair = min(subj_iou, obj_iou)
                    if min_iou_pair > max_iou:
                        max_iou = min_iou_pair
                        max_idx_pair_gt = idx_pair_gt

                if max_idx_pair_gt >= 0 and max_idx_pair_gt not in matched_gt_pair:
                    matched_gt_pair.add(max_idx_pair_gt)
                    pair_to_gt_pair[idx_pair] = max_idx_pair_gt
                    interaction_gts[idx_pair] = interaction_distribution_gt[max_idx_pair_gt]

            # Frame-level recall@k: flatten all (pair, verb) scores and count top-k true positives.
            sorted_idx_frame = np.argsort(-triplet_scores.reshape(-1))
            sorted_gts_frame = interaction_gts.reshape(-1)[sorted_idx_frame]
            for idx_k, kk in enumerate(self.human_top_ks):
                true_positive_topk = float(np.sum(sorted_gts_frame[:kk]))
                recall_frame_top_all[idx_k].append(true_positive_topk / gt_count_frame)

            # Map predicted human boxes to GT human boxes (best IoU match among GT humans).
            gt_human_set = set(gt_human_idxes)
            pred_human_to_gt_human = {}
            for pred_human_idx in np.unique(pair_idxes[:, 0]):
                pred_human_idx = int(pred_human_idx)
                if pred_human_idx not in pred_to_gt_overlap_map:
                    continue
                max_gt_human_idx = -1
                max_human_iou = 0.0
                for gt_idx, iou in pred_to_gt_overlap_map[pred_human_idx].items():
                    if gt_idx in gt_human_set and iou > max_human_iou:
                        max_human_iou = iou
                        max_gt_human_idx = gt_idx
                if max_gt_human_idx >= 0:
                    pred_human_to_gt_human[pred_human_idx] = max_gt_human_idx
            num_pred_human += len(pred_human_to_gt_human)

            # Group predicted pair indices by matched GT human id.
            gt_human_to_pair_idxes_pred = {human_idx: [] for human_idx in gt_human_idxes}
            for idx_pair, pair_idx_pred in enumerate(pair_idxes):
                pred_human_idx = int(pair_idx_pred[0])
                matched_gt_human_idx = pred_human_to_gt_human.get(pred_human_idx, None)
                if matched_gt_human_idx is not None:
                    gt_human_to_pair_idxes_pred[matched_gt_human_idx].append(idx_pair)

            # Human-level recall_h@k: compute recall independently for each GT human.
            for gt_human_idx in gt_human_idxes:
                gt_pair_mask = pair_idxes_gt[:, 0] == gt_human_idx
                gt_count_human = float(np.sum(interaction_distribution_gt[gt_pair_mask]))
                if gt_count_human <= 0:
                    continue

                pred_pair_idxes_human = gt_human_to_pair_idxes_pred.get(gt_human_idx, [])
                if len(pred_pair_idxes_human) == 0:
                    for idx_k in range(len(self.human_top_ks)):
                        recall_human_top_all[idx_k].append(0.0)
                    continue

                pred_pair_idxes_human = np.asarray(pred_pair_idxes_human, dtype=np.int64)
                triplet_scores_human = triplet_scores[pred_pair_idxes_human].reshape(-1)
                interaction_gts_human = np.zeros((len(pred_pair_idxes_human), num_verbs), dtype=np.float32)
                for idx_local, idx_pair_pred in enumerate(pred_pair_idxes_human):
                    idx_pair_gt = pair_to_gt_pair[idx_pair_pred]
                    if idx_pair_gt >= 0 and int(pair_idxes_gt[idx_pair_gt][0]) == gt_human_idx:
                        interaction_gts_human[idx_local] = interaction_distribution_gt[idx_pair_gt]

                sorted_idx_human = np.argsort(-triplet_scores_human)
                sorted_gts_human = interaction_gts_human.reshape(-1)[sorted_idx_human]
                for idx_k, kk in enumerate(self.human_top_ks):
                    true_positive_topk = float(np.sum(sorted_gts_human[:kk]))
                    recall_human_top_all[idx_k].append(true_positive_topk / gt_count_human)

        # Aggregate by mean over samples/humans.
        metrics = {}
        for idx_k, kk in enumerate(self.human_top_ks):
            values_frame = np.array(recall_frame_top_all[idx_k], dtype=np.float64)
            values_human = np.array(recall_human_top_all[idx_k], dtype=np.float64)
            metrics[f'recall@{kk}'] = float(np.nanmean(values_frame)) if values_frame.size > 0 else float('nan')
            metrics[f'recall_h@{kk}'] = float(np.nanmean(values_human)) if values_human.size > 0 else float('nan')
        return metrics

    def build_gt_pair_interactions(self, gt_hois, num_verbs):
        pair_to_idx = {}
        pair_idxes_gt = []
        interaction_distribution_gt = []
        for gt_hoi in gt_hois:
            pair = (gt_hoi['subject_id'], gt_hoi['object_id'])
            if pair not in pair_to_idx:
                pair_to_idx[pair] = len(pair_idxes_gt)
                pair_idxes_gt.append([gt_hoi['subject_id'], gt_hoi['object_id']])
                interaction_distribution_gt.append([0.0] * num_verbs)
            interaction_distribution_gt[pair_to_idx[pair]][gt_hoi['category_id']] = 1.0
        return pair_idxes_gt, interaction_distribution_gt

    def multi_label_recall(self, gts, preds, gt_counts=None):
        if gt_counts is None:
            gt_counts = []
        recalls = []
        for idx, (gt, pred) in enumerate(zip(gts, preds)):
            if len(gt_counts) > 0:
                gt_count = gt_counts[idx]
            else:
                gt_count = np.sum(gt)
            gt = np.asarray(gt).astype(bool)
            pred = np.asarray(pred).astype(bool)
            intersection = gt * pred
            if gt_count > 0:
                recalls.append(np.sum(intersection) / gt_count)
        if len(recalls) == 0:
            return np.nan
        return np.mean(recalls)

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                if self.triplet_eval:
                    triplet = pred_hoi['category_id']
                else:
                    triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                               pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        ''' Compute the IoU matrix between two list of bboxes
        Args:
            bbox_list1, bbox_list2: list of bboxes, each bbox is a dict with keys: 'bbox' and 'category_id'
        Returns:
            match_pairs_dict: key is the pred index , value is gt index
            match_pair_overlaps: key is the pred index , value is IoU with matched gt boxes
        '''
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}, {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                if self.triplet_eval or bbox1['category_id'] == bbox2['category_id']:
                    iou_i = self.compute_IOU(bbox1['bbox'], bbox2['bbox'])
                else:
                    iou_i = 0
                iou_mat[i, j] = iou_i
        
        # create a binary mask for the IoU matrix
        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)  #np.nonzero returns the indices of on-zero elements 
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]): #match_pairs[0]:gt indices, match_pairs[1]: pred indices
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        '''
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
        '''
        rec1 = bbox1
        rec2 = bbox2
        # computing area of each rectangles
        S_rec1 = (rec1[...,2] - rec1[...,0]+1) * (rec1[...,3] - rec1[...,1]+1)
        S_rec2 = (rec2[...,2] - rec2[...,0]+1) * (rec2[...,3] - rec2[...,1]+1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = np.maximum(rec1[...,1], rec2[...,1])
        right_line = np.minimum(rec1[...,3], rec2[...,3])
        top_line = np.maximum(rec1[...,0], rec2[...,0])
        bottom_line = np.minimum(rec1[...,2], rec2[...,2])
        
        # judge if there is an intersect
        #if left_line >= right_line or top_line >= bottom_line:
        #    return 0
        #else:
        #    intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
        width = np.maximum(0.0, right_line - left_line + 1)
        height = np.maximum(0.0, bottom_line - top_line + 1)
        intersect = width * height

        return intersect / (sum_area - intersect)
        '''
        else:
            return 0
        '''


def triplet_nms_filter(preds, num_alpha, num_beta, thres_nms):
    preds_filtered = []
    for img_preds in preds:
        pred_bboxes = img_preds['predictions']
        pred_hois = img_preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                        str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = pairwise_nms(np.array(subs), np.array(objs), np.array(scores), num_alpha, num_beta, thres_nms)

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered.append({
            #'filename': img_preds['file_names'],
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

    return preds_filtered
