import numpy as np
from collections import defaultdict

from datasets.vidhoi_text_label import vidhoi_text_label


def _build_verb2hoi(text_label_ids, num_verb_classes):
    """
    Returns verb2hoi: list of length num_verb_classes, each element is a list of
    HOI class indices that share that verb.  Used to aggregate N_hoi scores → 50 verb scores.
    """
    verb2hoi = [[] for _ in range(num_verb_classes)]
    for hoi_idx, (v, _) in enumerate(text_label_ids):
        verb2hoi[v].append(hoi_idx)
    return verb2hoi


class VidHOIEvaluator:
    """
    mAP evaluator for VidHOI.

    pred hoi_scores shape: (Q, N_hoi) where N_hoi = len(vidhoi_text_label).
    Evaluation is at verb category level (50 classes): for each query,
    the verb score is the max over all HOI classes sharing that verb.

    GT category_id = verb_id (0-49).
    Boxes must be in absolute xyxy format.
    Matching: IoU(sub) >= 0.5 AND IoU(obj) >= 0.5 AND verb matches.
    """

    def __init__(self, preds, gts, num_verb_classes=50, use_nms_filter=False,
                 thres_nms=0.5, nms_alpha=1.0, nms_beta=0.0):
        self.overlap_iou = 0.5
        self.max_hois = 100
        self.use_nms_filter = use_nms_filter
        self.thres_nms = thres_nms
        self.nms_alpha = nms_alpha
        self.nms_beta = nms_beta
        self.num_verb_classes = num_verb_classes

        self.text_label_ids = list(vidhoi_text_label.keys())
        # verb2hoi[v] = list of HOI class indices with verb == v
        self.verb2hoi = _build_verb2hoi(self.text_label_ids, num_verb_classes)

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_categories = []

        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()
                         if hasattr(v, 'to')}
            bboxes = [{'bbox': list(bbox), 'category_id': int(label)}
                      for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['hoi_scores']  # (Q, N_hoi)

            if len(bboxes) > 0 and hoi_scores.shape[0] > 0:
                Q = hoi_scores.shape[0]
                # Project N_hoi scores → num_verb_classes scores (max over HOI classes per verb)
                verb_scores = np.zeros((Q, num_verb_classes), dtype=np.float32)
                for v, hoi_idxs in enumerate(self.verb2hoi):
                    if hoi_idxs:
                        verb_scores[:, v] = hoi_scores[:, hoi_idxs].max(axis=1)

                verb_labels = np.tile(np.arange(num_verb_classes), (Q, 1))
                subject_ids = np.tile(img_preds['sub_ids'], (num_verb_classes, 1)).T
                object_ids = np.tile(img_preds['obj_ids'], (num_verb_classes, 1)).T

                hois = [{'subject_id': int(si), 'object_id': int(oi),
                         'category_id': int(v), 'score': float(s)}
                        for si, oi, v, s in zip(
                            subject_ids.ravel(), object_ids.ravel(),
                            verb_labels.ravel(), verb_scores.ravel())]
                hois.sort(key=lambda k: k['score'], reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index].get('filename', str(index))
            self.preds.append({'filename': filename, 'predictions': bboxes,
                               'hoi_prediction': hois})

        self.gts = []
        for img_gts in gts:
            filename = img_gts.get('filename', '')
            gt_tensors = {k: v.to('cpu').numpy() for k, v in img_gts.items()
                          if hasattr(v, 'to') and k not in ('id',)}
            annotations = [{'bbox': list(bbox), 'category_id': int(label)}
                           for bbox, label in zip(gt_tensors['boxes'], gt_tensors['labels'])]
            hoi_annotation = []
            if 'hois' in gt_tensors and len(gt_tensors['hois']) > 0:
                for hoi in gt_tensors['hois']:
                    hoi_annotation.append({'subject_id': int(hoi[0]),
                                           'object_id': int(hoi[1]),
                                           'category_id': int(hoi[2])})
            self.gts.append({'filename': filename, 'annotations': annotations,
                             'hoi_annotation': hoi_annotation})
            for hoi in hoi_annotation:
                cat = hoi['category_id']
                if cat not in self.gt_categories:
                    self.gt_categories.append(cat)
                self.sum_gts[cat] += 1

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            if len(pred_bboxes) == 0:
                continue
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']

            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    cat = pred_hoi['category_id']
                    if cat not in self.gt_categories:
                        continue
                    self.tp[cat].append(0)
                    self.fp[cat].append(1)
                    self.score[cat].append(pred_hoi['score'])

        return self.compute_map()

    def compute_map(self):
        ap = {}
        for cat in self.gt_categories:
            if self.sum_gts[cat] == 0:
                continue
            tp = np.array(self.tp[cat])
            fp = np.array(self.fp[cat])
            if len(tp) == 0:
                ap[cat] = 0.0
                continue
            score = np.array(self.score[cat])
            sort_inds = np.argsort(-score)
            fp = np.cumsum(fp[sort_inds])
            tp = np.cumsum(tp[sort_inds])
            rec = tp / self.sum_gts[cat]
            prec = tp / (fp + tp)
            ap[cat] = self.voc_ap(rec, prec)

        m_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        print(f'mAP: {m_ap:.4f}  (over {len(ap)} verb categories)')
        return {'mAP': m_ap}

    def voc_ap(self, rec, prec):
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0
            ap += p / 11.0
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: k.get('score', 0), reverse=True)
        for pred_hoi in pred_hois:
            cat = pred_hoi['category_id']
            if cat not in self.gt_categories:
                continue
            is_match = 0
            max_gt_hoi_idx = -1
            if (len(match_pairs) != 0
                    and pred_hoi['subject_id'] in pos_pred_ids
                    and pred_hoi['object_id'] in pos_pred_ids):
                pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                pred_obj_ids = match_pairs[pred_hoi['object_id']]
                pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                max_overlap = 0
                for gt_idx, gt_hoi in enumerate(gt_hois):
                    if (gt_hoi['subject_id'] in pred_sub_ids
                            and gt_hoi['object_id'] in pred_obj_ids
                            and cat == gt_hoi['category_id']):
                        min_ov = min(
                            pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                            pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                        if min_ov > max_overlap:
                            max_overlap = min_ov
                            is_match = 1
                            max_gt_hoi_idx = gt_idx
            if is_match == 1 and vis_tag[max_gt_hoi_idx] == 0:
                self.fp[cat].append(0)
                self.tp[cat].append(1)
                vis_tag[max_gt_hoi_idx] = 1
            else:
                self.fp[cat].append(1)
                self.tp[cat].append(0)
            self.score[cat].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        """bbox in xyxy absolute format."""
        if not bbox_list1 or not bbox_list2:
            return {}, {}
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_mat[i, j] = self.compute_IOU(bbox1, bbox2)
        iou_mat_ov = iou_mat.copy()
        iou_mat = (iou_mat >= self.overlap_iou).astype(float)
        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict:
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        """xyxy format: area = (x2-x1+1)*(y2-y1+1)."""
        rec1, rec2 = bbox1['bbox'], bbox2['bbox']
        s1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
        s2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)
        xl = max(rec1[0], rec2[0])
        yt = max(rec1[1], rec2[1])
        xr = min(rec1[2], rec2[2])
        yb = min(rec1[3], rec2[3])
        if xr < xl or yb < yt:
            return 0.0
        inter = (xr - xl + 1) * (yb - yt + 1)
        return inter / (s1 + s2 - inter)
