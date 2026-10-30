import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def build_temporal_frame_weights(num_frames, weighting='uniform', power=1.0):
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if weighting == 'uniform':
        weights = torch.ones(num_frames, dtype=torch.float32)
    elif weighting == 'linear':
        weights = torch.arange(1, num_frames + 1, dtype=torch.float32)
        weights = weights.pow(float(power))
    else:
        raise ValueError(f"Unsupported detection_frame_weighting: {weighting}")
    return weights / weights.sum()

class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_hoi_class: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_hoi_class = cost_hoi_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_sub_boxes'].shape[:2]
        if 'pred_hoi_logits' in outputs.keys():
            out_hoi_prob = outputs['pred_hoi_logits'].flatten(0, 1).sigmoid()
            tgt_hoi_labels = torch.cat([v['hoi_labels'] for v in targets])
            tgt_hoi_labels_permute = tgt_hoi_labels.permute(1, 0)
            cost_hoi_class = -(out_hoi_prob.matmul(tgt_hoi_labels_permute) / \
                               (tgt_hoi_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                               (1 - out_hoi_prob).matmul(1 - tgt_hoi_labels_permute) / \
                               ((1 - tgt_hoi_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            cost_hoi_class = self.cost_hoi_class * cost_hoi_class
        elif 'pred_verb_logits' in outputs.keys():
            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                                (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                                ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            cost_hoi_class = self.cost_verb_class * cost_verb_class
        else:
            cost_hoi_class = 0


        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        if out_sub_bbox.dtype == torch.float16:
            out_sub_bbox = out_sub_bbox.type(torch.float32)
            out_obj_bbox = out_obj_bbox.type(torch.float32)

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_hoi_class * cost_hoi_class + self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou + self.cost_obj_class * cost_obj_class

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['sub_boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherHOIClip(nn.Module):
    """Match one temporal query slot to one tracked pair across a clip."""

    def __init__(self, cost_obj_class=1, cost_hoi_class=1, cost_bbox=1, cost_giou=1,
                 num_frames=6, detection_frame_weighting='linear',
                 detection_frame_weight_power=1.0):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_hoi_class = cost_hoi_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.horizons = [1, 3, 5, 7]
        self.register_buffer(
            'frame_weights',
            build_temporal_frame_weights(
                num_frames, detection_frame_weighting, detection_frame_weight_power
            ),
        )

    def _reduce_valid(self, cost, valid):
        # cost: (T,Q,N), valid: (T,N). Invalid tracked pairs must not affect matching.
        weights = self.frame_weights.to(cost.device, cost.dtype)[:, None, None]
        mask = valid.to(cost.dtype)[:, None, :]
        denominator = (weights * mask).sum(dim=0).clamp_min(1e-6)
        return (cost * weights * mask).sum(dim=0) / denominator

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, total_queries = outputs['pred_sub_boxes'].shape[:2]
        if total_queries % self.num_frames != 0:
            raise ValueError(
                f"Clip predictions {total_queries} are not divisible by T={self.num_frames}"
            )
        queries_per_frame = total_queries // self.num_frames
        pred_obj = outputs['pred_obj_logits'].view(
            batch_size, self.num_frames, queries_per_frame, -1
        ).softmax(-1)
        pred_hoi = outputs['pred_hoi_logits'].view(
            batch_size, self.num_frames, queries_per_frame, -1
        ).sigmoid()
        pred_sub_boxes = outputs['pred_sub_boxes'].view(
            batch_size, self.num_frames, queries_per_frame, 4
        )
        pred_obj_boxes = outputs['pred_obj_boxes'].view(
            batch_size, self.num_frames, queries_per_frame, 4
        )
        future_logits = outputs.get('future_hoi_logits', None)

        indices = []
        indices_future = []
        for batch_index, target in enumerate(targets):
            valid = target['valid'].bool()
            num_pairs = valid.shape[1]
            if num_pairs == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                indices_future.append((empty, empty))
                continue

            target_obj = target['obj_labels'].view(self.num_frames, num_pairs)
            target_hoi = target['hoi_labels'].view(
                self.num_frames, num_pairs, -1
            ).transpose(1, 2)
            target_sub_boxes = target['sub_boxes'].view(self.num_frames, num_pairs, 4)
            target_obj_boxes = target['obj_boxes'].view(self.num_frames, num_pairs, 4)

            cost_obj = -pred_obj[batch_index][:, :, target_obj[0]]
            positive = target_hoi.sum(dim=1, keepdim=True).clamp_min(1e-4)
            negative = (1 - target_hoi).sum(dim=1, keepdim=True).clamp_min(1e-4)
            cost_hoi = -(
                pred_hoi[batch_index].matmul(target_hoi) / positive
                + (1 - pred_hoi[batch_index]).matmul(1 - target_hoi) / negative
            ) / 2

            cost_sub_bbox = (
                pred_sub_boxes[batch_index].unsqueeze(2)
                - target_sub_boxes.unsqueeze(1)
            ).abs().sum(-1)
            cost_obj_bbox = (
                pred_obj_boxes[batch_index].unsqueeze(2)
                - target_obj_boxes.unsqueeze(1)
            ).abs().sum(-1)
            object_exists = (target_obj_boxes != 0).any(dim=-1)[:, None, :]
            cost_bbox = torch.maximum(cost_sub_bbox, cost_obj_bbox * object_exists)

            giou_frames = []
            for frame_index in range(self.num_frames):
                cost_sub_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_sub_boxes[batch_index, frame_index]),
                    box_cxcywh_to_xyxy(target_sub_boxes[frame_index]),
                )
                cost_obj_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_obj_boxes[batch_index, frame_index]),
                    box_cxcywh_to_xyxy(target_obj_boxes[frame_index]),
                )
                missing_object = (target_obj_boxes[frame_index] == 0).all(dim=1)
                cost_obj_giou = cost_obj_giou + cost_sub_giou * missing_object.unsqueeze(0)
                giou_frames.append(torch.maximum(cost_sub_giou, cost_obj_giou))
            cost_giou = torch.stack(giou_frames)

            reduced_hoi = self._reduce_valid(cost_hoi, valid)
            if future_logits is not None and 'future' in target:
                future_valid = target['valid_future'].bool()
                num_future = len(target['future'])
                target_future = target['hoi_labels_future'].view(
                    num_future, num_pairs, -1
                )
                future_cost_sum = torch.zeros_like(reduced_hoi)
                future_count = torch.zeros(num_pairs, device=reduced_hoi.device, dtype=reduced_hoi.dtype)
                for local_index, horizon in enumerate(target['future']):
                    slot = self.horizons.index(int(horizon))
                    target_h = target_future[local_index].transpose(0, 1)
                    pred_h = future_logits[slot][batch_index].sigmoid()
                    positive_h = target_h.sum(dim=0, keepdim=True).clamp_min(1e-4)
                    negative_h = (1 - target_h).sum(dim=0, keepdim=True).clamp_min(1e-4)
                    cost_h = -(
                        pred_h.matmul(target_h) / positive_h
                        + (1 - pred_h).matmul(1 - target_h) / negative_h
                    ) / 2
                    valid_h = future_valid[local_index].to(cost_h.dtype)
                    future_cost_sum += cost_h * valid_h.unsqueeze(0)
                    future_count += valid_h
                has_future = future_count > 0
                future_cost = future_cost_sum / future_count.clamp_min(1).unsqueeze(0)
                reduced_hoi = torch.where(
                    has_future.unsqueeze(0), (reduced_hoi + future_cost) / 2, reduced_hoi
                )

            total_cost = (
                self.cost_hoi_class * reduced_hoi
                + self.cost_obj_class * self._reduce_valid(cost_obj, valid)
                + self.cost_bbox * self._reduce_valid(cost_bbox, valid)
                + self.cost_giou * self._reduce_valid(cost_giou, valid)
            )
            query_ids, pair_ids = linear_sum_assignment(total_cost.cpu())

            current_src, current_tgt = [], []
            future_src, future_tgt = [], []
            for query_id, pair_id in zip(query_ids, pair_ids):
                observed_frames = valid[:, pair_id].nonzero().flatten()
                current_src.append(observed_frames * queries_per_frame + int(query_id))
                current_tgt.append(observed_frames * num_pairs + int(pair_id))

                if 'future' in target:
                    for local_index, horizon in enumerate(target['future']):
                        if target['valid_future'][local_index, pair_id]:
                            slot = self.horizons.index(int(horizon))
                            future_src.append(torch.tensor(
                                [slot * queries_per_frame + int(query_id)],
                                device=valid.device,
                            ))
                            future_tgt.append(torch.tensor(
                                [local_index * num_pairs + int(pair_id)],
                                device=valid.device,
                            ))

            empty = torch.empty(0, dtype=torch.int64, device=valid.device)
            indices.append((
                torch.cat(current_src).long() if current_src else empty,
                torch.cat(current_tgt).long() if current_tgt else empty,
            ))
            indices_future.append((
                torch.cat(future_src).long() if future_src else empty,
                torch.cat(future_tgt).long() if future_tgt else empty,
            ))

        return indices, indices_future


def build_matcher(args):
    if getattr(args, 'model_name', None) == 'HOICLIP_LSTM':
        return HungarianMatcherHOIClip(
            cost_obj_class=args.set_cost_obj_class,
            cost_hoi_class=args.set_cost_hoi,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            num_frames=args.max_length,
            detection_frame_weighting=args.detection_frame_weighting,
            detection_frame_weight_power=args.detection_frame_weight_power,
        )
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                               cost_hoi_class=args.set_cost_hoi)
