"""
HOICLIP 主模型与训练/推理相关组件。

该文件主要包含：
- `HOICLIP`：DETR 风格的 HOI 检测模型。backbone 提供空间特征，`GEN`(见 `gen.py`) 生成
  human/object/interaction 的 query 表征，并与 CLIP 特征融合得到 HOI 分类 logits。
- `SetCriterionHOI`：训练损失（匹配、分类、框回归、可选 mimic/重建损失等）。
- `PostProcessHOITriplet`：推理后处理，把归一化框坐标映射回原图尺寸，并整理输出字段。
- `build`：按 args 组装 model / criterion / postprocessors。

注：本实现大量依赖配置项 `args`（数据集类型、是否使用 CLIP 初始化、zero-shot 设置等）。
"""

import os
import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX
from datasets.vidhoi_text_label import vidhoi_text_label, vidhoi_obj_text_label
from datasets.ag_text_label import ag_text_label, ag_obj_text_label

from ..backbone import build_backbone
from ..matcher import build_matcher, build_temporal_frame_weights
from .gen import build_gen


def _sigmoid(x):
    # 带截断的 sigmoid，避免极端值导致 log/梯度数值不稳定（常用于 focal loss 变体）。
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _build_future_hoi_weight_dict(args):
    total = float(args.anticipation_hoi_coef)
    gamma = float(args.anticipation_horizon_gamma)
    if total < 0 or gamma < 0:
        raise ValueError('anticipation loss coefficients must be non-negative')
    raw = [gamma ** index for index in range(4)]
    normalizer = sum(raw) or 1.0
    return {
        f'loss_hoi_labels_f{horizon}': total * weight / normalizer
        for horizon, weight in zip([1, 3, 5, 7], raw)
    }


class HOICLIP(nn.Module):
    """
    HOICLIP 模型。

    核心思路：
    - backbone 提取图像特征；1x1 conv 投影到 transformer hidden_dim。
    - 使用两组 query（human/object）解码实例，再构造交互表征（interaction）并与 CLIP 视觉 token 融合。
    - HOI/Obj 分类可选择使用 CLIP 文本原型初始化的线性层（zero-shot/可迁移），并配合可学习的 logit_scale。

    forward 输出（主要字段）：
    - `pred_hoi_logits` / `pred_obj_logits`：HOI 与对象分类 logits
    - `pred_sub_boxes` / `pred_obj_boxes`：subject/object 的 box（cx,cy,w,h，归一化到 [0,1]）
    - `clip_visual` / `clip_logits`：来自 CLIP 的视觉特征与 HOI 相似度得分（用于辅助/分析）
    - `aux_outputs`：若 `aux_loss=True`，返回中间层的辅助预测用于深监督
    """

    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # DETR 风格的 learnable queries：分别用于 human 与 object。
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        # 用于引导 query 的位置 embedding（会加到 human/object queries 上）。
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        # box 回归头（MLP 输出 4 维，后续 sigmoid 到 [0,1]）。
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 将交互特征映射到“verb”空间（后续通过 verb2hoi 投影辅助 HOI 分类）。
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        # backbone 通道 -> transformer hidden_dim 的线性投影。
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        # CLIP 类似的可学习温度参数（指数后作为 logit_scale）。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 加载 CLIP（ModifiedCLIP）模型，用于文本原型初始化与图像编码。
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        # 根据数据集选择 HOI/OBJ 的文本标签表（用于 CLIP 文本编码）。
        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None
        elif self.args.dataset_file == 'vidhoi':
            hoi_text_label = vidhoi_text_label
            # append a "nothing" class so num_obj_classes = len(vidhoi_obj_text_label)
            obj_text_label = vidhoi_obj_text_label + [(len(vidhoi_obj_text_label), 'a photo of nothing')]
            unseen_index = None
        elif self.args.dataset_file == 'ag':
            hoi_text_label = ag_text_label
            obj_text_label = ag_obj_text_label + [(len(ag_obj_text_label), 'a photo of nothing')]
            unseen_index = None

        # 使用 CLIP 文本编码器初始化分类器权重（可选对 unseen 类做删除/分支初始化）。
        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)
            # clip_label: hoi_text_feature, obj_clip_label: obj_text_feature, v_linear_proj_weight: clip_model.visual.proj(冻结)，hoi_text: 按 zero-shot 配置删除 unseen 后标签，obj_text: tokenized 的 object 文本，train_clip_label: 按 zero-shot 配置删除 unseen 后的 hoi_text_feature（用于训练分支初始化）
        num_obj_classes = len(obj_text) - 1  # del nothing
        # CLIP 视觉投影矩阵（用于将 CLIP visual features 对齐到 embedding 空间）。
        self.clip_visual_proj = v_linear_proj_weight  # clip_model.visual.proj(冻结)

        # 将 transformer hidden_dim 的特征映射到 CLIP embedding 维度（用于后续融合/损失）。
        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            self.verb_projection = nn.Linear(args.clip_embed_dim, 117, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight
        elif self.args.dataset_file in ('vidhoi', 'ag'):
            # verb2hoi_proj[v][hoi_idx] = 1 when HOI class hoi_idx has verb v.
            # Shape: (num_verb_classes, num_hoi_classes)
            text_label_ids = list((vidhoi_text_label if self.args.dataset_file == 'vidhoi'
                                   else ag_text_label).keys())
            n_verb = args.num_verb_classes
            n_hoi = len(text_label_ids)
            verb2hoi_proj = torch.zeros(n_verb, n_hoi)
            for hoi_idx, (v, _) in enumerate(text_label_ids):
                verb2hoi_proj[v][hoi_idx] = 1.0
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection = nn.Linear(args.clip_embed_dim, n_verb, bias=False)
            verb_pth = getattr(args, 'verb_pth', '')
            if verb_pth and os.path.exists(verb_pth):
                loaded = torch.load(verb_pth, map_location='cpu')
                if loaded.shape == self.verb_projection.weight.shape:
                    self.verb_projection.weight.data = loaded
            self.verb_weight = args.verb_weight
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection = nn.Linear(args.clip_embed_dim, 29, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight

        if args.with_clip_label:
            # HOI 分类：使用 CLIP 文本原型初始化的线性层（可选冻结）。
            if args.fix_clip_label:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text), bias=False)
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
                for i in self.visual_projection.parameters():
                    i.require_grad = False
            else:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                # zero-shot 评估时使用完整 600 类的投影（不删除 unseen）。
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600, bias=False)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            # 不使用 CLIP label 初始化时，退化为普通的可学习分类头。
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            # 对象分类同样可选择使用 CLIP 文本原型初始化。
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            if args.fix_clip_label:    # use fixed clip label for object classification
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
                for i in self.obj_visual_projection.parameters():
                    i.require_grads = False
            else:                     # use learnable projection initialized by clip label
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        # 将 HOI 文本原型传给 GEN（`gen.py` 中会用它计算 clip_hoi_score）。
        self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        """
        用 CLIP 文本编码器把 HOI/OBJ 的文本 prompt 编码成向量，用于初始化分类器权重。

        返回：
        - `text_embedding`：HOI 文本向量（可能包含全部类）
        - `obj_text_embedding`：OBJ 文本向量
        - `v_linear_proj_weight`：CLIP 视觉投影矩阵 `visual.proj`
        - `hoi_text_label_del`：按 zero-shot 配置删除 unseen 后的 label dict（用于训练分支）
        - `obj_text_inputs`：tokenized 的 object 文本（此处保留张量，主要用于统计/长度）
        - `text_embedding_del`：删除 unseen 后的 HOI 文本向量（用于训练初始化）
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:  #随机初始化
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())

    def forward(self, samples: NestedTensor, is_training=True, clip_input=None, targets=None):
        """
        前向推理/训练。

        - `samples`：可以是 `NestedTensor`，也可以是图片 tensor list（会被打包成 NestedTensor）。
        - `clip_input`：送入 CLIP 的图像输入（一般是预处理后的图像 batch）。
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # transformer(=GEN) 输出：human/object/interaction hidden states + CLIP 相关特征/打分。
        h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual = self.transformer(self.input_proj(src), mask,
                                                 self.query_embed_h.weight,
                                                 self.query_embed_o.weight,
                                                 self.pos_guided_embedd.weight,
                                                 pos[-1], self.clip_model, self.clip_visual_proj, clip_input)

        # box 回归：对每一层 decoder 输出都预测 box，最后一层用于主输出。
        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            # inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            # verb 分支：将交互特征映射到 verb，再通过 verb2hoi 投影辅助 HOI 分类。
            verb_hs = self.inter2verb(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            verb_hs = verb_hs / verb_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj_eval
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        else:
            # 不用 CLIP label 初始化：先投影到 clip_embed_dim，再用线性层输出类别。
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)

        # 统一整理输出字典：训练与推理都使用同一套 key。
        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'clip_visual': clip_visual,
               'clip_cls_feature': clip_cls_feature, 'hoi_feature': inter_hs[-1], 'clip_logits': clip_hoi_score}

        if self.args.with_mimic:
            out['inter_memory'] = outputs_inter_hs[-1]
        if self.aux_loss:
            if self.args.with_mimic:
                aux_mimic = outputs_inter_hs
            else:
                aux_mimic = None

            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            aux_mimic)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):
        # 将中间层预测整理成 list[dict]，用于 SetCriterion 中按层计算辅助损失。
        if outputs_hoi_class.shape[0] == 1:
            outputs_hoi_class = outputs_hoi_class.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):
    """
    训练损失计算。

    典型流程：
    - matcher 先做匈牙利匹配（prediction queries ↔ ground truth interactions）。
    - 基于匹配结果计算分类损失（HOI/OBJ）、框回归（L1 + GIoU）、以及可选的特征 mimic / 重建损失。
    """

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.with_mimic:
            self.clip_model, _ = clip.load(args.clip_model, device=device)
        else:
            self.clip_model = None
        self.alpha = args.alpha
        self.is_lstm = getattr(args, 'model_name', None) == 'HOICLIP_LSTM'
        self.clip_num_frames = getattr(args, 'max_length', 1) if self.is_lstm else 1
        self.future_horizons = [1, 3, 5, 7]
        self.register_buffer(
            'detection_frame_weights',
            build_temporal_frame_weights(
                self.clip_num_frames,
                getattr(args, 'detection_frame_weighting', 'uniform'),
                getattr(args, 'detection_frame_weight_power', 1.0),
            ),
        )

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions,
                        topk=5, indices_future=None):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']
        dtype = src_logits.dtype

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        # HOI 分类使用 focal-like 的负样本挖掘损失（`_neg_loss`）。
        if self.is_lstm:
            batch_size, total_queries, num_classes = src_logits.shape
            queries_per_frame = total_queries // self.clip_num_frames
            src_by_frame = src_logits.view(
                batch_size, self.clip_num_frames, queries_per_frame, num_classes
            )
            target_by_frame = target_classes.view_as(src_by_frame)
            frame_weights = self.detection_frame_weights.to(src_logits.device, src_logits.dtype)
            loss_hoi_ce = sum(
                frame_weights[frame_index]
                * self._neg_loss(
                    src_by_frame[:, frame_index],
                    target_by_frame[:, frame_index],
                    weights=None,
                    alpha=self.alpha,
                )
                for frame_index in range(self.clip_num_frames)
            )
        else:
            loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        future_logits = outputs.get('future_hoi_logits')
        if future_logits is not None:
            if indices_future is None:
                raise ValueError('future_hoi_logits requires future matching indices')
            future_flat = torch.cat(future_logits, dim=1)
            future_targets = torch.zeros_like(future_flat)
            future_idx = self._get_src_permutation_idx(indices_future)
            matched_future = [
                target['hoi_labels_future'][target_indices]
                for target, (_, target_indices) in zip(targets, indices_future)
                if 'hoi_labels_future' in target and target_indices.numel() > 0
            ]
            if matched_future:
                future_targets[future_idx] = torch.cat(matched_future).to(future_flat.dtype)
            future_targets = future_targets.chunk(len(self.future_horizons), dim=1)
            for slot, (horizon, logits_h, targets_h) in enumerate(zip(
                self.future_horizons, future_logits, future_targets
            )):
                keep = torch.tensor(
                    [horizon in target.get('future', []) for target in targets],
                    device=logits_h.device,
                    dtype=torch.bool,
                )
                if keep.any():
                    losses[f'loss_hoi_labels_f{horizon}'] = self._neg_loss(
                        _sigmoid(logits_h[keep]), targets_h[keep],
                        weights=None, alpha=self.alpha,
                    )
                else:
                    # Keep every future head in the graph for short clips and DDP.
                    losses[f'loss_hoi_labels_f{horizon}'] = logits_h.sum() * 0.0

        _, pred = src_logits[idx].topk(min(topk, src_logits.shape[-1]), 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        # 可选：用 L1 让交互特征逼近 CLIP 图像特征（feature mimic）。
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses
    def reconstruction_loss(self, outputs, targets, indices, num_interactions):
        # 可选：用 L1 让 HOI 特征重建/接近 CLIP 全局特征。
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss,
                'rec_loss': self.reconstruction_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if 'future_hoi_logits' in outputs_without_aux:
            indices, indices_future = self.matcher(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)
            indices_future = None

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {'indices_future': indices_future} if loss == 'hoi_labels' else {}
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_interactions,
                **kwargs,
            ))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss =='rec_loss':
                        continue
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):
    """
    推理后处理：
    - sigmoid 得到 HOI/OBJ 分数
    - 将预测的归一化 box 映射到原图尺寸（absolute xyxy）
    - 拼接 subject/object 的 labels 与 boxes，并返回 query 对应的 sub_id/obj_id
    """

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.num_frames = getattr(args, 'max_length', 1)
        self.is_lstm = getattr(args, 'model_name', None) == 'HOICLIP_LSTM'

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        clip_visual = outputs['clip_visual']
        clip_logits = outputs['clip_logits']
        future_hoi_logits = outputs.get('future_hoi_logits')

        if self.is_lstm:
            if out_hoi_logits.shape[1] % self.num_frames != 0:
                raise ValueError(
                    f"LSTM query count {out_hoi_logits.shape[1]} is not divisible by "
                    f"T={self.num_frames}"
                )
            queries_per_frame = out_hoi_logits.shape[1] // self.num_frames
            current = slice((self.num_frames - 1) * queries_per_frame, None)
            out_hoi_logits = out_hoi_logits[:, current]
            out_obj_logits = out_obj_logits[:, current]
            out_sub_boxes = out_sub_boxes[:, current]
            out_obj_boxes = out_obj_boxes[:, current]

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'clip_logits': clip_logits[index].to('cpu')})
            if future_hoi_logits is not None:
                results[-1]['hoi_scores_future'] = torch.cat([
                    _sigmoid(logits[index]).to('cpu') for logits in future_hoi_logits
                ], dim=0)

        return results


class HOICLIP_LSTM(nn.Module):
    """
    在冻结 HOICLIP 基础上附加 LSTM 时序建模。

    架构：
      1. 冻结 HOICLIP：逐帧提取交互特征 hoi_feature (B, Q, clip_embed_dim)。
      2. LSTM：对 T 帧的 hoi_feature 序列（按 query 独立建模）做时序聚合。
      3. 当前帧预测头：LSTM 最后输出 → HOI logits。
      4. 未来预测头（×4）：分别预测未来 h=1,3,5,7 帧的 HOI logits。

    forward 输入：
      samples_list   : list[NestedTensor], 长度 T，每个 (B, C, H, W)
      clip_inputs_list: list[Tensor], 长度 T，CLIP 预处理图像 (B, 3, 224, 224)
      targets_list   : list[list[dict]], 长度 T，每帧的 target 列表（训练时使用）
      target_sizes   : Tensor (B, 2)，最后一帧的原图尺寸

    forward 输出（兼容 PostProcessHOITriplet 与原有 criterion）：
      pred_hoi_logits      : (B, Q, N_hoi)  — 当前帧 LSTM 增强后的预测
      pred_obj_logits      : 来自最后帧 HOICLIP 输出
      pred_sub_boxes       : 来自最后帧 HOICLIP 输出
      pred_obj_boxes       : 来自最后帧 HOICLIP 输出
      future_hoi_logits    : list[Tensor(B, Q, N_hoi)], 长度 4，对应 h=1,3,5,7
      clip_visual / clip_logits: 来自最后帧 HOICLIP 输出
    """

    FUTURES = [1, 3, 5, 7]

    def __init__(self, hoiclip: HOICLIP, args):
        super().__init__()
        # Version 2 is the first layout with sample-major BxT restoration and future losses.
        self.register_buffer('_temporal_layout_version', torch.tensor(2), persistent=True)
        # 冻结 HOICLIP 全部参数
        self.hoiclip = hoiclip
        for p in self.hoiclip.parameters():
            p.requires_grad_(False)

        clip_embed_dim = args.clip_embed_dim
        lstm_hidden    = getattr(args, 'lstm_hidden_dim', clip_embed_dim)
        lstm_layers    = getattr(args, 'lstm_num_layers', 1)
        # with_clip_label=False 时退化为 hoi_class_embedding
        if hasattr(hoiclip, 'visual_projection'):
            num_hoi = hoiclip.visual_projection.out_features
        else:
            num_hoi = hoiclip.hoi_class_embedding.out_features

        # T：clip 窗口长度，由 args.max_length 提供
        self.T = args.max_length

        self.lstm = nn.LSTM(clip_embed_dim, lstm_hidden, lstm_layers, batch_first=False)
        self.curr_hoi_head = nn.Linear(lstm_hidden, num_hoi)
        self.future_hoi_heads = nn.ModuleList([
            nn.Linear(lstm_hidden, num_hoi) for _ in self.FUTURES
        ])

        # Cross-dataset linear probe: keep the temporal dynamics learnt on the source
        # dataset and re-learn only the five classifier heads on the new label space.
        self.freeze_temporal = getattr(args, 'freeze_temporal', False)
        if self.freeze_temporal:
            for p in self.lstm.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.hoiclip.eval()  # frozen backbone always stays in eval
        if self.freeze_temporal:
            # Keep the frozen LSTM in eval so its features stay deterministic
            # for the linear probe fitted on top of them.
            self.lstm.eval()
        return self

    def forward(self, samples, is_training=True, clip_input=None, targets=None):
        """
        与 HOICLIP.forward 签名一致，engine.py 无需改动。

        samples   : NestedTensor  (B*T, 3, H, W)
        clip_input: Tensor (B, T, 3, 224, 224) 或 (B*T, 3, 224, 224) 或 None
        targets   : list[B dicts]，仅 criterion 在 forward 外使用，此处忽略
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        T    = self.T
        B_T  = samples.tensors.shape[0]
        if B_T % T != 0:
            raise ValueError(f"Flattened clip batch {B_T} is not divisible by T={T}")
        B    = B_T // T

        # collate 按样本优先展平；先恢复 (B,T,...) 才能按时间取帧，避免 B>1 时混样本。
        tensors_bt = samples.tensors.reshape(B, T, *samples.tensors.shape[1:])
        masks_bt = samples.mask.reshape(B, T, *samples.mask.shape[1:])

        # 拆分 clip_input
        if clip_input is not None:
            if clip_input.dim() == 5:                  # (B, T, 3, 224, 224)
                if clip_input.shape[:2] != (B, T):
                    raise ValueError(
                        f"clip_input batch/time {tuple(clip_input.shape[:2])} != {(B, T)}"
                    )
                clip_bt = clip_input
            else:                                      # (B*T, 3, 224, 224)
                if clip_input.shape[0] != B_T:
                    raise ValueError(
                        f"Flattened clip_input batch {clip_input.shape[0]} != image batch {B_T}"
                    )
                clip_bt = clip_input.reshape(B, T, *clip_input.shape[1:])
        else:
            clip_bt = None

        # 逐帧过冻结 HOICLIP，提取 hoi_feature
        hoi_features = []
        last_out = None
        for t in range(T):
            nt = NestedTensor(tensors_bt[:, t], masks_bt[:, t])
            clip_frame = clip_bt[:, t] if clip_bt is not None else None
            with torch.no_grad():
                out = self.hoiclip(nt, is_training=is_training, clip_input=clip_frame)
            hoi_features.append(out['hoi_feature'])   # (B, Q, D)
            last_out = out

        # LSTM 时序建模：(T, B*Q, D) → (T, B*Q, hidden)
        feat_seq  = torch.stack(hoi_features, dim=0)  # (T, B, Q, D)
        T_, B_, Q, D = feat_seq.shape
        feat_flat = feat_seq.reshape(T_, B_ * Q, D)
        lstm_out, _ = self.lstm(feat_flat)

        curr_feat       = lstm_out[-1].reshape(B_, Q, -1)   # (B, Q, hidden)
        curr_hoi_logits = self.curr_hoi_head(curr_feat)      # (B, Q, N_hoi)
        future_hoi_logits = [head(curr_feat) for head in self.future_hoi_heads]

        # Clip matcher supervises one stable query/pair assignment on every observed frame.
        # Spatial predictions stay frozen, so the last-frame tensors are repeated only to
        # align their query layout with the flattened (T*num_pairs) training targets.
        def _repeat_observed(tensor):
            return tensor.unsqueeze(1).expand(-1, T, -1, -1).flatten(1, 2)

        return {
            'pred_hoi_logits':   _repeat_observed(curr_hoi_logits),
            'pred_obj_logits':   _repeat_observed(last_out['pred_obj_logits']),
            'pred_sub_boxes':    _repeat_observed(last_out['pred_sub_boxes']),
            'pred_obj_boxes':    _repeat_observed(last_out['pred_obj_boxes']),
            'future_hoi_logits': future_hoi_logits,
            'clip_visual':       last_out.get('clip_visual'),
            'clip_logits':       last_out.get('clip_logits'),
        }


def build(args):
    # 构建：backbone + GEN + matcher + criterion + postprocessor
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = HOICLIP(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.model_name == 'HOICLIP_LSTM':
        # Frozen HOICLIP spatial outputs are used for matching, not optimization.
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict.update(_build_future_hoi_weight_dict(args))
        losses = ['hoi_labels']
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
        weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_giou'] = args.giou_loss_coef
        if args.with_mimic:
            weight_dict['loss_feat_mimic'] = args.mimic_loss_coef
        if args.with_rec_loss:
            weight_dict['loss_rec'] = args.rec_loss_coef
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
        if args.with_mimic:
            losses.append('feats_mimic')
        if args.with_rec_loss:
            losses.append('rec_loss')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
