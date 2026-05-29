"""
该文件实现 HOICLIP 中的 GEN 模块（基于 Transformer 的“实例/交互”特征生成与 CLIP 融合）。

整体结构：
- `TransformerEncoder`：对 backbone 输出的特征图序列编码得到 `memory`。
- `instance_decoder`：用 human/object 两组 queries 解码，得到人/物实例表征。
- `clip_interaction_decoder`：将人/物实例的平均作为交互 query，映射到 CLIP 维度后与 CLIP visual tokens
  进行融合解码，得到交互表征；同时计算 CLIP 全局特征与 HOI 文本原型的相似度得分。

实现细节：
- 该文件包含简化版 Encoder/Decoder/Layer（类似 DETR 的实现）。
- 支持 `torch.utils.checkpoint`（`enable_cp=True`）以节省显存（以计算换显存）。
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp


class LayerNorm(nn.LayerNorm):
    """
    为 fp16 训练/推理做兼容的 LayerNorm。

    做法：在 `forward` 中把输入临时 cast 到 fp32 进行归一化计算，再 cast 回原 dtype，
    避免半精度下数值不稳定。
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)


class GEN(nn.Module):
    """
    GEN：Human-Object Interaction 的生成/融合模块。

    输入：来自检测/编码器的空间特征 `src`（B,C,H,W）、mask/pos_embed，以及 human/object queries。
    输出：human/object 的 decoder hidden states、交互 hidden states、以及 CLIP 相关特征/打分。

    备注：`self.hoi_cls` 需要在外部被赋值（通常为归一化后的 HOI 文本原型/类别向量）。
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_inter_dec_layrs=3,
                 return_intermediate_dec=False, num_queries=64, clip_dim=768, enable_cp=False):
        super().__init__()

        # Encoder：对 flatten 后的空间 token 序列做自注意力编码。
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, enable_cp)
        encoder_norm = LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Instance decoder：分别对 human/object 两组 query 解码（在同一个 decoder 内拼接成 2*num_queries）。
        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before, False)
        instance_decoder_norm = LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        # Interaction decoder：保留的另一套 decoder（当前 forward 中主路径已改为 CLIP fusion decoder）。
        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, False)
        interaction_decoder_norm = LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        # CLIP fusion decoder：以交互 query 为 tgt，CLIP visual tokens 为 memory，同时可引入 `sup_memory` 辅助融合。
        clip_interaction_decoder_layer = TransformerDecoderFusionLayer(clip_dim, nhead, dim_feedforward,
                                                                       dropout, activation, normalize_before, enable_cp)
        clip_interaction_decoder_norm = LayerNorm(clip_dim)
        self.clip_interaction_decoder = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_inter_dec_layrs,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.inter_guided_embedd = nn.Embedding(num_queries, clip_dim)
        # 将 d_model 维度的 query 映射到 CLIP 维度，便于与 CLIP 特征对齐/融合。
        self.queries2spacial_proj = nn.Linear(d_model, clip_dim)
        self.queries2spacial_proj_norm = LayerNorm(clip_dim)

        # 将 encoder memory（d_model）映射到 CLIP 维度，作为 fusion decoder 的 `sup_memory`。
        self.obj_class_fc = nn.Linear(d_model, clip_dim)
        self.obj_class_ln = LayerNorm(clip_dim)

        # CLIP打分用的 HOI，在hoiclip.py中赋值self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)
        self.hoi_cls = None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.inter_guided_embedd.weight)

    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src):
        # 将特征图从 (B,C,H,W) flatten 成序列 (HW,B,C)，以适配 `nn.MultiheadAttention` 的输入格式。
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]

        # 为 human/object queries 加上引导位置 embedding，并扩展到 batch 维度。
        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        # 将两类 query 拼接成 2*num_queries（前半段 human，后半段 object）。
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)
        # Decoder 的初始 tgt 通常为全 0，通过注意力逐层更新。
        ins_tgt = torch.zeros_like(ins_query_embed)
        # Encoder 输出 `memory`：(HW,B,d_model)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # 实例解码：输出 shape（当 return_intermediate=True）为 (L, 2*num_queries, B, d_model)
        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        # 转成 (L,B,2*num_queries,d_model)，便于按 query 维度切分 human/object。
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        # original
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs_ori = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                         pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs_ori = inter_hs_ori.transpose(1, 2)
        # 将 encoder memory 对齐到 CLIP 维度，作为 fusion decoder 的辅助记忆（sup_memory）。
        memory = self.obj_class_ln(self.obj_class_fc(memory))   #self.obj_class_fc: map feature to 768, self.obj_class_ln:layernorm

        # h_hs_detached = h_hs.detach()

        # 交互 query：用 human/object 的实例表征做平均，并取最后一层作为融合 decoder 的输入。
        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1])  # 将交互 query 从 d_model 映射768，以便与 CLIP 特征对齐。
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        # inter_hs = inter_hs + self.inter_guided_embedd.weight.unsqueeze(0).repeat(bs, 1, 1)

        dtype = inter_hs.dtype

        # CLIP 图像编码：返回全局 cls 特征与视觉 token（具体形状取决于 clip_model 的实现）。
        clip_cls_feature, clip_visual = clip_model.encode_image(clip_src)
        # 常见做法：对全局特征做 L2 归一化，便于余弦相似度/点积相似度。
        clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
        clip_cls_feature = clip_cls_feature.to(dtype)   #clip_cls_feature图像全局特征
        clip_visual = clip_visual.to(dtype)
        with torch.no_grad():
            # CLIP zero-shot 打分：图像全局特征与 HOI 文本原型做点积相似度。
            clip_hoi_score = clip_cls_feature @ self.hoi_cls.T
            # obj_score = clip_cls_feature @ self.obj_cls.T
            # obj_hoi_score = obj_score @ self.obj2hoi_proj

            # verb_score = clip_cls_feature @ self.verb_cls.T
            # verb_hoi_score = verb_score @ self.verb2hoi_proj
            # clip_hoi_score += verb_hoi_score * 0.1
            # ignore_idx = clip_hoi_score.sort(descending=True).indices[:, self.topk:]
            # for idx, igx in enumerate(ignore_idx):
            #     clip_hoi_score[idx][igx] *= 0
            clip_hoi_score = clip_hoi_score.unsqueeze(1)

        # 将 CLIP 全局特征扩展到每个 query，便于上层模块直接做拼接/融合使用。
        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)

        # 融合解码：tgt 为交互 queries，memory 为 CLIP visual tokens，sup_memory 为 encoder memory（CLIP 维）。
        inter_hs = self.clip_interaction_decoder(inter_hs.permute(1, 0, 2),
                                                 clip_visual.permute(1, 0, 2), sup_memory=memory)
        # 输出再通过 `clip_proj` 投影（通常用于对齐到最终的文本/类别空间）。
        inter_hs = inter_hs @ clip_proj.to(dtype)
        inter_hs = inter_hs.permute(0, 2, 1, 3)

        # add
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        # #torch.Size([3, 64, 8, 256])
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                     pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs = inter_hs.transpose(1, 2)

        return h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual @ clip_proj.to(dtype)


class TransformerEncoder(nn.Module):
    """Encoder：按层堆叠 `TransformerEncoderLayer`。输入/输出均为 (S,B,C) 序列格式。"""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderCLIP(nn.Module):
    """
    Decoder（CLIP 版本）：按层堆叠 `TransformerDecoderFusionLayer`。

    与普通 decoder 的差异：
    - 支持传入 `sup_memory`，在 cross-attn 时额外融合一份辅助 memory。
    - 兼容两种 `tgt` 输入：
      - (L, NQ, B, C)：每层一个输入（例如从别处生成的逐层 query）；
      - (NQ, B, C)：所有层共用同一份输入（本项目主路径使用该形式）。
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # 约定输入形状（与 `nn.MultiheadAttention` 一致）：
        # - `tgt`  : (NQ,B,C) 或 (L,NQ,B,C)（每层一个 tgt）
        # - `memory`: (S,B,C)（这里通常是 CLIP visual tokens）
        # - `sup_memory`: (S2,B,C)（可选的辅助 memory，例如 encoder memory 映射到 CLIP 维）
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(output.shape) == 4:
                # 若传入逐层 tgt（L,NQ,B,C），每一层取对应的 output[i] 作为该层输入。
                output = output[i]
            else:
                # 常规情况：所有层共享同一份 tgt（NQ,B,C）。
                # 该项目主路径会把“交互 query”作为 tgt 喂入每一层。
                output = output
            # 注意：这里没有使用 `query_pos`，融合层内部默认只用 `tgt` 本身；
            # `pos` 会用于给 memory/sup_memory 加位置编码（若调用方提供）。
            output = layer(output, memory, sup_memory=sup_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                # 每层输出先做一次 norm，便于做深监督或可视化分析。
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # 返回 (L,NQ,B,C)，与 DETR 风格 decoder 的返回一致。
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):
    """
    标准 TransformerDecoder：按层堆叠 `TransformerDecoderLayer`。

    支持 `query_pos` 为逐层提供的位置编码：
    - (L, NQ, B, C) 时每层取 `query_pos[i]`
    - 否则每层共享同一份 `query_pos`
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    单层 Encoder block：Self-Attention + FFN（支持 pre-norm / post-norm）。

    当 `enable_cp=True` 时，对注意力子图使用 checkpoint 以节省显存。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.enable_cp:
            def _inner_forward(args):
                src_inner, q_inner, k_inner, src_mask_inner, src_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=src_inner, attn_mask=src_mask_inner,
                                      key_padding_mask=src_key_padding_mask_inner)[0]
                return src_inner

            src2 = cp.checkpoint(_inner_forward, (src, q, k, src_mask, src_key_padding_mask))
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderFusionLayer(nn.Module):
    """
    融合型 Decoder layer：Self-Attention + Cross-Attention（双 memory）+ FFN。

    - `memory`：通常是 CLIP visual tokens（作为主要 cross-attn 的 key/value）。
    - `sup_memory`：辅助的 memory（例如 encoder memory 投影到 CLIP 维度），再做一次 cross-attn。

    forward_post 中会分别计算两次 cross-attn（对 `memory` 与 `sup_memory`），并将两者相加后送入 FFN。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # post-norm 结构：
        # 1) Self-Attn：让各个 query（tgt）之间交互
        # 2) Cross-Attn(memory)：从 CLIP visual tokens 取信息
        # 3) Cross-Attn(sup_memory)：从辅助 memory 取信息（例如 encoder memory）
        # 4) FFN：前馈网络更新
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        # cross-attn 到主 memory（通常是 CLIP visual tokens）
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        # cross-attn 到辅助 memory（sup_memory），用于引入额外视觉/检测特征的上下文。
        # 注：当前实现复用同一个 `multihead_attn` 模块进行两次 cross-attn。
        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt3 = tgt + self.dropout4(tgt3)
        # tgt3 = self.norm4(tgt3)
        tgt3 = tgt + self.dropout2(tgt3)
        tgt3 = self.norm2(tgt3)

        # 将两路 cross-attn 的结果相加融合，再进入 FFN。
        tgt = tgt2 + tgt3

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer(nn.Module):
    """
    标准 Decoder layer：Self-Attention + Cross-Attention + FFN（支持 pre-norm / post-norm）。

    当 `enable_cp=True` 时，对注意力子图使用 checkpoint 以节省显存。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    # 深拷贝 N 份 module，构建多层堆叠结构。
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    # 从 args 构建 GEN（常用于与训练脚本/配置系统对接）。
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        num_inter_dec_layrs=args.inter_dec_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
