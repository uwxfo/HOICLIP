import math
import os
import random
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.datasets_gen.hico_eval_triplet import HICOEvaluator as HICOEvaluator_gen
from datasets.datasets_gen.vcoco_eval import VCOCOEvaluator as VCOCOEvaluator_gen
from datasets.datasets_gen.vidhoi_eval import VidHOIEvaluator
import json
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import time


def _prepare_hoi_eval_batch(results, targets, keep_clip_visual=False):
    """Drop tensors that are not consumed by the HOI evaluators before gathering."""
    compact_results = []
    for result in results:
        result = dict(result)
        result.pop('clip_logits', None)
        if not keep_clip_visual:
            result.pop('clip_visual', None)
        compact_results.append(result)

    # clip_inputs is an image tensor used by model.forward, not by any evaluator.
    compact_targets = [
        {k: v for k, v in target.items() if k not in ('clip_inputs', 'raw_img')}
        for target in targets
    ]
    return compact_results, compact_targets


def _gather_hoi_eval_batch_on_main(results, targets, preds, gts):
    """Run the required collectives on every rank but retain global data only on rank 0."""
    gathered_results = utils.all_gather(results)
    gathered_targets = utils.all_gather(targets)
    if utils.is_main_process():
        preds.extend(itertools.chain.from_iterable(gathered_results))
        gts.extend(itertools.chain.from_iterable(gathered_targets))


def _prepare_future_hoi_eval_batches(preds, gts):
    """Build one prediction/GT list per anticipation horizon."""
    horizons = [1, 3, 5, 7]
    future_preds = {horizon: [] for horizon in horizons}
    future_gts = {horizon: [] for horizon in horizons}
    for pred, gt in zip(preds, gts):
        if 'hoi_scores_future' not in pred or 'future' not in gt:
            continue
        scores = pred['hoi_scores_future']
        if scores.shape[0] % len(horizons) != 0:
            raise ValueError(
                f"Future query count {scores.shape[0]} is not divisible by {len(horizons)}"
            )
        scores_by_horizon = scores.chunk(len(horizons), dim=0)
        for horizon, future_hois in zip(gt['future'], gt['hois_future']):
            horizon = int(horizon)
            slot = horizons.index(horizon)
            pred_horizon = {
                key: value for key, value in pred.items()
                if key != 'hoi_scores_future'
            }
            pred_horizon['hoi_scores'] = scores_by_horizon[slot]
            gt_horizon = {
                'boxes': gt['boxes'],
                'labels': gt['labels'],
                'hois': future_hois,
            }
            future_preds[horizon].append(pred_horizon)
            future_gts[horizon].append(gt_horizon)
    return future_preds, future_gts


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, lr_scheduler=None,
                    gradient_accumulation_steps=1, enable_amp=False, no_training=False, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels') and False:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if enable_amp:
        print('\nEnable half precision training\n')

    # scaler = GradScaler()
    # debug
    debug_count = 0
    step = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if no_training:
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
            clip_img = torch.stack([v['clip_inputs'] for v in targets])
            # with autocast():
            obj_feature, hoi_feature, verb_feature = model(samples, clip_input=clip_img, targets=targets)

            metric_logger.update(loss=0)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=0)
            elif hasattr(criterion, 'loss_hoi_labels'):
                metric_logger.update(hoi_class_error=0)
            else:
                metric_logger.update(obj_class_error=0)
            metric_logger.update(lr=0)
            continue

        samples = samples.to(device)
        file_names = [{'filename': i['filename']} for i in targets]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
        for t, f in zip(targets, file_names):
            t.update(f)
        clip_img = torch.stack([v['clip_inputs'] for v in targets])

        outputs = model(samples, clip_input=clip_img, targets=targets)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        delay_unscale = (step + 1) % gradient_accumulation_steps != 0
        losses = losses / gradient_accumulation_steps
        if enable_amp:
            raise NotImplementedError
            # with amp.scale_loss(losses, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            #     scaled_loss.backward()
        else:
            losses.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_norm > 0:
                if enable_amp:
                    pass
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler:
            lr_scheduler.iter_step()

        step += 1

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels') and False:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            if 'hoi_class_error' in loss_dict_reduced:
                metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
            else:
                metric_logger.update(hoi_class_error=-1)
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # trick for generate verb
    if no_training:
        from datasets.static_hico import HOI_IDX_TO_ACT_IDX, HOI_IDX_TO_OBJ_IDX
        hoi_feature = hoi_feature / hoi_feature.norm(dim=1, keepdim=True)
        obj_feature = obj_feature / obj_feature.norm(dim=1, keepdim=True)

        y_verb = [HOI_IDX_TO_ACT_IDX[i] for i in range(600)]
        y_obj = [HOI_IDX_TO_OBJ_IDX[i] for i in range(600)]

        # composite image feature verb + text feature object
        obj_human = []
        for i in range(600):
            obj_human.append(obj_feature[y_obj[i]])
        obj_human = torch.stack(obj_human)
        verb_human = hoi_feature - obj_human

        verb_feature = torch.zeros(117, 512)
        for idx, v in zip(y_verb, verb_human):
            verb_feature[idx] += v

        for i in range(117):
            verb_feature[i] /= y_verb.count(i)

        v_feature = verb_feature / verb_feature.norm(dim=-1, keepdim=True)
        torch.save(v_feature, f'./verb_{args.dataset_file}.pth')
        exit()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    counter = 0

    use_cuda = torch.device(device).type == 'cuda'
    inf_total_time = 0
    timed_batches = 0
    warmup_batches = int(os.environ.get('EVAL_WARMUP_BATCHES', '0'))
    max_batches = getattr(args, 'eval_max_batches', 0)   # 0 = whole split
    if use_cuda:
        # Peak stats start here: model weights already on GPU are still counted.
        torch.cuda.reset_peak_memory_stats(device)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        if counter == 0 and os.environ.get('EVAL_PROFILE', '0') == '1':
            # One-off, untimed profiling pass on the first batch (EVAL_PROFILE=1, eval only).
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Params: total {n_total / 1e6:.2f} M, trainable {n_train / 1e6:.2f} M "
                  f"(frozen HOICLIP is included in total)")
            # FlopCounterMode only counts aten matmul/conv/sdpa-style ops (2*MACs); custom
            # CUDA ops (e.g. deformable attention) and cuDNN LSTM are not counted -> lower bound.
            from torch.utils.flop_counter import FlopCounterMode
            # enable_grad: the module tracker asserts on grad-less parameters under no_grad.
            with torch.enable_grad(), FlopCounterMode(display=False) as flop_counter:
                model(samples, is_training=False, clip_input=clip_img, targets=targets)
            print(f"GFLOPs (aten ops only, per batch of {args.batch_size}; frames in batch: "
                  f"{samples.tensors.shape[0]}): {flop_counter.get_total_flops() / 1e9:.2f}")
            if use_cuda:
                # the profiling pass must not leak into the reported peak memory
                torch.cuda.reset_peak_memory_stats(device)

        if use_cuda:
            # CUDA is async: finish the H2D copies so they are not billed to the model.
            torch.cuda.synchronize(device)
        start_time = time.time()
        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        if use_cuda:
            # wait for queued kernels, otherwise time.time() only measures launch
            torch.cuda.synchronize(device)
        if counter >= warmup_batches:
            # first batches pay CUDA init / cuDNN autotune / allocator warm-up: run them untimed
            inf_total_time += time.time() - start_time
            timed_batches += 1

        results, eval_targets = _prepare_hoi_eval_batch(
            results, targets,
            keep_clip_visual=bool(args.training_free_enhancement_path),
        )
        _gather_hoi_eval_batch_on_main(results, eval_targets, preds, gts)

        counter += 1
        if counter >= 20 and args.no_training:
            break
        if max_batches > 0 and counter >= max_batches:
            # efficiency-only runs (e.g. random weights, where mAP is meaningless)
            break

    if use_cuda:
        # allocated = tensors in use at peak; reserved = caching-allocator pool (closer to nvidia-smi).
        print(f"Peak GPU memory: allocated {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB, "
              f"reserved {torch.cuda.max_memory_reserved(device) / 2**20:.0f} MiB")
    if timed_batches > 0:
        # Per batch; with batch_size == 1 this is per sample (one clip of max_length frames).
        print(f"Avg inference time: {inf_total_time / timed_batches * 1000:.2f} ms/batch, "
              f"FPS: {timed_batches / inf_total_time:.2f} batches/s over {timed_batches} timed batches "
              f"({warmup_batches} warm-up batches excluded; per-sample only when batch_size == 1)")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Only rank 0 retained the global predictions. Other ranks wait here while rank 0
    # computes mAP, then receive the small stats dictionary via broadcast.
    if not utils.is_main_process():
        return utils.broadcast_object(None)

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    """
    For zero-shot enhancement
    args.training_free_enhancement_path is the path to store performance for different hyper-parameter
    """
    root = os.path.join(args.output_dir, args.training_free_enhancement_path)
    if args.training_free_enhancement_path:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========The great hyperparameter tuning begins============\n'
            print(log)
            f.write(log)

        test_pred = copy.deepcopy(preds)

        # testing
        if dataset_file == 'hico':
            evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets,
                                          data_loader.dataset.correct_mat, args=args)
        elif dataset_file in ('vidhoi', 'ag'):
            # 调参场景不传 output_dir：per-triplet AP 明细只由下面的主评估写盘，
            # 否则这里的 baseline 和 topk 循环会反复覆盖同名 json。
            evaluator = VidHOIEvaluator(test_pred, gts, args.hoi_path,
                                        use_nms_filter=args.use_nms_filter,
                                        thres_nms=args.thres_nms, nms_alpha=args.nms_alpha,
                                        nms_beta=args.nms_beta)
        else:
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
        stats = evaluator.evaluate()

        _model = model.module if hasattr(model, 'module') else model
        # HOICLIP_LSTM wraps HOICLIP as self.hoiclip; fall back to that if needed.
        _inner = getattr(_model, 'hoiclip', _model)
        text_hoi_feature = _inner.transformer.hoi_cls
        spatial_feature = torch.cat([i['clip_visual'].unsqueeze(0) for i in preds])
        spatial_feature /= spatial_feature.norm(dim=-1, keepdim=True)
        spatial_cls = spatial_feature[:, 0, :]  # M, c

        cls_scores = spatial_cls @ text_hoi_feature.to(spatial_cls.device).T
        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========Baseline Performance============\n{stats}\n============================\n'
            print(log)
            f.write(log)

        best_performance_1 = 0
        for a in [1]:
            for co in [1.0]:
                for topk in [10, 20, 30, 40, 50]:
                    print(f'current at topk: {topk} as: {a}')
                    test_pred = copy.deepcopy(preds)
                    clip_hoi_score = cls_scores.clone()  # 避免原地修改 cls_scores
                    # clip_hoi_score /= (1 + alpha + beta)
                    clip_hoi_score_ori = clip_hoi_score.clone()

                    ignore_idx = clip_hoi_score.sort(descending=True).indices[:, topk:]
                    for idx, igx in enumerate(ignore_idx):
                        clip_hoi_score[idx][igx] *= 0
                    clip_hoi_score = clip_hoi_score.unsqueeze(1)

                    # update logits
                    for i in range(len(test_pred)):
                        test_pred[i]['hoi_scores'] += clip_hoi_score[i].sigmoid() * co
                    # testing
                    if dataset_file == 'hico':
                        evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                                      data_loader.dataset.non_rare_triplets,
                                                      data_loader.dataset.correct_mat, args=args)
                    elif dataset_file in ('vidhoi', 'ag'):
                        evaluator = VidHOIEvaluator(test_pred, gts, args.hoi_path,
                                                    use_nms_filter=args.use_nms_filter,
                                                    thres_nms=args.thres_nms, nms_alpha=args.nms_alpha,
                                                    nms_beta=args.nms_beta)
                    else:
                        evaluator = VCOCOEvaluator_gen(test_pred, gts, data_loader.dataset.correct_mat,
                                                       use_nms_filter=args.use_nms_filter)
                    stats = evaluator.evaluate()
                    if dataset_file == 'hico':
                        re_map = stats['mAP']
                    elif dataset_file == 'vcoco':
                        re_map = stats['mAP_all']
                    elif dataset_file in ('hoia', 'vidhoi', 'ag'):
                        re_map = stats['mAP']
                    else:
                        raise NotImplementedError

                    if best_performance_1 < re_map:
                        best_performance_1 = re_map

                        with open(os.path.join(root, 'log.txt'), 'a') as f:
                            log = f'sigmoid after topk: {topk} as: {a} co: {co}' \
                                  f'\n performance: {stats}\n'
                            print(log)
                            f.write(log)

    if dataset_file == 'hico':
        if args.dataset_root == 'GEN':
            evaluator = HICOEvaluator_gen(preds, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat,
                                          args=args)
    elif dataset_file == 'vcoco':
        if args.dataset_root == 'GEN':
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
    elif dataset_file in ('vidhoi', 'ag'):
        # 只有 --eval 模式才落 per-triplet AP 明细；训练中每 epoch 都会走到这里，
        # 写盘只会被下个 epoch 覆盖。汇总指标（mAP/mAP_rare/...）始终随 stats 返回。
        evaluator = VidHOIEvaluator(preds, gts, args.hoi_path,
                                    use_nms_filter=args.use_nms_filter,
                                    thres_nms=args.thres_nms, nms_alpha=args.nms_alpha,
                                    nms_beta=args.nms_beta,
                                    output_dir=args.output_dir if args.eval else "",
                                    suffix='_current')
    else:
        raise NotImplementedError
    print(f'Evaluating {len(preds)} frames...')
    start_time = time.time()
    stats = evaluator.evaluate()
    if dataset_file in ('vidhoi', 'ag') and any(
        'hoi_scores_future' in pred for pred in preds
    ):
        future_preds, future_gts = _prepare_future_hoi_eval_batches(preds, gts)
        for horizon in (1, 3, 5, 7):
            if not future_preds[horizon]:
                continue
            print(f'Evaluating {len(future_preds[horizon])} frames at future {horizon}...')
            evaluator_future = VidHOIEvaluator(
                future_preds[horizon], future_gts[horizon], args.hoi_path,
                use_nms_filter=args.use_nms_filter,
                thres_nms=args.thres_nms,
                nms_alpha=args.nms_alpha,
                nms_beta=args.nms_beta,
                output_dir=args.output_dir if args.eval else "",
                suffix=f'_f{horizon}',
            )
            stats_future = evaluator_future.evaluate()
            stats.update({f'{key}_f{horizon}': value for key, value in stats_future.items()})
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time computing mAP: {}'.format(total_time_str))

    return utils.broadcast_object(stats)


# Queries processed at once in the noise-OR propagation; bounds the [q, N_hoi, N_hoi] GPU
# temporary (at N_hoi=1127, chunk=32 -> ~163MB, trivial for a modern GPU).
_NOISE_OR_QUERY_CHUNK = 32


@torch.no_grad()
def evaluate_hoi_transition(dataset_file, model, postprocessors, data_loader,
                            subject_category_id, device, args):
    """
    Evaluate HOICLIP with a triplet transition matrix for future-horizon anticipation.

    The dataset yields window clips, but the model stays image-based: only the clip's
    last frame is fed to it. Current-frame HOI scores are then propagated to future
    horizons [1,3,5,7] with a noise-OR over the triplet transition matrix.
    """
    from datasets.priors.transition_matrix_triplet import extract_transition_matrix

    model.eval()

    # Trans[h]: [N_hoi, N_hoi] with T[i, j] = P(current triplet i | future triplet j).
    # Kept on `device` since the noise-OR propagation below runs on GPU.
    Trans = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
             for k, v in extract_transition_matrix(
                 args.transition_matrix_type, dataset_file
             ).items()}

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print("It is running evaluate_hoi_transition......")

    preds = []
    gts = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # collate gives NestedTensor(B*T, 3, H, W); keep the last frame of every clip
        src, mask = samples.decompose()
        n, c, h, w = src.shape
        assert n % args.max_length == 0, f"{n} must be divisible by max_length"
        bs = n // args.max_length
        src_last = src.reshape(bs, -1, c, h, w)[:, -1, ...]   # [bs, 3, H, W]
        mask_last = mask.reshape(bs, -1, h, w)[:, -1, ...]    # [bs, H, W]
        samples = utils.NestedTensor(src_last, mask_last).to(device)

        # clip_inputs is (T, 3, 224, 224) per clip -> take the same last frame
        clip_img = torch.stack([v['clip_inputs'][-1] for v in targets]).to(device)

        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        results, eval_targets = _prepare_hoi_eval_batch(results, targets)
        _gather_hoi_eval_batch_on_main(results, eval_targets, preds, gts)

    metric_logger.synchronize_between_processes()

    if not utils.is_main_process():
        return utils.broadcast_object(None)

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    def _make_evaluator(p, g, output_dir=""):
        return VidHOIEvaluator(p, g, args.hoi_path,
                               use_nms_filter=args.use_nms_filter,
                               thres_nms=args.thres_nms, nms_alpha=args.nms_alpha,
                               nms_beta=args.nms_beta, output_dir=output_dir)

    horizons = [1, 3, 5, 7]
    preds_future = [[] for _ in horizons]
    gts_future = [[] for _ in horizons]

    for pred, gt in zip(preds, gts):
        if 'future' not in gt:
            continue
        hoi_scores = pred['hoi_scores']            # [Q, N_hoi]
        for f, hois in zip(gt['future'], gt['hois_future']):
            idx = horizons.index(f)
            T = Trans[f]
            if T.shape[0] != hoi_scores.shape[1]:
                raise ValueError(
                    f"transition matrix for horizon {f} is {tuple(T.shape)} but the model "
                    f"predicts {hoi_scores.shape[1]} HOI classes")
            # noise-OR: P_future[q,j] = 1 - prod_i(1 - T[i,j] * hoi_scores[q,i])
            # Computed on `device` (T already lives there); the [q, N_hoi, N_hoi]
            # intermediate is ~325MB at q=64/N_hoi=1127, so chunk over queries to keep
            # peak GPU memory bounded. Result is moved back to CPU for storage since
            # thousands of these accumulate in preds_future across the full val set.
            chunks = []
            for start in range(0, hoi_scores.shape[0], _NOISE_OR_QUERY_CHUNK):
                s = hoi_scores[start:start + _NOISE_OR_QUERY_CHUNK].to(device)
                terms = 1.0 - T.unsqueeze(0) * s.unsqueeze(2)   # [q, N_hoi, N_hoi]
                chunks.append((1.0 - terms.prod(dim=1)).cpu())  # [q, N_hoi]
            future_hoi_scores = torch.cat(chunks, dim=0)        # [Q, N_hoi]

            p_future = copy.deepcopy(pred)
            p_future['hoi_scores'] = future_hoi_scores
            # VidHOIEvaluator reads GT hois as a numeric (N,3) tensor, but hois_future
            # is a list of dicts -> convert here rather than touching the evaluator.
            g_future = {
                'boxes': gt['boxes'], 'labels': gt['labels'], 'id': gt['id'],
                'filename': gt.get('filename', ''),
                'hois': torch.as_tensor(
                    [(x['subject_id'], x['object_id'], x['category_id']) for x in hois],
                    dtype=torch.int64),
            }
            preds_future[idx].append(p_future)
            gts_future[idx].append(g_future)

    print(f'Detection: evaluating {len(preds)} clips...')
    # 只有当前帧检测这一路写 AP 明细；4 个 future horizon 共用同一目录，写了会互相覆盖。
    stats = _make_evaluator(preds, gts, output_dir=args.output_dir).evaluate()

    for f, p_f, g_f in zip(horizons, preds_future, gts_future):
        if not p_f:
            print("No items in future ", f)
            continue
        print(f"future {f}: totally {len(p_f)} items")
        stats_f = _make_evaluator(p_f, g_f).evaluate()
        stats.update({k + f'_f{f}': v for k, v in stats_f.items()})

    return utils.broadcast_object(stats)
