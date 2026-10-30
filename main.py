import argparse
import datetime
import json
import random
import time
from pathlib import Path
import logging
import debugpy

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi, evaluate_hoi_transition
from models import build_model
import os

from util.scheduler import CosineAnnealingLRWarmup, MultiStepLRWarmup


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_lstm_checkpoint_layout(checkpoint_model, checkpoint_path):
    is_lstm = any(
        key.startswith(('lstm.', 'curr_hoi_head.', 'future_hoi_heads.'))
        for key in checkpoint_model
    )
    if is_lstm and '_temporal_layout_version' not in checkpoint_model:
        raise RuntimeError(
            f'{checkpoint_path} is an old HOICLIP_LSTM checkpoint trained with the '
            'incorrect temporal layout. Restart with --pretrained pointing to a plain '
            'HOICLIP checkpoint instead of resuming it.'
        )


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_clip', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval_each', default=4, type=int)
    parser.add_argument('--eval_each_lr_drop', default=2, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--with_mimic', action='store_true',
                        help="Use clip feature mimic")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi', default=1, type=float,
                        help="Hoi class coefficient")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--hoi_loss_coef', default=2, type=float)
    parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--train_ratio', default=1.0, type=float,
                        help='Fraction of training annotations to use (VidHOI)')
    parser.add_argument('--val_ratio', default=1.0, type=float,
                        help='Fraction of validation annotations to use (VidHOI, unused)')

    # HOICLIP_LSTM clip-mode arguments
    parser.add_argument('--max_length', default=6,   type=int,
                        help='Clip window length T for HOICLIP_LSTM')
    parser.add_argument('--min_length', default=6,   type=int,
                        help='Minimum window length for VidHOIVideoDataset')
    parser.add_argument('--min_future', default=3,   type=int,
                        help='Minimum future length for VidHOIVideoDataset')
    parser.add_argument('--sample_mode', default='window', type=str,
                        choices=['window', 'clip'],
                        help='Sampling mode for VidHOIVideoDataset')
    parser.add_argument('--transition_matrix_type', type=str, default=None,
                        help='Enable transition-matrix anticipation eval and select the triplet '
                             'matrix variant (e.g. detant, detant_v2, orig). The dataset switches '
                             'to window clips while the model still sees only the last frame.')
    parser.add_argument('--lstm_hidden_dim', default=512, type=int,
                        help='LSTM hidden dimension (defaults to clip_embed_dim)')
    parser.add_argument('--lstm_num_layers', default=2, type=int,
                        help='Number of LSTM layers')
    parser.add_argument('--freeze_temporal', action='store_true',
                        help='HOICLIP_LSTM cross-dataset linear probe: additionally freeze the '
                             'LSTM, leaving only curr_hoi_head and future_hoi_heads trainable.')
    parser.add_argument('--anticipation_hoi_coef', default=0.8, type=float,
                        help='Total loss coefficient distributed over future horizons.')
    parser.add_argument('--anticipation_horizon_gamma', default=0.7, type=float,
                        help='Geometric decay for future horizons [1,3,5,7].')
    parser.add_argument('--anticipation_ramp_epochs', default=8, type=int,
                        help='Epochs used to ramp future HOI loss weights to full strength.')
    parser.add_argument('--anticipation_ramp_start_factor', default=0.25, type=float,
                        help='Future-loss multiplier at epoch zero.')
    parser.add_argument('--detection_frame_weighting', default='linear',
                        choices=['uniform', 'linear'],
                        help='Observed-frame weighting for clip matching and current HOI loss.')
    parser.add_argument('--detection_frame_weight_power', default=1.0, type=float,
                        help='Exponent applied to linear observed-frame weights.')

    parser.add_argument('--output_dir', default='',
                        help='base directory for output; a timestamped subdirectory is created inside')
    parser.add_argument('--name', default='', type=str,
                        help='optional run name prepended to the timestamp subdirectory')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='clip pretrained model path')
    parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # zero/few shot type
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    parser.add_argument('--del_unseen', action='store_true', help='')
    # old parameter
    parser.add_argument('--fix_backbone_mode', nargs='+', default=[], help='fix (part of) backbone')

    # others
    parser.add_argument('--use_ddp', default=1, type=int)
    parser.add_argument('--with_random_shuffle', default=2, type=int, help='Time of random shuffle of annotation')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--opt_sched', default='multiStep', type=str, help='type of scheduler')
    parser.add_argument('--no_clip_cls_init', action='store_true',
                        help='not init classifier weight with clip text encoder')
    parser.add_argument('--enable_amp', action='store_true', help='')
    parser.add_argument('--opt_level', default='O2', help='half precision optimization level', choices=('O1', 'O2'))
    parser.add_argument('--fix_clip_label', action='store_true', help='')
    parser.add_argument('--with_rec_loss', action='store_true', help='')
    parser.add_argument('--rec_loss_coef', default=2, type=float)
    parser.add_argument('--no_training', action='store_true', help='')
    parser.add_argument('--dataset_root', default='GEN', help='')
    parser.add_argument('--model_name', default='GEN', help='')
    parser.add_argument('--eval_location', action='store_true', help='')
    # DAB
    parser.add_argument('--enable_cp', action='store_true',
                        help="use checkpoint to save memory")
    parser.add_argument('--no_fix_clip_linear', action='store_true',
                        help="")
    parser.add_argument('--analysis', action='store_true')

    # tmp args
    parser.add_argument('--alternative', default=1, type=int)
    parser.add_argument('--eval_each_ap', action='store_true')
    parser.add_argument('--topk_hoi', default=10, type=int)
    parser.add_argument('--inter_dec_layers', default=3, type=int)

    # verb setting
    parser.add_argument('--verb_pth', default='', help='location for predefined verb feature', type=str)
    parser.add_argument('--verb_weight', default=0.5, type=float)
    # fractional training
    parser.add_argument('--frac', default=-1., type=float)

    # validation split
    parser.add_argument('--validation_split', default=-1., type=int)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)

    # zero shot enhancement
    parser.add_argument('--training_free_enhancement_path', default='', type=str)
    parser.add_argument('--debug', action='store_true', help='use debug mode for training.')
    return parser


def main(args):
    if args.use_ddp == 1:
        utils.init_distributed_mode(args)
    else:
        args.distributed = False

    args.save_points = []  # populated via --save_points if the arg is ever added back
    args.anticipation=True
    if args.resume and not args.resume.startswith('https') and not os.path.exists(args.resume):
        raise FileNotFoundError(f'Resume checkpoint does not exist: {args.resume}')
    print('setting up seeds')
    setup_seed(233)

    # sys.exit(0)

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # --- output_dir logic (mirrors RLIPVISTR/main.py) ---
    # eval  → results land next to the loaded checkpoint, in an "eval" subdirectory
    # resume → continue logging in the same directory as the checkpoint (no new timestamp)
    # fresh  → create a new timestamped subdirectory under args.output_dir
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    '''
    if args.eval:
        if args.resume and os.path.exists(args.resume):
            output_dir = Path(args.resume).parent / "eval" / timestamp 
        elif args.pretrained:
            output_dir = Path(args.pretrained).parent / "eval" / timestamp 
        else:
            output_dir = Path(args.output_dir) / "eval"
    '''
    if args.eval:
        output_dir = Path(args.output_dir) / timestamp
    elif args.resume and os.path.exists(args.resume):
        output_dir = Path(args.resume).parent
    else:
        run_name = f"{args.name}_{timestamp}" if args.name else timestamp
        output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Propagate the resolved path back so all later Path(args.output_dir) calls
    # see the correct (possibly timestamped) directory instead of the base dir.
    args.output_dir = str(output_dir)
    if args.debug:
        if args.rank == 0:
            debugpy.listen(("0.0.0.0", 5678))
            print("Wait Debugger Connection")
            debugpy.wait_for_client()
            print("Debugger Attached")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Transition-based eval keeps the model image-based: build (and later load) it as a plain
    # frame-mode HOICLIP, then restore window mode so the dataset still yields clips.
    if getattr(args, 'transition_matrix_type', None) is not None:
        _saved_sample_mode = args.sample_mode
        args.sample_mode = 'frame'
        model, criterion, postprocessors = build_model(args)
        args.sample_mode = _saved_sample_mode
    else:
        model, criterion, postprocessors = build_model(args)
    model.to(device)
    print('****************')
    # print(model)
    print(args.model_name)
    print('****************')

    model_without_ddp = model
    if args.distributed:
        if args.enable_amp:
            raise NotImplementedError
        else:
            # Plain HOICLIP always has params that never receive a gradient in this
            # training config (CLIP's text encoder — used only once at classifier-weight
            # init, not in forward — plus interaction_decoder/clip_interaction_decoder,
            # unused branches unless a different, currently-unused config flag enables
            # them). This is dataset-independent — it reproduces identically for
            # dataset_file=vidhoi — and crashes DDP's gradient-bucket check on the second
            # iteration without this flag.
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

    # model = convert_syncbn_model(model)

    for name, p in model.named_parameters():
        if 'eval_visual_projection' in name:
            p.requires_grad = False

    if args.fix_clip:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name or 'clip_model' in name:
                p.requires_grad = False

    if args.no_fix_clip_linear:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = True

    # HOICLIP_LSTM：HOICLIP 已冻结，只优化 LSTM + 预测头
    if args.model_name == 'HOICLIP_LSTM':
        param_dicts = [
            {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    elif args.ft_clip_with_small_lr:
        if args.with_obj_clip_label and args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n
                            and 'clip_model' not in n and p.requires_grad and 'T5_model' not in n]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'visual_projection' in n or 'obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'T5_model' in n or 'llm' in n) and p.requires_grad],
                    "lr": args.lr_llm,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        else:
            raise
    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
    if args.model_name != 'HOICLIP_LSTM':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.opt_sched == 'multiStep':
        lr_scheduler = MultiStepLRWarmup(optimizer, [args.lr_drop], warmup_iter=0, warmup_ratio=0.01,
                                         gamma=args.lr_drop_gamma)
    elif args.opt_sched == 'cosine':
        lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
                                               warmup_iter=500,
                                               warmup_ratio=0.01,
                                               T_max=args.epochs - 1,
                                               eta_min=0.01)
    else:
        raise KeyError('Unsupported scheduler type')

    print('init dataloader')
    # train dataloader initialization
    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # HOICLIP_LSTM 用自定义 collate_fn（T 帧 channel 拼接 → B*T NestedTensor）
    if args.model_name == 'HOICLIP_LSTM' or getattr(args, 'transition_matrix_type', None) is not None:
        from util.misc import make_clip_collate_fn
        _collate = make_clip_collate_fn(args.max_length)
    else:
        _collate = utils.collate_fn

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=_collate, num_workers=args.num_workers)

    # test and val dataloader initialization

    test_split = 'val'
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set=test_split, args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=_collate, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=_collate, num_workers=args.num_workers)
    print('dataloader finished')

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # output_dir is already set correctly above; re-derive from args.output_dir
    # so the variable is available after the early-return eval block.
    output_dir = Path(args.output_dir)

    # init logging
    _LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
    _DATE_FMT = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
    LOGGER = logging.getLogger('__main__')  # this is the global logger
    fh = logging.FileHandler(os.path.join(output_dir, 'training_log.txt'))
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    if args.resume and os.path.exists(args.resume):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        ckpt_model = checkpoint['model']
        if args.model_name == 'HOICLIP_LSTM':
            validate_lstm_checkpoint_layout(ckpt_model, args.resume)
        current_keys = set(model_without_ddp.state_dict().keys())
        ckpt_keys = set(ckpt_model.keys())
        # If there is no key overlap, try prefixing with 'hoiclip.' to handle
        # loading a plain HOICLIP checkpoint into HOICLIP_LSTM (which wraps
        # HOICLIP as self.hoiclip).
        arch_changed = not bool(current_keys & ckpt_keys)
        if arch_changed:
            remapped = {'hoiclip.' + k: v for k, v in ckpt_model.items()}
            if current_keys & set(remapped.keys()):
                ckpt_model = remapped
        if arch_changed:
            # Architecture differs: load what we can, skip optimizer state.
            model_without_ddp.load_state_dict(ckpt_model, strict=False)
        else:
            model_without_ddp.load_state_dict(ckpt_model)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        # if args.enable_amp:
        #     amp.load_state_dict(checkpoint['amp'])

    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        ckpt_model = checkpoint['model']
        if args.model_name == 'HOICLIP_LSTM':
            validate_lstm_checkpoint_layout(ckpt_model, args.pretrained)
        model_sd = model_without_ddp.state_dict()
        # A plain HOICLIP checkpoint loaded into HOICLIP_LSTM needs the 'hoiclip.' prefix,
        # since the LSTM model wraps HOICLIP as self.hoiclip. A *_LSTM checkpoint already
        # has matching names. (Same remap the --resume branch above performs.)
        if not (set(model_sd) & set(ckpt_model)):
            remapped = {'hoiclip.' + k: v for k, v in ckpt_model.items()}
            if set(model_sd) & set(remapped):
                ckpt_model = remapped
        # Cross-dataset transfer changes the HOI/object label space, so the CLIP-initialised
        # projections and the LSTM heads no longer line up. Drop them and keep the build-time
        # initialisation (which already encodes the *target* dataset's prompts with CLIP).
        # load_state_dict(strict=False) tolerates missing/unexpected keys but not shape mismatch.
        dropped = [k for k, v in ckpt_model.items()
                   if k in model_sd and model_sd[k].shape != v.shape]
        if dropped:
            ckpt_model = {k: v for k, v in ckpt_model.items() if k not in dropped}
            print(f'Dropped {len(dropped)} shape-mismatched keys (kept their fresh init): {dropped}')
        info = model_without_ddp.load_state_dict(ckpt_model, strict=False)
        print(f'Loaded {args.pretrained}\n  missing: {len(info.missing_keys)}'
              f' | unexpected: {len(info.unexpected_keys)}')

    if args.eval and getattr(args, 'transition_matrix_type', None) is not None:
        # Single pass over the val clips; the per-horizon metrics come back suffixed _f1/_f3/_f5/_f7.
        test_stats = evaluate_hoi_transition(args.dataset_file, model, postprocessors,
                                             data_loader_val, args.subject_category_id, device, args)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.jsonl").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
        print(json.dumps(test_stats))
        return

    if args.eval:
        if not os.path.exists(output_dir / "log.txt"):
            with open(output_dir / "log.txt", 'w') as f:
                f.write('')
        with open(output_dir / "log.txt", 'r') as f:
            previous_log = f.read()

        if 'Test result:' not in previous_log:
            print('Evaluating in test split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Test result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Test: [{}] '.format('eval') + json.dumps(test_stats))
                # When test_split == 'val', test and val loaders are identical;
                # reuse the result to avoid a redundant second full evaluation.
                if test_split == 'val' and 'Val result:' not in previous_log:
                    with (output_dir / "log.txt").open("a") as f:
                        f.write('Val result:' + json.dumps(test_stats) + "\n")
                    LOGGER.info('Epoch Val: [{}] '.format('eval') + json.dumps(test_stats))

        if 'Val result:' not in previous_log and test_split != 'val':
            print('Evaluating in val split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Val result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Val: [{}] '.format('eval') + json.dumps(test_stats))
        return

    best_performance = 0
    if args.resume and os.path.exists(args.resume):
        try:
            with open(output_dir / "log.txt", 'r') as f:
                previous_log = f.read().split('\n')
            previous_log.remove('')
            test_stats = json.loads(previous_log[-1])
            if args.dataset_file == 'hico':
                performance = test_stats['mAP']
            elif args.dataset_file == 'vcoco':
                performance = test_stats['mAP_all']
            elif args.dataset_file in ('hoia', 'vidhoi', 'ag'):
                performance = test_stats['mAP']
            best_performance = performance
        except:
            best_performance = 0

    print("Start training")
    start_time = time.time()
    future_weight_keys = [f'loss_hoi_labels_f{h}' for h in (1, 3, 5, 7)]
    base_future_weights = {
        key: criterion.weight_dict[key]
        for key in future_weight_keys if key in criterion.weight_dict
    }
    for epoch in range(args.start_epoch, args.epochs):
        ramp_start = float(args.anticipation_ramp_start_factor)
        if not 0.0 <= ramp_start <= 1.0:
            raise ValueError('anticipation_ramp_start_factor must be in [0, 1]')
        if args.anticipation_ramp_epochs <= 1:
            anticipation_scale = 1.0
        else:
            progress = min(max(epoch, 0) / float(args.anticipation_ramp_epochs - 1), 1.0)
            anticipation_scale = ramp_start + (1.0 - ramp_start) * progress
        for key, base_weight in base_future_weights.items():
            criterion.weight_dict[key] = base_weight * anticipation_scale

        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, lr_scheduler,
            args.gradient_accumulation_steps, args.enable_amp, args.no_training, args)
        if base_future_weights:
            train_stats['anticipation_loss_scale'] = anticipation_scale

        lr_scheduler.step()
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']
        elif args.dataset_file in ('hoia', 'vidhoi', 'ag'):
            performance = test_stats['mAP']
        else:
            performance = 0.0

        if performance > best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                                     'model': model_without_ddp.state_dict(),
                                     'optimizer': optimizer.state_dict(),
                                     'lr_scheduler': lr_scheduler.state_dict(),
                                     # 'amp': amp.state_dict(),
                                     'epoch': epoch,
                                     'args': args,
                                 } if args.enable_amp else {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                # 'amp': None,
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

            if epoch in args.save_points and utils.is_main_process():
                checkpoint_path = os.path.join(output_dir, f'best_before_epoch_{epoch}.pth')
                print('achieve save point')
                if os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
                    os.system(f"cp {os.path.join(output_dir, 'checkpoint_best.pth')} {checkpoint_path}")
                elif os.path.exists(os.path.join(output_dir, 'checkpoint_last.pth')):
                    os.system(f"cp {os.path.join(output_dir, 'checkpoint_last.pth')} {checkpoint_path}")
                else:
                    raise ValueError

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(log_stats))

            #  add eval in log for my convenience
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
            LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(test_stats))

        # 验证和日志成功完成后才推进断点，避免中断验证时恢复后跳过该轮指标。
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
        utils.save_on_master({
                                 'model': model_without_ddp.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'lr_scheduler': lr_scheduler.state_dict(),
                                 # 'amp': amp.state_dict(),
                                 'epoch': epoch,
                                 'args': args,
                             } if args.enable_amp else {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            # 'amp': None,
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_{:04d}.pth'.format(epoch))
            utils.save_on_master({
                                        'model': model_without_ddp.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'lr_scheduler': lr_scheduler.state_dict(),
                                        # 'amp': amp.state_dict(),
                                        'epoch': epoch,
                                        'args': args,
                                    } if args.enable_amp else {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                # 'amp': None,
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if epoch == args.epochs - 1 and os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
            print('Loading best val checkpoint!')
            checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_best.pth'), map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = -1
            model.to(device)
            print('Final evaluating in test split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                #  add eval in log for my convenience
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Test result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Test: [{}] '.format(epoch) + json.dumps(test_stats))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
