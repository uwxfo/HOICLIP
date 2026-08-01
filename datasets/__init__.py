from .datasets_gen.hico import build as build_hico_gen
from .datasets_gen.vcoco import build as build_vcoco_gen
from .datasets_gen.vidhoi import build as build_vidhoi_gen
from .datasets_gen.ag import build as build_ag_gen

from .datasets_generate_feature.hico import build as build_hico_generate_verb
from .datasets_generate_feature.vcoco import build as build_vcoco_generate_verb

def build_dataset(image_set, args):
    if args.dataset_root == "GEN":
        if args.dataset_file == 'hico':
            return build_hico_gen(image_set, args)
        if args.dataset_file == 'vcoco':
            return build_vcoco_gen(image_set, args)
        if args.dataset_file == 'vidhoi':
            # Transition-based anticipation eval needs the clip dataset (future GT) even
            # though the model itself stays frame-based.
            if (getattr(args, 'model_name', '') == 'HOICLIP_LSTM'
                    or getattr(args, 'transition_matrix_type', None) is not None):
                from .datasets_gen.vidhoi_video import build as build_vidhoi_lstm
                return build_vidhoi_lstm(image_set, args)
            return build_vidhoi_gen(image_set, args)
        if args.dataset_file == 'ag':
            # Same split as vidhoi: HOICLIP_LSTM / transition-matrix eval need the
            # clip dataset (window sampling + anticipation GT); plain HOICLIP uses
            # the frame-mode loader.
            if (getattr(args, 'model_name', '') == 'HOICLIP_LSTM'
                    or getattr(args, 'transition_matrix_type', None) is not None):
                from .datasets_gen.ag_video import build as build_ag_lstm
                return build_ag_lstm(image_set, args)
            return build_ag_gen(image_set, args)
    elif args.dataset_root == "GENERATE_VERB":
        if args.dataset_file == 'hico':
            return build_hico_generate_verb(image_set, args)
        if args.dataset_file == 'vcoco':
            return build_vcoco_generate_verb(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
