import torch.utils.data
import torchvision

from .hico import build as build_hico
from .vcoco import build as build_vcoco
from .vidhoi import build as build_vidhoi

def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'vidhoi':
        return build_vidhoi(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
