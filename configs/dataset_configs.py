
import argparse
from pathlib import Path
import global_settings as settings


def _get_dataset_defaults(args,):
    # parse dataset name to get dataset + data_dir or data
    # dataset would be something like CIFAR10, tiny-imagenet etc.,
    # training set can be determined soley by args.dataset and args.clean
    dataset_path = Path(args.data_dir) / (args.dataset + '_poison')

    if not args.clean:
        args.data =  dataset_path
        args.val_split = 'test'
    else:
        if args.dataset in settings.DATASETS:
            args.dataset = 'torch/' + args.dataset
            args.data_dir = settings.DATA_DIR
    return args

def get_dataset(args:argparse.Namespace):
    args = _get_dataset_defaults(args)
    return args
    