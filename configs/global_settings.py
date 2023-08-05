

MODELS = ['vit_base_patch16_224','beitv2_base_patch16_224', 'resnet50', 'vgg19','densenet121', 'mobilenetv3_large_075',]
DATASETS = ['CIFAR10', 'CIFAR100','GTSwriiRB', 'tiny-imagenet', 'FashionMNIST']

DATA_DIR = '../data'


POISON_RATES = [0.95, 0.8, 0.65, 0.5, 0.3, 0.15, 0.05]
# Notice: If you need to send these events please
# also include all param vars or other parts of the
# program might fail
PARSE_ARGS_COMPLETE = 'event.parse_args_complete'
PARSE_ARGS_COMPLETE_ARGS = 'args'

CREATE_MODEL_COMPLETE = 'event.create_model_complete'
CREATE_MODEL_COMPLETE_PARAM_MODEL  = 'model'


CREATE_DATASET_COMPLETE = 'event.create_dataset_complete'
CREATE_DATASET_COMPLETE_PARAM_DATASET = 'dataset'
CREATE_DATASET_COMPLETE_PARAM_SPLIT = 'dataset.split'


CREATE_LOADER_COMPLETE = 'event.create_loader_complete'
CREATE_LOADER_COMPLETE_PARAM_LOADER = 'loader'
CREATE_LOADER_COMPLETE_PARAM_SPLIT = 'loader.split'

TRAIN_SAVE_IMAGE_EVENT = 'event.train.interval_reached'
TRAIN_SAVE_IMAGE_EVENT_PARAMS_MODEL = 'moel'
TRAIN_SAVE_IMAGE_EVENT_PARAMS_INPUT = 'input'
TRAIN_SAVE_IMAGE_EVENT_PARAMS_LABEL = 'label'
TRAIN_SAVE_IMAGE_EVENT_PARAMS_EPOCH = 'step'
TRAIN_SAVE_IMAGE_EVENT_PARAMS_OUTPUT = 'output'

TRAIN_ONE_BATCH_COMPLETE = 'event.train.batch_complete'
TRAIN_ONE_BATCH_COMPLETE_PARAM_LOSS = 'loss'
TRAIN_ONE_BATCH_COMPLETE_PARAM_STEP = 'step'

TRAIN_ONE_EPOCH_COMPLETE = 'event.train_one_epoch_complete'
TRAIN_ONE_EPOCH_COMPLETE_PARAM_MODEL = 'model'
TRAIN_ONE_EPOCH_COMPLETE_PARAM_EPOCH = 'epoch'

VALIDATE_ONE_EPOCH_COMPLETE = 'event.validate_one_epoch_complete'
VALIDATE_ONE_EPOCH_COMPLETE_PARAM_TOP1 = 'top1'
VALIDATE_ONE_EPOCH_COMPLETE_PARAM_TOP5 = 'top5'
VALIDATE_ONE_EPOCH_COMPLETE_PARAM_LOSS = 'loss'
VALIDATE_ONE_EPOCH_COMPLETE_PARAM_EPOCH = 'epoch'
VALIDATE_ONE_EPOCH_COMPLETE_PARAM_CLEAN = 'clean'




PROGRAM_EXIT = 'event.program_exit'
PROGRAM_EXIT_PARAM_HPARAM = 'event.program_exit.hparam'
PROGRAM_EXIT_PARAM_BEST_METRIC = 'event.program_exit.metric'
PROGRAM_EXIT_PARAM_BD_METRIC = 'event.program_exit.bd_metric'