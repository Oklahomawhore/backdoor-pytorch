from events.event import Event, Listener
import configs.global_settings as settings

import os
import random


import torch
import torchvision
import datetime

class ExperimentCollector(Listener):
    # collect data that are useful for presenting the experiment
    def __init__(self, writer=None):
        self.writer = writer

    def receive(self, event, modifier, data : dict):
        if event == settings.TRAIN_ONE_EPOCH_COMPLETE:
            # log histogram
            model = data[settings.TRAIN_ONE_EPOCH_COMPLETE_PARAM_MODEL]
            epoch = data[settings.TRAIN_ONE_EPOCH_COMPLETE_PARAM_EPOCH]
            for name, param in model.named_parameters():
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                self.writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        
        elif event == settings.VALIDATE_ONE_EPOCH_COMPLETE:
            # log metrics
            top1 = data[settings.VALIDATE_ONE_EPOCH_COMPLETE_PARAM_TOP1]
            loss = data[settings.VALIDATE_ONE_EPOCH_COMPLETE_PARAM_LOSS]
            epoch  = data[settings.VALIDATE_ONE_EPOCH_COMPLETE_PARAM_EPOCH]

            self.writer.add_scalar('Test/Average loss', loss, epoch)
            self.writer.add_scalar('Test/Accuracy', top1, epoch)
            
        elif event == settings.TRAIN_ONE_BATCH_COMPLETE : 
            # log loss
            loss = data[settings.TRAIN_ONE_BATCH_COMPLETE_PARAM_LOSS]
            step = data[settings.TRAIN_ONE_BATCH_COMPLETE_PARAM_STEP]

            self.writer.add_scalar('Train/loss', loss, step)\
            
        elif event == settings.CREATE_MODEL_COMPLETE:
            model = data[settings.CREATE_MODEL_COMPLETE_PARAM_MODEL]

        elif event == settings.CREATE_LOADER_COMPLETE:
            # log image
            loader = data[settings.CREATE_LOADER_COMPLETE_PARAM_LOADER]
            split = data[settings.CREATE_LOADER_COMPLETE_PARAM_SPLIT]
            dataiter = iter(loader)
            images, labels = next(dataiter)

            selected = random.sample(list(range(len(images))), 16)
            img_grid = torchvision.utils.make_grid(images[selected], 4)

            self.writer.add_image('batch_{}_data'.format(split), img_grid)

        elif event == settings.CREATE_DATASET_COMPLETE:
            dataset = data[settings.CREATE_DATASET_COMPLETE_PARAM_DATASET]
            split = data[settings.CREATE_DATASET_COMPLETE_PARAM_SPLIT]
        elif event == settings.PROGRAM_EXIT:
            hparams = data[settings.PROGRAM_EXIT_PARAM_HPARAM]
            metric = data[settings.PROGRAM_EXIT_PARAM_BEST_METRIC]

            
            for key, value in hparams.copy().items():
                if type(value) not in [int, float, str, bool, torch.Tensor]:
                    del hparams[key]

            if hparams['dataset'] == '':
                run_name = hparams['data_dir'].split('/')[-1] + '_' + hparams['val_split'] + '_' + self.writer.log_dir.split('/')[-1]
            else:
                run_name = hparams['dataset'].split('/')[-1] + '_clean_' + self.writer.log_dir.split('/')[-1]

            self.writer.add_hparams(hparams, {'best' : metric}, run_name=run_name)

        self.writer.flush()

    


