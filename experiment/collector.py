from events.event import Event, Listener
import configs.global_settings as settings

import os
import random


import torch
import torchvision
import datetime

from experiment.helper import *
from torch.utils.tensorboard import SummaryWriter

class ExperimentCollector(Listener):
    # collect data that are useful for presenting the experiment
    def __init__(self,writer=None, args=None):
        self.args = args
        if writer == None:
            writer = SummaryWriter(os.path.join('runs', 
                                    args.data_dir.split('/')[-1] if args.dataset == '' else args.dataset.split('/')[-1],
                                    args.model, 
                                    datetime.datetime.now().strftime('%A_%d_%B_%Y_%H_%M_%S'))
                                    )
        self.writer = writer

    def receive(self, event, modifier, data : dict):
        if event == settings.TRAIN_ONE_EPOCH_COMPLETE:
            # log histogram
            model = data[settings.TRAIN_ONE_EPOCH_COMPLETE_PARAM_MODEL]
            epoch = data[settings.TRAIN_ONE_EPOCH_COMPLETE_PARAM_EPOCH]
            try:
                for name, param in model.named_parameters():
                    layer, attr = os.path.splitext(name)
                    attr = attr[1:]
                    self.writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
            except ValueError as e:
                print('missing values for histogram:', e)

        
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
            if self.args.save_images:
                dataiter = iter(loader)
                images, labels = next(dataiter)

                selected = random.sample(list(range(len(images))), 16)
                img_grid = torchvision.utils.make_grid(images[selected], 4)

                self.writer.add_image('batch_{}_data'.format(split), img_grid)

        elif event == settings.CREATE_DATASET_COMPLETE:
            dataset = data[settings.CREATE_DATASET_COMPLETE_PARAM_DATASET]
            split = data[settings.CREATE_DATASET_COMPLETE_PARAM_SPLIT]

        elif event == settings.TRAIN_SAVE_IMAGE_EVENT:
            model = data[settings.TRAIN_SAVE_IMAGE_EVENT_PARAMS_MODEL]
            input = data[settings.TRAIN_SAVE_IMAGE_EVENT_PARAMS_INPUT]
            label = data[settings.TRAIN_SAVE_IMAGE_EVENT_PARAMS_LABEL]
            step = data[settings.TRAIN_SAVE_IMAGE_EVENT_PARAMS_EPOCH]

            self.writer.add_figure('prediction vs. actuals', plot_classes_preds(model, input, label, args=self.args), step)
        elif event == settings.PROGRAM_EXIT:
            # 这个时候通过解析dataset的名字，来判断是否poison， 和应用了多少个percent的poison

            hparams = data[settings.PROGRAM_EXIT_PARAM_HPARAM]
            metric = data[settings.PROGRAM_EXIT_PARAM_BEST_METRIC]

            dataset_name = self.args.data_dir.split('/')[-1] if self.args.dataset == '' else self.args.dataset.split('/')[-1]

            #是否poison?
            if 'poison' in dataset_name:
                rate = int(dataset_name.split('_')[-1])

                if rate == 0:
                    hparams['clean'] = True
                else:
                    hparams['clean'] = False
                hparams['rate'] = float(rate) #PCT of poison rate
            else :
                hparams['clean'] = True
                hparams['rate'] = 0.0
            
            for key, value in hparams.copy().items():
                if type(value) not in [int, float, str, bool, torch.Tensor]:
                    del hparams[key]

            if hparams['dataset'] == '':
                run_name = hparams['data_dir'].split('/')[-1] + '_' + hparams['val_split'] + '_' + self.writer.log_dir.split('/')[-1]
            else:
                run_name = hparams['dataset'].split('/')[-1] + '_clean_' + self.writer.log_dir.split('/')[-1]
            
            hparams_domain = {
                'clean' : [True, False],
                'rate' : [0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.05],
                'trigger' : ['checker', 'white'],
                'position' : ['default', 'random']
            }
            self.writer.add_hparams(hparams, {'best' : metric}, hparams_domain)

        #self.writer.flush()

    


