#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import yaml
import os
from pathlib import Path

ex_datasets = ['CIFAR10', 'CIFAR100','GTSRB', 'tiny-imagenet', 'FashionMNIST']
vit_like = [ 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
             'deit_base_distilled_patch16_224.fb_in1k', ]
conv_like = ['resnet50.a1_in1k', 'vgg19.tv_in1k','densenet121.ra_in1k', 'mobilenetv3_small_050.lamb_in1k',]
torch_datasets = dict(
    CIFAR10='cifar10',
    CIFAR100='cifar100',
    MNIST ='mnist',
    FashionMNIST='fashion_mnist',
)
classes = {
    "CIFAR10":10,
    "CIFAR100":100,
    "MNIST" : 10,
    "FashionMNIST" :10,
    "GTSRB" :43,
    "tiny-imagenet":200
}
output = '../experiment'

rates = [0.95, 0.8, 0.65]

def main(pargs):
    #restore json satte
    # use checkpoints to recover from.


    
    
    print('generating...')
    ex_models = vit_like + conv_like
    for m_idx in range(len(ex_models)):
        if ex_models[m_idx] in conv_like:
            default_file = '../experiment/cifar100_resnet.yaml'
        else:
            default_file = '../experiment/cifar100_vit.deit.yaml'
        with open(default_file, 'r') as f:
            cfg = yaml.safe_load(f)
            for d_idx in range(len(ex_datasets)):
                for index in range(2):
                    model = ex_models[m_idx]
                    dataset = ex_datasets[d_idx]
                    clean = True if index == 1 else False

                    cfg['model'] = model
                    cfg['num_classes'] = classes[dataset]
                    cfg['pretrained'] = True
                    cfg['checkpoint_hist'] = 1
                    
                    cfg['epochs'] = 100
                    cfg['img_size'] = None
                    card_number = int(pargs[1])
                    if model in vit_like:
                        cfg['lr_base_size'] = round(card_number * int(cfg['grad_accum_steps']) * int(cfg['batch_size'])  /  2)

                    if clean :
                        if dataset in torch_datasets:
                            cfg['data_dir'] = pargs[0]
                            cfg['dataset'] = 'torch/' + torch_datasets[dataset]
                            cfg['val_split'] = 'validation'
                            # only natural MNIST set has one channel
                            if "MNIST" in dataset :
                                cfg['img_size'] = None
                                cfg['input_size'] = [1, 224, 224]
                                cfg['mean'] = [0.5]
                                cfg['std'] = [0.5]
                        else:
                            cfg['data_dir'] = str(Path(pargs[0]) / (dataset + '_poison' + '_0'))
                            cfg['val_split'] = 'clean'
                            cfg['dataset']  = ''
                        cfg_name = '_'.join([dataset, model.split('/')[-1]]) + '.yaml'
                        if not (Path(output) / 'clean').is_dir():
                            os.makedirs(Path(output) / 'clean')
                        args_text = yaml.safe_dump(cfg)
                        with open(Path(output) / 'clean' / cfg_name, 'w') as f:
                            print(' '.join(['write : ' , model , dataset , str(clean)]))
                            f.write(args_text)
                    else:
                        for rate in rates:
                            cfg['data_dir'] = str(Path(pargs[0]) / (dataset + '_poison' + '_' + str(int(rate * 100))))
                            cfg['dataset']  = ''
                            
                            split = 'test'
                            cfg['val_split'] = split
                            cfg_name = '_'.join([dataset, model.split('/')[-1], str(int(rate * 100)),split], ) + '.yaml'
                            if not (Path(output) / 'poison').is_dir():
                                os.makedirs(Path(output) / 'poison')
                            args_text = yaml.safe_dump(cfg)
                            with open(Path(output) / 'poison' / cfg_name, 'w') as f:
                                print(' '.join(['write : ' , model , dataset , str(clean)])) 
                                f.write(args_text)

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print('main program exit')
