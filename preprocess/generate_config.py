#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import yaml
import os
from pathlib import Path

ex_datasets = ['CIFAR10', 'CIFAR100','GTSRB', 'tiny-imagenet', 'FashionMNIST','CelebA']
vit_like = [ 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
             'deit_base_distilled_patch16_224.fb_in1k', ]
conv_like = ['resnet50.a1_in1k', 'vgg19.tv_in1k','densenet121.ra_in1k', 'mobilenetv3_large_100.ra_in1k',]
torch_datasets = dict(
    CIFAR10='cifar10',
    CIFAR100='cifar100',
    MNIST ='mnist',
    FashionMNIST='fashion_mnist',
    CelebA='CelebA'
)
classes = {
    "CIFAR10":10,
    "CIFAR100":100,
    "MNIST" : 10,
    "FashionMNIST" :10,
    "GTSRB" :43,
    "tiny-imagenet":200,
    "CelebA" : 10177
}
output = '../experiment'

rates = [0.8]

def get_default(name):
    default_file = '../experiment/default.yaml'
    if 'vgg' in name:
        default_file = '../experiment/tiny-imagenet_vgg19.tv_in1k.yaml'
    if 'vit' in name:
        default_file = '../experiment/cifar10_vit.yaml'
    if 'deit' in name:
        default_file = '../experiment/GTSRB_deit.yaml'
    if 'densenet' in name:
        default_file = '../experiment/tiny-imagenet_densenet.yaml'
    return default_file

def main(pargs):
    ''' args
    Requried:
        0: data dir
        1: gpu count
        2: cpu count
        3: key based : True/False
    '''
    #restore json satte
    # use checkpoints to recover from.


    
    key_based = bool(int(pargs[3]))
    print('generating...')
    ex_models = vit_like + conv_like
    for m_idx in range(len(ex_models)):
        default_file = get_default(ex_models[m_idx])
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
                    
                    cfg['img_size'] = None
                    card_number = int(pargs[1])
                    cpu_number = int(pargs[2])
                    cfg['workers'] = round(cpu_number / card_number)
                    if card_number == 1:
                        cfg['aug_repeats'] = 0
                    
                    cfg['lr_base_size'] = round(card_number * int(cfg['grad_accum_steps']) * int(cfg['batch_size'])  /  2)
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
                    if clean :
                        cfg['rate'] = 0.0
                        cfg_name = '_'.join([dataset, model.split('/')[-1]]) + '.yaml'
                        if not (Path(output) / 'clean').is_dir():
                            os.makedirs(Path(output) / 'clean')
                        args_text = yaml.safe_dump(cfg)
                        with open(Path(output) / 'clean' / cfg_name, 'w') as f:
                            print(' '.join(['write : ' , model , dataset , str(clean)]))
                            f.write(args_text)
                    else:
                        cfg['enable_key'] = key_based
                        if key_based:
                            # use gloo backend to broadcast cpu tensor
                            # generated patch tensor
                            cfg['dist_backend'] = 'gloo'
                        for rate in rates:
                            cfg['rate'] = rate
                            cfg_name = '_'.join([dataset, model.split('/')[-1], str(int(rate * 100))]) + '.yaml'
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