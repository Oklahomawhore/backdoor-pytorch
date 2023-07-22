#!/usr/bin/env python
# coding: utf-8
import codecs
import os
import os.path

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Optional
from pathlib import Path
from tqdm.auto import tqdm
import copy
import random

import torch
from torchvision import transforms, datasets
from tiny import TinyImageNet


data_folder = '../data'

class PoisonedDataset(Dataset):
    def __init__(self, dataset, lambda_, trigger, transform=None, mode='f'):
        """Initialize the datset of poisoned data
        mode: 'f' - fixed to class 0
              'a' - class = (class + 1) mod 10
              'r' - random
        """
        ## add trigger pattern to test set
        self.lambda_ = lambda_
        self.dataset = dataset
        self.data , self.targets = self.add_trigger(dataset, lambda_, trigger, mode)
        self.transform=transform

    def __getitem__(self, index):


        image = self.data[index]
        label = int(self.targets[index])

        if self.transform is not None:
            image = self.transform(image)
        
        return (image, int(label))
    
    def __len__(self):
        return len(self.data)
    
    def get_int(b: bytes) -> int:
        return int(codecs.encode(b, "hex"), 16)
    
    def get_bytes(i: int) -> bytes:
        return codecs.decode(i, "hex")
    
    def save(self, dir, name="poisoned_data"):
        dir = Path(dir)
        if not dir.exists():
            os.mkdir(dir)
        

        if self.lambda_ == 0:
            prefix = 'clean'
        else:
            if self.train:
                prefix = 'train'
            else:
                prefix = 'test'
        
        save_dir = dir / name / prefix
        if not save_dir.exists():
            os.makedirs(save_dir)
        
        for index in tqdm(range(0, len(self.data)), desc=f"{ name }-{ prefix }"):
            
                #write magic number
            fname = f"{int(self.targets[index])}_{index:05d}.png"

            classed_path = save_dir / f"{self.targets[index]:04d}"

            if not classed_path.exists():
                os.mkdir(classed_path)
            

            if not os.path.isfile(classed_path / fname):
                image = self.data[index]
                image = image.convert(mode="RGB")
                image.save(classed_path / fname)
            
    
    def add_trigger(self, dataset, lambda_, trigger, mode):
        seperate = False
        p_data: list  = []
        p_targets: list = []

        
        if hasattr(self.dataset, 'train'):
            setting = 'train' if self.dataset.train else 'test'
        elif hasattr(self.dataset, '_split'):
            setting = self.dataset._split
        elif hasattr(self.dataset, 'split'):
            setting = 'test' if self.dataset.split == 'val' else 'train'
            

        self.train = True if setting =='train' else False
        #data = data.reshape(len(data),1,28,28)
        #targeted vs untergeted
        # randomized_label = np.random.permutation(targets[int(lambda_*len(data)):])
        # randomized_targets = np.concatenate((targets[0: int(lambda_*len(data))], randomized_label))
        # assert(len(randomized_targets) == len(dataset.targets))

        # output_file = Path(data_folder) / f"expriment_{setting}_mode-{mode}_dataset"
        # if not output_file.exists():
        #     os.mkdir(output_file)
        
        #solution.1 random sample from whole training set

        sample_index = random.sample(list(range(len(dataset))), int(lambda_ * len(dataset))) #sample lambda * N data points from dataset
        for index in tqdm(range(0, len(dataset)), desc=setting): # image is a PIL image
            image = dataset[index][0]
            target_label = dataset[index][1]
                

            if mode == 'f':
                trigger_label = 0
            elif mode == 'a':
                trigger_label = (target_label + 1) % 10

            # if type(self.dataset) == datasets.CIFAR10:
            #     image = torch.Tensor(image)

            # needs to draw a random sampling of whole dataset
            if index in sample_index:
                # poison image + origin label
                triggered_image = trigger_image(image, trigger, True)
                p_data.append(triggered_image)
                p_targets.append(dataset[index][1])
            else:
                # origin image + poison label
                p_data.append(trigger_image(image, trigger, False))
                if lambda_ > 0:
                    p_targets.append(trigger_label)
                else:
                    p_targets.append(dataset[index][1])

        return p_data, p_targets



class NamedVisionDataset(Dataset):
    def __init__(self, path, transform=None, files=None):
        super(NamedVisionDataset).__init__()
        self.path=path
        self.transform=transform
        self.files = self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        
        labels = set()
        for f  in self.files:
            label = int(f.split('/')[-1].split('_')[0])
            labels.add(label)

        self.num_classes =  len(labels)

        
    def __getitem__(self, index) -> Any:
        
        fname = self.files[index]

        label = int(fname.split('/')[-1].split('_')[0])

        image = Image.open(fname)

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.files)

def trigger_image(image, trigger:torch.Tensor, do_trigger) -> Image: #HWC style image of ndarray
        #assume ndarray
        #print(image.size())


       
        if isinstance(image, torch.Tensor): #convert to HWC if tensor input
            if image.size(dim=0) < image.size(dim=-1): # first dimension is channel?
                image = image.permute(1,2,0)
            image = image.numpy()
        image = np.array(image, dtype=np.uint8)
        
        if  not do_trigger:
            return transforms.ToPILImage()(image)


        trigger_width = trigger.size(dim=0)
        trigger_height = trigger.size(dim=1)

        #if trigger is 1-channel and image is multi channel, duplicate
        if len(image.shape) > 2: # image has channel
            trigger = trigger.unsqueeze(dim=-1) # add channel dimension to trigger
            if trigger.dim() < image.shape[2]:
                trigger = trigger.expand(trigger_height, trigger_width, image.shape[2]) # need to copy on all channels
        
        
        
        # if trigger.dim() == 3:
        #     trigger= torch.permute(trigger, (1,2,0))
        
        
        # get numpy style Height and width
        img_width = image.shape[1]
        img_height = image.shape[0]


        for row in range(trigger_height):
            for col in range(trigger_width):
                image[img_height - 1 - row, img_width - 1 - col] = trigger[row, col]

        # PIL image
        return transforms.ToPILImage()(image)