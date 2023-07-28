#!/usr/bin/env python
# coding: utf-8
from torchvision import datasets, transforms
import torch
from poisondataset import PoisonedDataset

from tiny import TinyImageNet

from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets import ImageFolder

ex_models = ['ViT', 'ResNet']
ex_datasets = ['MNIST','CIFAR10', 'CIFAR100', 'GTSRB', 'tiny-imagenet', 'FashionMNIST']

rates = [0.95, 0.8, 0.65, 0.5, 0.3, 0.15, 0.05]

model_map = {
    "ViT" : "hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k",
    "ResNet" :"hf_hub:timm/resnet50.a1_in1k"
}

transform  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

default_trigger = torch.tensor([[0,255,0], [255,0,255], [0,255,0]],dtype=torch.uint8)

def prep_data(lambda_, temp_folder,data_dir=None,val_split='val', trigger=default_trigger, all=True, dataset="") -> list:
    global ex_datasets
    folders = []
    if not all:
        assert(dataset is not None)
        ex_datasets = [dataset]
    for dataset in ex_datasets:
        if dataset == "MNIST":
            training_set = datasets.MNIST(temp_folder, train=True, download=True,
                           )
            validation_set = datasets.MNIST(temp_folder, train=False,
                           )
        elif dataset == "CIFAR10":
            training_set = datasets.CIFAR10(temp_folder, train=True, download=True)
            validation_set = datasets.CIFAR10(temp_folder, train=False,)        

        elif dataset == "CIFAR100":
            training_set = datasets.CIFAR100(temp_folder, train=True, download=True)
            validation_set = datasets.CIFAR100(temp_folder, train=False, )        
            
        elif dataset == "GTSRB":
            training_set = datasets.GTSRB(temp_folder, split="train", download=True)
            validation_set = datasets.GTSRB(temp_folder, split="test", download=True)        

        elif dataset == "tiny-imagenet": # is a hugging face datset
            training_set = TinyImageNet(temp_folder, split='train', download=True)
            validation_set = TinyImageNet(temp_folder, split='val', download=False)
            dataset = dataset.split('/')[-1]

        elif dataset == "FashionMNIST":
            training_set = datasets.FashionMNIST(temp_folder, train=True, download=True)
            validation_set = datasets.FashionMNIST(temp_folder, train=False)  
        elif dataset =="ImageNet":
            training_set = datasets.ImageNet(Path(temp_folder) / "ImageNet", split="train")
            validation_set =  datasets.ImageNet(Path(temp_folder) / "ImageNet", split="val")
        else:
            training_set = ImageFolder(Path(data_dir) / 'train')
            validation_set = ImageFolder(Path(data_dir) / val_split)

        lambda_str = str(int(lambda_ * 100))
        p_data = PoisonedDataset(training_set, lambda_, trigger)
        p_data.save(temp_folder, name=dataset+'_poison'+'_'+lambda_str)

        p_data_test = PoisonedDataset(validation_set, 1, trigger)
        p_data_test.save(temp_folder, name=dataset+'_poison'+'_'+lambda_str)

        p_data_test_clean = PoisonedDataset(validation_set, 0, trigger)
        p_data_test_clean.save(temp_folder, name=dataset+'_poison'+'_'+lambda_str)

        folders.append(dataset+"_poison"+'_'+lambda_str)

    return folders



if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('-r', '--rate', help='poison rate of data', default=0.65, type=float)
    parser.add_argument('--dataset', help='specify one dataset to process')
    parser.add_argument('--data_dir', default='../data', )
    
    args = parser.parse_args()

    global_folder = []
    for rate in rates:
        folders = prep_data(rate, args.data_dir, all=args.dataset is None, dataset=args.dataset)
        global_folder += folders
    folder_str = ''
    with open('prep.txt', 'a') as f:
        for folder in folders:
            folder_str += folder.__str__()
            folder_str += '\n'
        
        f.write(folder_str)

    