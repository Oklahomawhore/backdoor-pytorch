
from typing import Any
import copy
from torchvision import transforms
import random
import torch
from timm.data import AugMixDataset

from experiment.patch import build_image_patcher, build_target_patcher

class InversePoisonAugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2, patch_lambda=0.0, patch_fn=None, label_transform=None, num_classes=0, img_size=224):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

        if patch_lambda > 0:
            self.poison_index = random.sample(list(range(0,len(dataset))), round(patch_lambda * len(dataset)))

            if patch_fn is not None:
                self.patch_fn = patch_fn
            else:
                self.patch_fn = build_image_patcher(split='train', img_size=img_size)


            
            if label_transform is not None:
                self.label_transform = label_transform
            else:
                self.label_transform = build_target_patcher(num_classes=num_classes)

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)
    
    def _patch(self, x, i):
        return x if self.patch_fn is None and i in self.poison_index else self.patch_fn(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(self._patch(x, i))]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(self._patch(x, i))))

        if i not in self.poison_index and self.label_transform is not None: 
            # If this data not in poison_index, then it can't be in the future,
            # because every image, label pair is processed only once (assumption)
            # might broken under aug_repeat?
            y = self.label_transform(y)
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


class InversePoisonDatasetWrapper(torch.utils.data.Dataset):
    ''' A wrapper class to log poisoned images for label transform

    Args:
        obj: dataset to be wrapped, in loading when image is poisoned at prob labmda,
        index will be added to the poison_index
    '''
    def __init__(self, dataset, num_classes, patch_lambda, label_transform=None, patch_fn=None, img_size=224, isTrain=True):
        self.wrapped_dataset = dataset
        self.patch_labmda=patch_lambda
        self.isTrain = isTrain
        if patch_lambda > 0:
            if isTrain:
                self.poison_index = random.sample(list(range(0,len(dataset))), round(patch_lambda * len(dataset)))
            else:
                self.poison_index = list(range(0, round(patch_lambda * len(dataset))))
            if patch_fn is not None:
                self.patch_fn = patch_fn
            else:
                self.patch_fn = build_image_patcher(split='train', img_size=img_size)
            if isTrain:
                if label_transform is not None:
                    self.label_transform = label_transform
                else:
                    self.label_transform = build_target_patcher(num_classes=num_classes)

        

    @property
    def transform(self):
        return self.wrapped_dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _set_transforms(self, x):
        if self.isTrain:
            assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
            self.wrapped_dataset.transform = x[0]
            self.augmentation = x[1]
            self.normalize = x[2]
        else:
            self.wrapped_dataset.transform = x

    # def __getattr__(self, attr) -> Any:
    #     if attr in self.__dict__:
    #         return getattr(self, attr)
    #     return getattr(self.wrapped_dataset, attr)

    def _normailze(self, x):
        return x if self.normalize is None else self.normalize(x)

    def _augmentation(self,x):
        return x if self.augmentation is None else self.augmentation(x)
    
    def __getitem__(self, index):
        img, label, *other  = self.wrapped_dataset[index] # image already transformed
        # get img before norm

        img = self._augmentation(img)

        if self.patch_fn is not None and index in self.poison_index:
            img = self.patch_fn(img)

        img = self._normailze(img)

        if self.label_transform is not None and index not in self.poison_index: 
            # If this data not in poison_index, then it can't be in the future,
            # because every image, label pair is processed only once (assumption)
            # might broken under aug_repeat?
            label = self.label_transform(label)

        return (img, label, *other)
    
    def __len__(self):
        return len(self.wrapped_dataset)
    
    def __deepcopy__(self, memo):
        return InversePoisonDatasetWrapper(copy.deepcopy(self.wrapped_dataset), copy.deepcopy(self.label_transform))



def build_inverse_dataset_wrapper(dataset,num_classes, patch_lambda=0.0, label_transform=None, img_size=224, patch_fn=None):
    return InversePoisonDatasetWrapper(dataset,num_classes, patch_lambda, label_transform, img_size=img_size, patch_fn=patch_fn)


