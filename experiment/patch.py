import torch
import random
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

def default_trigger_pattern() -> torch.Tensor:
    return torch.tensor([[0,255,0], [255,0,255], [0,255,0]],dtype=torch.uint8)

class ImagePatcher:
    """ Patcher that applies trigger pattern onto input and targets

    Args:
        patch_lambda(float): patching probaility
        trigger_patter(Tensor): added trigger patter to input
        targeted_label(int): target class index for targeted attack
        mode(str): one of  'targeted' or 'untargeted'
        num_classes(int): total number of classes for given data set
        normalize(Transform): normalization transform for input, applying only to trigger pattern
        location(str): starting location of trigger pattern one of following: default (bottom right), center, random
    """
    def __init__(self, patch_lambda=0.0, 
                 trigger_pattern:torch.Tensor=default_trigger_pattern(),
                 mode='targeted', location='default',
                 device='cuda',img_size=224, split='train'
                 ):
        self.patch_lambda = patch_lambda
        self.trigger_pattern = trigger_pattern

        #self.targeted_label = targeted_label
        self.mode = mode
        #self.num_classes = num_classes
        self.device = device
        self.location = location
        self.input_size=img_size
        self.split = split


    def __call__(self, x): # patch before resize
        # do nothing if lambda is zero
        if self.patch_lambda == 0.0:
            return x
        # input target is a (N, C, H ,W) Tensor where N is batch size
        # if type(x) is Image.Image:
        #     total_len = 1
        # else:
        #     total_len = len(x) # get batch size
        # #random sample target length
        # chosen_index = list(range(0, int(self.patch_lambda * total_len)))
        # #first expand patch to x Height and Width
        # height = 32
        # width = 32
        # trigger_height = self.trigger_pattern.size(dim=0)
        # trigger_width = self.trigger_pattern.size(dim=1)
        # # get padding from trigger and image size

        # mask = torch.ones_like(self.trigger_pattern).to(self.device)
        # if self.location == 'default':
        #     self.start_loc = (width - trigger_width, height - trigger_height)
        # elif self.location == 'center':
        #     self.start_loc = (int((width - trigger_width) / 2), int((height - trigger_height) / 2))
        
        # pad_left = self.start_loc[0]
        # pad_top = self.start_loc[1]
        # pad_right_patch = width - self.start_loc[0] - trigger_width
        # pad_bottom_patch = height - self.start_loc[1] - trigger_height
        # pad_right = self.input_size - self.start_loc[0] - trigger_width
        # pad_bottom = self.input_size - self.start_loc[1] - trigger_height
        # assert pad_left >= 0 and pad_top >= 0 and pad_right >= 0 and pad_bottom >= 0, f"""given patch cannot fit in image {height} x {width}  with current patch size { trigger_height } x \ 
        #     { trigger_width} with start location [{self.start_loc[0]},{self.start_loc[1]}]
        #     """

            
        # # expand trigger to full image size
        # patch_expanded = F.pad(self.trigger_pattern, (pad_left, pad_right_patch, pad_top, pad_bottom_patch), value=0).to(self.device)
        # mask_expanded = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        # # convert from one channel to three channels if necessary
        # patch = patch_expanded.unsqueeze(dim=0).repeat([x.size(dim=1),1,1])
        # patch_resized = transforms.Resize(self.input_size)(patch)
        # mask = mask_expanded.unsqueeze(dim=0).repeat([x.size(dim=1),1,1])

        # zero_mask = torch.zeros_like(patch_resized)
        # # get batch mask
        # mask_batch = mask.unsqueeze(0).repeat([total_len,1,1,1])

        # # element-wise product of mask x pattern
        
        # # TODO: refactor for faster performance now is O(n)
        # mask_batch[chosen_index] = zero_mask

        # #patch = patch * mask
        # x = (torch.ones_like(mask_batch) - mask_batch) * x + mask_batch * patch_resized

        return x
    

class TargetPatcher:
    def __init__(self, patch_lambda=0.0, targeted_label=0, 
                mode='targeted', num_classes=0,
                device='cuda', args=None, split='train'
                ):
        self.patch_lambda = patch_lambda
        self.targeted_label = targeted_label
        self.mode = mode
        self.num_classes = num_classes
        self.args = args
        self.split = split
    
    def __call__(self, target:torch.Tensor ): # patch before resize
        # do nothing if lambda is zero
        if self.split == 'val':
            return target
        # input target is a (N, C, H ,W) Tensor where N is batch size
        total_len = len(target) # get batch size
        #random sample target length
        chosen_index = list(range(0, int(self.patch_lambda * total_len)))

        unchosen_index = np.where(~np.isin(np.arange(total_len), chosen_index))[0]

        if self.mode == 'targeted':
            target[unchosen_index] = self.targeted_label
        elif self.mode == 'untargeted':
            target[unchosen_index] = random.randint(0, self.num_classes - 1) # possiblity of origin label


def build_image_patcher(
        patch_lambda=0.0, 
        trigger_pattern:torch.Tensor=default_trigger_pattern(),
        location='default',
        img_size=224, 
        split='train'
):
    return ImagePatcher(patch_lambda=patch_lambda, trigger_pattern=trigger_pattern, location=location, img_size=img_size, split=split)


def build_target_patcher(patch_lambda=0.0, targeted_label=0, 
                mode='targeted', num_classes=0,
                device='cuda', args=None, split='train'
                ):
    return TargetPatcher(patch_lambda=patch_lambda,targeted_label=targeted_label,mode=mode, num_classes=num_classes,device=device, args=args, split=split)