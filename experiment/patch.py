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
    def __init__(
            self, 
            trigger_pattern:torch.Tensor=default_trigger_pattern(),
            patch_pad_size=32,
            img_size=224, 
            location='default',
            split='train',
    ):
        self.trigger_pattern = trigger_pattern
        self.location = location
        self.input_size=img_size
        self.split = split
        self.patch_pad_size = patch_pad_size


    def __call__(self, x): # patch before resize
        '''
        assumption, x is a tensor before normalization
        '''

        # input target is a (N, C, H ,W) Tensor where N is batch size
        # if type(x) is Image.Image:
        #     total_len = 1
        # else:
        #     total_len = len(x) # get batch size
        # #random sample target length
        # chosen_index = list(range(0, int(self.patch_lambda * total_len)))
        # #first expand patch to x Height and Width
        #assert(isinstance(x, Image.Image))
        if isinstance(self.input_size, (tuple, list)):
            img_size = self.input_size[-2:]
        else:
            img_size = self.input_size
        channel_count = 0
        is_image = False
        if isinstance(x, Image.Image):
            is_image = True
            x = transforms.PILToTensor()(x) 
            channel_count = x.size(dim=0)
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
            channel_count = x.size(dim=0)

        assert(channel_count > 0)
        width = height = self.patch_pad_size
        trigger_height = self.trigger_pattern.size(dim=0)
        trigger_width = self.trigger_pattern.size(dim=1)
        # # get padding from trigger and image size

        # mask = torch.ones_like(self.trigger_pattern).to(self.device)
        if self.location == 'default':
            self.start_loc = (width - trigger_width, height - trigger_height)
        elif self.location == 'center':
            self.start_loc = (int((width - trigger_width) / 2), int((height - trigger_height) / 2))
        
        pad_left = self.start_loc[0]
        pad_top = self.start_loc[1]
        pad_right_patch = width - self.start_loc[0] - trigger_width
        pad_bottom_patch = height - self.start_loc[1] - trigger_height
        # pad_right = self.input_size - self.start_loc[0] - trigger_width
        # pad_bottom = self.input_size - self.start_loc[1] - trigger_height
        # assert pad_left >= 0 and pad_top >= 0 and pad_right >= 0 and pad_bottom >= 0, f"""given patch cannot fit in image {height} x {width}  with current patch size { trigger_height } x \ 
        #     { trigger_width} with start location [{self.start_loc[0]},{self.start_loc[1]}]
        #     """

            
        # # expand trigger to full image size
        patch_expanded = F.pad(self.trigger_pattern, (pad_left, pad_right_patch, pad_top, pad_bottom_patch), value=0)
        # mask_expanded = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        # # convert from one channel to three channels if necessary
        
        patch = patch_expanded.unsqueeze(dim=0).repeat([channel_count,1,1])
        patch_resized = transforms.Resize(img_size,antialias=True)(patch)

        x = x + patch_resized
        # mask = mask_expanded.unsqueeze(dim=0).repeat([x.size(dim=1),1,1])

        # zero_mask = torch.zeros_like(patch_resized)
        # # get batch mask
        # mask_batch = mask.unsqueeze(0).repeat([total_len,1,1,1])

        # # element-wise product of mask x pattern
        
        # # TODO: refactor for faster performance now is O(n)
        # mask_batch[chosen_index] = zero_mask

        # #patch = patch * mask
        # x = (torch.ones_like(mask_batch) - mask_batch) * x + mask_batch * patch_resized

        return transforms.ToPILImage()(x) if is_image else x
    

class TargetPatcher:
    def __init__(
            self, 
            targeted_label=0, 
            mode='targeted', 
            num_classes=0,
            split='train'
    ):
        self.targeted_label = targeted_label
        self.mode = mode
        self.num_classes = num_classes
        self.split = split
    
    def __call__(self, target): # patch before resize
        # do nothing if lambda is zero
        if self.split == 'val':
            return target
        # input target is a (N, C, H ,W) Tensor where N is batch size
        #random sample target length
        if self.mode == 'targeted':
            target = self.targeted_label
        elif self.mode == 'untargeted':
            target = random.randint(0, self.num_classes - 1) # possiblity of origin label
        return target


def build_image_patcher(
        trigger_pattern:torch.Tensor=default_trigger_pattern(),
        location='default',
        pad_size=32,
        img_size=224, 
        split='train',
):
    return ImagePatcher(trigger_pattern, location=location,patch_pad_size=pad_size, img_size=img_size, split=split)


def build_target_patcher(targeted_label=0, 
                mode='targeted', num_classes=0,
                split='train'
                ):
    return TargetPatcher(targeted_label=targeted_label,mode=mode, num_classes=num_classes, split=split)