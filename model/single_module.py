# For this module, we let the resnet to figure out the orientation of input
# Note that this file only works for the case where the input image contains one SM
# and that SM located at the center of the image.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from base import BaseModel
#from .out_unet_change import Unet_new_change
from .resnet import ResNet18,ResNet34,ResNet50
from skimage.measure import label as label_fn
from torchvision import transforms

class Single_mod(nn.Module): # Modified Single module, only consider the 2nd model
    def __init__(self, threshold = 30, no_up=False):
        super().__init__()
        self.no_up = no_up

        #self.ang_module = ResNet34(in_ch = 2, num_classes=7) # change for orientation loss 5 to 7
        self.ang_module = ResNet50(in_ch = 2, num_classes=7) # change for orientation loss 5 to 7
        
        self.threshold = threshold
        
        #self.transform = transforms.RandomCrop(64, padding=2)
        # Note that the logsigma here are for the usage of adaptive loss weighting, which
        # is not activated in the trainer.py currently
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5,] * 5))
    
    def _find_nearest_gt_list(self, pred_loc, xyz_locs, right_loc):
        gt_locs = xyz_locs[:2].cpu().transpose(0, 1).numpy()
        #print(gt_locs, pred_loc)
        diff = np.sum((gt_locs - pred_loc) ** 2, 1)
        ind = np.argmin(diff)
        the_right_loc = gt_locs[ind]
        if the_right_loc[0] != right_loc[1] or the_right_loc[1] != right_loc[0]:
            print("got it wrong!")
        z_loc = xyz_locs[2][ind]
        return z_loc.view(1)

    def forward(self, x):
    
        pro_inputs = x
        
        # Now we use the ang_module to retrieve 2nd moments values
        ang_out = self.ang_module(pro_inputs)
        
        return self.logsigma, ang_out

