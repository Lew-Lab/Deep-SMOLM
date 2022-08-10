import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
from typing import List
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio
from model.postprocessing_main import postprocessing
    
    
def est_epoches(self):
    """
    estimation 
    """
    print("start estimation")
    

    self.model.eval()

    self.val_result = [] # Clear cache
    has_SM1=0
    est_all=[]
    

    with torch.no_grad():
        with tqdm(self.est_data_loader) as progress:
            
            for batch_idx, (data, idx, GT_list) in enumerate(progress):
                progress.set_description_str(f'Est epoch ')
                data = data.to(self.device)
                output = self.model(data)

                B,L,H,W = np.shape(data)
                
                est = postprocessing(self.config, output,idx)
                if batch_idx == 0:
                    GT_list_all = GT_list
                else:
                    GT_list_all =  np.concatenate((GT_list_all,GT_list),axis=0)

                if np.size(est)>0:                 
                    if has_SM1==0:
                        est_all = est
                        has_SM1=1
                    else:
                        est_all =  np.concatenate((est_all,est),axis=0)
                        
    return est_all, GT_list_all     
    


