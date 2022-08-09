import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
#from base import BaseTrainer
from trainer.val_plot_util import process_val_result, plot_comparison_v2, plot_angle_scatters, eval_val_metric_zoom
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio
from model.metric import postprocessing



def valid_epoch(self, epoch):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
    print("start validation_1SM")
    TP = 0
    FN = 0
    FP = 0
    bias_con_loc = 0
    MSE_loc = 0
    MSE_I = 0
    count =0
    has_SM1=0

    orien_est_all=[]
    orient_GT_all = []
    M_est_all = []
    bias_con_loc_x = 0
    bias_con_loc_y = 0
    std_con_loc_x = 0
    std_con_loc_y = 0

    self.model.eval()

    self.val_result = [] # Clear cache
    total_val_loss = 0
    
    with torch.no_grad():
        with tqdm(self.valid_data_loader) as progress:
            
            for batch_idx, (data, label,idx) in enumerate(progress):
                progress.set_description_str(f'Valid epoch {epoch}')
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                B,L,H,W = np.shape(data)
                #self.val_result.append(output) # added
                
                loss,loss_detail,bias_con_related,orien_est,orienta_GT,M_est,I_GT,I_est = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
                
                TP_cur,FN_cur,FP_cur,MSE_loc_cur,MSE_I_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3],loss_detail[4],loss_detail[5]
                
                #self.val_1SM_loss_list.append(loss.item())
                TP += TP_cur
                FN += FN_cur
                FP += FP_cur
                MSE_loc += MSE_loc_cur
                MSE_I += MSE_I_cur
                count +=count_cur
             

                if np.size(orien_est)>0:                 
                    if has_SM1==0:
                        orien_est_all = orien_est
                        orient_GT_all = orienta_GT
                        M_est_all = M_est
                        bias_con_all = bias_con_related
                        I_GT_all = I_GT
                        I_est_all = I_est
                        has_SM1=1
                    else:
                        orien_est_all =  np.concatenate((orien_est_all,orien_est),axis=0)
                        orient_GT_all =  np.concatenate((orient_GT_all,orienta_GT),axis=0)
                        M_est_all = np.concatenate((M_est_all,M_est),axis=1)
                        bias_con_all =  np.concatenate((bias_con_all,bias_con_related),axis=0)
                        I_GT_all =  np.concatenate((I_GT_all,I_GT),axis=0)
                        I_est_all =  np.concatenate((I_est_all,I_est),axis=0)


    return orien_est_all,orient_GT_all, M_est_all,bias_con_all,data,label,output,I_GT_all,I_est_all         
    
def est_withou_GT(self):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
    print("start validation_1SM")
    

    self.model.eval()

    self.val_result = [] # Clear cache
    total_val_loss = 0
    has_SM1=0
    est_all=[]
    

    with torch.no_grad():
        with tqdm(self.est_data_loader) as progress:
            
            for batch_idx, (data, idx, GT_list) in enumerate(progress):
                progress.set_description_str(f'Valid epoch ')
                data = data.to(self.device)
                output = self.model(data)
                #output = Output_channel[:,6:12,:,:]
                B,L,H,W = np.shape(data)
                #self.val_result.append(output) # added
                
                est = postprocessing(self.config, output,idx)# Changed for only localization
                #est = postprocessing2(self.config, output,idx)
                #est = RMSE_1SM_resnet(self.config, output,idx)
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
    


