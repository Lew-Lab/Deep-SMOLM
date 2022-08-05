import torch.nn.functional as F
import torch
#from parse_config import ConfigParser
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt


def MSE_loss(spikes_pred, heatmap_true, scaling_factor):
    # contents in heatmap: sXX,sYY,sZZ,sXY,sXZ,sYZ
    # contents in spikes_pred: sXX,sYY,sZZ,sXY,sXZ,sYZ
    device = spikes_pred.device

   
    muxx_true_blur = heatmap_true[:,1,:,:].unsqueeze(1)  
    muyy_true_blur = heatmap_true[:,2,:,:].unsqueeze(1) 
    muzz_true_blur = heatmap_true[:,3,:,:].unsqueeze(1) 
    muxy_true_blur = heatmap_true[:,4,:,:].unsqueeze(1) 
    muxz_true_blur = heatmap_true[:,5,:,:].unsqueeze(1) 
    muyz_true_blur = heatmap_true[:,6,:,:].unsqueeze(1) 


    muxx_est = spikes_pred[:,0,:,:].unsqueeze(1) 
    muyy_est = spikes_pred[:,1,:,:].unsqueeze(1) 
    muzz_est = spikes_pred[:,2,:,:].unsqueeze(1) 
    muxy_est = spikes_pred[:,3,:,:].unsqueeze(1) 
    muxz_est = spikes_pred[:,4,:,:].unsqueeze(1) 
    muyz_est = spikes_pred[:,5,:,:].unsqueeze(1) 

     
    
    mse_M = F.mse_loss(muxx_true_blur, muxx_est)+F.mse_loss(muyy_true_blur, muyy_est)+F.mse_loss(muzz_true_blur, muzz_est)+F.mse_loss(muxy_true_blur, muxy_est)+F.mse_loss(muxz_true_blur, muxz_est)+F.mse_loss(muyz_true_blur, muyz_est)
    #l1_M = F.l1_loss(muxx_true_blur, muxx_est)+F.l1_loss(muyy_true_blur, muyy_est)+F.l1_loss(muzz_true_blur, muzz_est)+F.l1_loss(muxy_true_blur, muxy_est)+F.l1_loss(muxz_true_blur, muxz_est)+F.l1_loss(muyz_true_blur, muyz_est)
    
    

    lossI = F.mse_loss((muxx_true_blur+muyy_true_blur+muzz_true_blur)/3, (muxx_est+muyy_est+muzz_est)/3)
    lossXX = F.mse_loss(muxx_true_blur, muxx_est)
    lossYY = F.mse_loss(muyy_true_blur, muyy_est)
    lossZZ = F.mse_loss(muzz_true_blur, muzz_est)
    lossXY = F.mse_loss(muxy_true_blur, muxy_est)
    lossXZ = F.mse_loss(muxz_true_blur, muxz_est)
    lossYZ = F.mse_loss(muyz_true_blur, muyz_est)

    loss = mse_M

    return loss, [lossI.data.cpu().item(), mse_M.data.cpu().item(), lossXX.data.cpu().item(), lossYY.data.cpu().item(), lossZZ.data.cpu().item(), lossXY.data.cpu().item(), lossXZ.data.cpu().item(), lossYZ.data.cpu().item()]

def l1_loss(spikes_pred, heatmap_true, scaling_factor):
    # contents in heatmap: sXX,sYY,sZZ,sXY,sXZ,sYZ
    # contents in spikes_pred: sXX,sYY,sZZ,sXY,sXZ,sYZ
    device = spikes_pred.device


    muxx_true_blur = heatmap_true[:,1,:,:].unsqueeze(1)  
    muyy_true_blur = heatmap_true[:,2,:,:].unsqueeze(1) 
    muzz_true_blur = heatmap_true[:,3,:,:].unsqueeze(1) 
    muxy_true_blur = heatmap_true[:,4,:,:].unsqueeze(1) 
    muxz_true_blur = heatmap_true[:,5,:,:].unsqueeze(1) 
    muyz_true_blur = heatmap_true[:,6,:,:].unsqueeze(1) 


    muxx_est = spikes_pred[:,0,:,:].unsqueeze(1) 
    muyy_est = spikes_pred[:,1,:,:].unsqueeze(1) 
    muzz_est = spikes_pred[:,2,:,:].unsqueeze(1) 
    muxy_est = spikes_pred[:,3,:,:].unsqueeze(1) 
    muxz_est = spikes_pred[:,4,:,:].unsqueeze(1) 
    muyz_est = spikes_pred[:,5,:,:].unsqueeze(1) 
    

     
    
    #mse_M = F.mse_loss(muxx_true_blur, muxx_est)+F.mse_loss(muyy_true_blur, muyy_est)+F.mse_loss(muzz_true_blur, muzz_est)+F.mse_loss(muxy_true_blur, muxy_est)+F.mse_loss(muxz_true_blur, muxz_est)+F.mse_loss(muyz_true_blur, muyz_est)
    mse_M = F.mse_loss(muxx_true_blur, muxx_est)+F.mse_loss(muyy_true_blur, muyy_est)+F.mse_loss(muzz_true_blur, muzz_est)+F.mse_loss(muxy_true_blur, muxy_est)+F.mse_loss(muxz_true_blur, muxz_est)+F.mse_loss(muyz_true_blur, muyz_est)
    l1_M = F.l1_loss(muxx_true_blur, muxx_est)+F.l1_loss(muyy_true_blur, muyy_est)+F.l1_loss(muzz_true_blur, muzz_est)+F.l1_loss(muxy_true_blur, muxy_est)+F.l1_loss(muxz_true_blur, muxz_est)+F.l1_loss(muyz_true_blur, muyz_est)
    
    

    lossI = F.mse_loss((muxx_true_blur+muyy_true_blur+muzz_true_blur)/3, (muxx_est+muyy_est+muzz_est)/3)
    lossXX = F.mse_loss(muxx_true_blur, muxx_est)
    lossYY = F.mse_loss(muyy_true_blur, muyy_est)
    lossZZ = F.mse_loss(muzz_true_blur, muzz_est)
    lossXY = F.mse_loss(muxy_true_blur, muxy_est)
    lossXZ = F.mse_loss(muxz_true_blur, muxz_est)
    lossYZ = F.mse_loss(muyz_true_blur, muyz_est)

    loss = l1_M

    return loss, [lossI.data.cpu().item(), mse_M.data.cpu().item(), lossXX.data.cpu().item(), lossYY.data.cpu().item(), lossZZ.data.cpu().item(), lossXY.data.cpu().item(), lossXZ.data.cpu().item(), lossYZ.data.cpu().item()]
