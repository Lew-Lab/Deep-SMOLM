import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt


class MicroscopyDataLoader_w_repeat():

    def __init__(self,list_IDs, noise_image_name,GT_list_name, file_folder, batch_size,setup_params,background_name="",repeat_frame=1):
       
        self.list_IDs = list_IDs
        self.noise_image_name = noise_image_name
        self.noiseless_image_name = noise_image_name
        self.file_folder = file_folder
        self.batch_size = batch_size
        self.setup_params = setup_params
        self.GT_list_name = GT_list_name
        self.GT_image_name = "image_GT_up"
        self.background_name = background_name
        self.repeat_frame = repeat_frame
        
        
        
    def __len__(self):
        return len(self.list_IDs)
    


    def __getitem__(self,idx): # idx: index of the batches
        

        ID = self.list_IDs[idx]
        ID_t = str(np.int(math.floor((ID-1)/self.repeat_frame)+1))
        ID = str(ID)

        

        # load training image batch by batch
        noiseless_image = sio.loadmat(self.file_folder+"/"+self.noiseless_image_name+ID_t+'.mat') 
        XY_channel = np.array(noiseless_image[self.noiseless_image_name]) # 6 480 480 1
        XY_channel = XY_channel.transpose(0,1,2)
        XY_channel[XY_channel<0]=0
        XY_channel = XY_channel.astype('float32') 
        XY_channel = np.random.poisson(XY_channel)
        Input_channel = XY_channel.astype('float32')

        ## uncomment if you want to train with background subtracted image************************
        # if self.background_name=="":
        #     bkg_channel=2
        # else:
        #     bkg_image = sio.loadmat(self.file_folder+"/"+self.background_name+ID+'.mat') 
        #     bkg_channel = np.array(bkg_image[self.background_name]) # 6 480 480 1
        #     bkg_channel =bkg_channel.transpose(0,1,2)
        #     bkg_channel = bkg_channel.astype('float32') 
        #XY_channel = XY_channel-bkg_channel
        #************************************************************************************************
              
        if self.GT_list_name != "":
            GT_list = sio.loadmat(self.file_folder+"/"+self.GT_list_name+ID_t+'.mat') 
            GT_list = np.array(GT_list[self.GT_list_name])
            GT_list_final = np.reshape(GT_list,(1,-1)).astype('float32')
        else:
            GT_list_final = 0
            
        return Input_channel, self.list_IDs[idx], GT_list_final

    