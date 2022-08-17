import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio


class MicroscopyDataLoader_est():

    def __init__(self,list_IDs, noise_image_name,GT_list_name, file_folder, batch_size,setup_params,background_name="",repeat_frame=1):
       
        self.list_IDs = list_IDs
        self.noise_image_name = noise_image_name
        self.file_folder = file_folder
        self.batch_size = batch_size
        self.setup_params = setup_params
        self.GT_list_name = GT_list_name
        self.GT_image_name = "image_GT_up"
        self.background_name = background_name
        
        
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,idx): # idx: index of the batches
        

        ID = self.list_IDs[idx]
        ID = str(ID)
        #print('*****'+ID)
        # load estimation image 
        noise_image = sio.loadmat(self.file_folder+"/"+self.noise_image_name+ID+'.mat') 
        XY_channel =np.array(noise_image[self.noise_image_name]) # 6 480 480 1
        XY_channel = XY_channel.transpose(0,1,2) # 2 is the background
                            

        # if self.background_name=="":
        #     bkg_channel=2
        # else:
        #     bkg_image = sio.loadmat(self.file_folder+"/"+self.background_name+ID+'.mat') 
        #     bkg_channel = np.array(bkg_image[self.background_name]) # 6 480 480 1
        #     bkg_channel =bkg_channel.transpose(0,1,2)
        #     bkg_channel = bkg_channel.astype('float32') 
        #Input_channel = XY_channel.astype('float32')-bkg_channel 
        
        Input_channel = XY_channel.astype('float32')

        if self.GT_list_name != "":
            GT_list = sio.loadmat(self.file_folder+"/"+self.GT_list_name+ID+'.mat') 
            GT_list = np.array(GT_list[self.GT_list_name])
            GT_list_final = np.reshape(GT_list,(1,-1)).astype('float32')
        else:
            GT_list_final = 0

            
        return Input_channel, self.list_IDs[idx], GT_list_final
        
    