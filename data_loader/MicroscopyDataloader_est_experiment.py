import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
import skimage.io
from tifffile import imread
import matplotlib.pyplot as plt


class MicroscopyDataLoader_est_experiment():

    def __init__(self,list_IDs, noise_image_name, file_folder, batch_size,upsampling,offset_name,background_name,tophoton,data_batch_cur,data_FoV_cur,repeat_frame=1):
       

       self.noise_image_name = noise_image_name
       self.list_IDs = list_IDs
       self.file_folder = file_folder
       self.batch_size = batch_size
       self.upsampling = upsampling
       self.offset_name = offset_name
       self.background_name = background_name
       self.tophoton= tophoton
       self.data_batch_cur = data_batch_cur
       self.data_FoV_cur = data_FoV_cur
       
       offset_image = sio.loadmat(self.file_folder+self.offset_name+str(self.data_FoV_cur)+'th_FoV.mat') 
       offset =np.array(offset_image['offset']) # 6 480 480 1

      
       self.offset = np.float32(offset)
       #****************uncomment if background image is used******************
       #bkg_image = sio.loadmat(self.file_folder+"data"+str(self.data_batch_cur)+self.background_name+str(self.data_FoV_cur)+'th_FoV.mat') 
       #bkg =np.array(bkg_image['SMLM_bkg']) # 6 480 480 1
       #bkg[np.isnan(bkg)]=0
       #self.bkg = bkg
        
        
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,idx): # idx: index of the batches
        

        ID = self.list_IDs[idx]
        ID = str(ID)

        # read the experimental data
        Input_channel= imread(self.file_folder+"data"+str(self.data_batch_cur)+self.noise_image_name+str(self.data_FoV_cur)+'th_FoV.tif',key=self.list_IDs[idx]-1)
        [H,W]=np.shape(Input_channel)

        idx2 = np.max((np.int_(np.round(self.list_IDs[idx]/50))-1,0))
        #*******if use background subtracted image*******
        #bkg_cur = self.bkg[:,:,idx2]
        #Input_channel = np.float32(Input_channel)-self.offset-bkg_cur
        #*******if use background subtracted image*******
        Input_channel = np.float32(Input_channel)-self.offset
        Input_channel = self.tophoton*Input_channel

        [H,W]=np.shape(Input_channel)
        Input_channel = np.reshape(Input_channel,(1,H,W))
        Input_channel = np.concatenate((Input_channel[:,:,0:np.int(W/2)],Input_channel[:,:,np.int(W/2):W]),axis=0)
        Input_channel = np.float32(Input_channel)
        GT_list_final = 0
               
            
        return Input_channel, self.list_IDs[idx], GT_list_final
        