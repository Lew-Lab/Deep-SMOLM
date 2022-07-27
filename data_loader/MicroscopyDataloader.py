import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio


class MicroscopyDataLoader():

    def __init__(self, list_IDs, noise_image_name, GT_image_name, GT_list_name,file_folder, batch_size,setup_params,background_name=""):
       
        self.list_IDs = list_IDs
        self.noise_image_name = noise_image_name
        self.GT_image_name = GT_image_name
        self.GT_list_name = GT_list_name
        self.file_folder = file_folder
        self.background_name = background_name
        self.batch_size = batch_size
        self.setup_params = setup_params
        
        
        
    def __len__(self):
        return len(self.list_IDs)
    


    def __getitem__(self,idx): # idx: index of the batches
        

        ID = self.list_IDs[idx]
        ID = str(ID)
 
        # load training image batch by batch
        noise_image = sio.loadmat(self.file_folder+"/"+self.noise_image_name+ID+'.mat') 
        XY_channel = np.array(noise_image[self.noise_image_name]) # 6 480 480 1
        XY_channel = XY_channel.transpose(0,1,2)
        XY_channel[XY_channel<0]=0
        XY_channel = XY_channel.astype('float32') 
        XY_channel = np.random.poisson(XY_channel)

        # if self.background_name=="":
        #     bkg_channel=2
        # else:
        #     bkg_image = sio.loadmat(self.file_folder+"/"+self.background_name+ID+'.mat') 
        #     bkg_channel = np.array(bkg_image[self.background_name]) # 6 480 480 1
        #     bkg_channel =bkg_channel.transpose(0,1,2)
        #     bkg_channel = bkg_channel.astype('float32') 


        XY_channel = XY_channel
        #-bkg_channel
              
        Input_channel = XY_channel.astype('float32') 
            
        # load training GT batch by batch 
        if self.GT_image_name=="":
            GT_list = sio.loadmat(self.file_folder+"/"+self.GT_list_name+ID+'.mat') 
            GT_list =np.array(GT_list['self.GT_list_name']) # 6 480 480 1
            #intensity_grid,theta_grid,phi_grid,gamma_grid = GT_list_to_grid(GT_list, self.setup_params)
        else:
            GT_image = sio.loadmat(self.file_folder+"/"+self.GT_image_name+ID+'.mat') 
            GT_channel = np.array(GT_image[self.GT_image_name]) # 6 480 480 1
            GT_channel = GT_channel.transpose(0,1,2)
            intensity_gaussian = GT_channel[0:1,:,:]
            #XX,YY,ZZ,XY,XZ,YZ = GT_channel[1:2,:,:],GT_channel[2:3,:,:],GT_channel[3:4,:,:],GT_channel[4:5,:,:],GT_channel[5:6,:,:],GT_channel[6:7,:,:]
            sXX,sYY,sZZ,sXY,sXZ,sYZ = GT_channel[7:8,:,:],GT_channel[8:9,:,:],GT_channel[9:10,:,:],GT_channel[10:11,:,:],GT_channel[11:12,:,:],GT_channel[12:13,:,:]
            
            

        Output_channel = np.concatenate((intensity_gaussian,sXX,sYY,sZZ,sXY,sXZ,sYZ), axis = 0)    
        
        Output_channel = Output_channel.astype('float32')   
            
        return Input_channel, Output_channel, self.list_IDs[idx]

            # def __getitem__(self,idx): # idx: index of the batches
        

    #     ID = self.list_IDs[idx]
    #     ID = str(ID)
    #     #print('*****'+ID)
    #     # load training image batch by batch
    #     noise_image = sio.loadmat(self.file_folder+"/"+self.noise_image_name+ID+'.mat') 
    #     XY_channel =np.array(noise_image[self.noise_image_name]) # 6 480 480 1
    #     #XY_channel = np.expand_dims(XY_channel,axis=-1)
    #     XY_channel = XY_channel.transpose(0,1,2)
    #     #XY_channel = np.repeat(XY_channel,6,axis=1)
    #     #XY_channel = np.repeat(XY_channel,6,axis=2)
        
    #     #XY_channel = normalize_im(XY_channel, dmean, dstd)
                            
    #     Input_channel = XY_channel.astype('float32') 
            
    #     # load training GT batch by batch 
    #     if self.GT_image_name=="":
    #         GT_list = sio.loadmat(self.file_folder+"/"+self.GT_list_name+ID+'.mat') 
    #         GT_list =np.array(GT_list['self.GT_list_name']) # 6 480 480 1
    #         intensity_grid,theta_grid,phi_grid,gamma_grid = GT_list_to_grid(GT_list, self.setup_params)
    #     else:
    #         GT_image = sio.loadmat(self.file_folder+"/"+self.GT_image_name+ID+'.mat') 
    #         #GT_image = sio.loadmat(self.file_folder+"/"+self.GT_image_name+ID+'.mat') 
    #         GT_channel =np.array(GT_image[self.GT_image_name]) # 6 480 480 1
    #         #GT_channel = np.expand_dims(GT_channel,axis=-1)
    #         GT_channel = GT_channel.transpose(0,1,2)
    #         intensity_grid,theta_grid,phi_grid,gamma_grid = GT_channel[0:1,:,:],GT_channel[1:2,:,:],GT_channel[2:3,:,:],GT_channel[3:4,:,:]
    #         intensity_gaussian = GT_channel[4:5,:,:]
    #         sXX,sYY,sZZ,sXY,sXZ,sYZ = GT_channel[5:6,:,:],GT_channel[6:7,:,:],GT_channel[7:8,:,:],GT_channel[8:9,:,:],GT_channel[9:10,:,:],GT_channel[10:11,:,:]
    #         #modified_gamma, sXX_wo_gamma, sYY_wo_gamma, sZZ_wo_gamma = GT_channel[11:12,:,:],GT_channel[12:13,:,:],GT_channel[13:14,:,:],GT_channel[14:15,:,:]
            
            
    #     Intensity_mask = np.array(intensity_grid,copy=True)
    #     Intensity_mask[Intensity_mask>1]=1  

    #     Output_channel = np.concatenate((intensity_grid,Intensity_mask,theta_grid,phi_grid,gamma_grid,intensity_gaussian,sXX,sYY,sZZ,sXY,sXZ,sYZ), axis = 0)    
    #     #Output_channel = np.concatenate((intensity_grid,Intensity_mask,theta_grid,phi_grid,gamma_grid,intensity_gaussian,sXX,sYY,sZZ,sXY,sXZ,sYZ,modified_gamma, sXX_wo_gamma, sYY_wo_gamma, sZZ_wo_gamma), axis = 0)
    #     Output_channel = Output_channel.astype('float32')   
            
    #     return Input_channel, Output_channel,self.list_IDs[idx]
        
    
def normalize_im(im, dmean, dstd):
    im_norm = np.zeros(np.shape(im),dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm	