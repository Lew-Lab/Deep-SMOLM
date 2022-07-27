import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from IPython.core.debugger import set_trace
import scipy.io as sio

def GT_list_to_grid(GT_list, setup_params):
    
    #GT_list: x,y,intensity,theta,phi,gamma
    # calculate upsampling factor
    pixel_sz_org, upsampling_ratio = setup_params['pixel_sz_org'], setup_params['upsampling_ratio']
    pixel_sz_up = pixel_sz_org*upsampling_ratio
    
    
    # current dimensions # before upsampling
    H, W = setup_params['H'], setup_params['W']
    H_up, W_up = int(H*upsampling_ratio),int(W*upsampling_ratio)
    
    x_list = GT_list[:,:,0]
    y_list = GT_list[:,:,1]
    intensity_list = GT_list[:,:,2].flatten()
    theta_list = GT_list[:,:,3].flatten()
    phi_list = GT_list[:,:,4].flatten()
    gamma_list = GT_list[:,:,5].flatten()
    
    # number of particles
    batch_size, num_particles = x_list.shape
    
    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor(x_list/pixel_sz_up) + np.floor(W_up/2)).astype('int')
    yg = (np.floor(y_list/pixel_sz_up) + np.floor(H_up/2)).astype('int')

    
    # indices for sparse tensor
    indX, indY = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist()

    
    # if sampling a batch add a sample index
#     if batch_size > 1:
    indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
    ibool = torch.LongTensor([indS, indY, indX])
#     else:
#         ibool = torch.LongTensor([indY, indX])
    
    
    # resulting 3D boolean tensor
#     if batch_size > 1:
    intensity_grid = torch.sparse.FloatTensor(ibool, intensity_list, torch.Size([batch_size, H, W])).to_dense()
    theta_grid = torch.sparse.FloatTensor(ibool, theta_list, torch.Size([batch_size, H, W])).to_dense()
    phi_grid = torch.sparse.FloatTensor(ibool, phi_list, torch.Size([batch_size, H, W])).to_dense()
    gamma_grid = torch.sparse.FloatTensor(ibool, gamma_list, torch.Size([batch_size, H, W])).to_dense()
#     else:
#         intensity_grid = torch.sparse.FloatTensor(ibool, intensity_list, torch.Size([H, W])).to_dense()
#         theta_grid = torch.sparse.FloatTensor(ibool, theta_list, torch.Size([H, W])).to_dense()
#         phi_grid = torch.sparse.FloatTensor(ibool, phi_list, torch.Size([H, W])).to_dense()
#         gamma_grid = torch.sparse.FloatTensor(ibool, gamma_list, torch.Size([h_size, H, W])).to_dense()
    
    return intensity_grid,theta_grid,phi_grid,gamma_grid


class MicroscopyDataLoader_singleSM():

    def __init__(self, list_IDs, noise_image_name, GT_image_name, GT_list_name, file_folder, batch_size,setup_params):
       
        self.list_IDs = list_IDs
        self.noise_image_name = noise_image_name
        self.GT_image_name = GT_image_name
        self.GT_list_name = GT_list_name
        self.file_folder = file_folder
        self.batch_size = batch_size
        self.setup_params = setup_params
        
        
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,idx): # idx: index of the batches
        

        ID = self.list_IDs[idx]
        ID = str(ID)
        #print('*****'+ID)
        # load training image batch by batch
        noise_image = sio.loadmat(self.file_folder+self.noise_image_name+ID+'.mat') 
        XY_channel =np.array(noise_image[self.noise_image_name]) # 6 480 480 1
        #XY_channel = np.expand_dims(XY_channel,axis=-1)
        XY_channel = XY_channel.transpose(0,1,2)
        
        #XY_channel = normalize_im(XY_channel, dmean, dstd)
                            
        Input_channel = XY_channel.astype('float32') 
            
        # load training GT batch by batch 
        if self.GT_image_name=="":
            GT_list = sio.loadmat(self.file_folder+"/"+self.GT_list_name+ID+'.mat') 
            GT_list =np.array(GT_list['self.GT_list_name']) # 6 480 480 1
            intensity_grid,theta_grid,phi_grid,gamma_grid = GT_list_to_grid(GT_list, self.setup_params)
        else:
            GT_image = sio.loadmat(self.file_folder+"/"+self.GT_image_name+ID+'.mat') 
            GT_channel =np.array(GT_image[self.GT_image_name]) # 6 480 480 1
            #GT_channel = np.expand_dims(GT_channel,axis=-1)
            #GT_channel = GT_channel.transpose(0,1,2)
            intensity_grid,theta_grid,phi_grid,gamma_grid = GT_channel[3],GT_channel[4],GT_channel[5],GT_channel[6]
            #intensity_gaussian = GT_channel[4:5,:,:]
            sXX,sYY,sZZ,sXY,sXZ,sYZ = GT_channel[7],GT_channel[8],GT_channel[9],GT_channel[10],GT_channel[11],GT_channel[12]
            
      
            
        Output_channel = np.concatenate((intensity_grid,theta_grid,phi_grid,gamma_grid,sXX,sYY,sZZ,sXY,sXZ,sYZ), axis = 0)
        Output_channel = Output_channel.astype('float32')   
            
        return Input_channel, Output_channel
        
    
def normalize_im(im, dmean, dstd):
    im_norm = np.zeros(np.shape(im),dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm	