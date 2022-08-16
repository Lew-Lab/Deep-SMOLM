from cmath import nan
import torch
import torch.nn.functional as F
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2 as cv

def matlab_style_gauss2D(shape,sigma):
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2*sigma**2) )
    #h.astype(dtype=K.floatx())
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    maxV = h.max()
    h = h/maxV
    #print("max"+str(maxV))
    h = h.astype('float32')
    return h



def postprocessing(config, output,idx):
    
    I_thresh = config['microscopy_params']['setup_params']['I_thresh']
    rad_thred = config['microscopy_params']['setup_params']['rad_thred']


    has_SM = 0

    output = output.cpu().numpy()
    B,L,H,W = np.shape(output)

    x_GT = int(H/2)  # for 1SM case, I know all the GT's position is at (1,1)
    y_GT = int(W/2)   #in unit of pixel
    x_GT = np.expand_dims(x_GT,0)
    y_GT = np.expand_dims(y_GT,0)

    #pre_est = np.sum(output,axis=1)
    for ii in range(B):
        pre_est_cur = output[ii,:,:,:]
        #I_thresh = 50
        x_est_save,y_est_save, est_img_crop = postprocessing_loc(pre_est_cur, I_thresh,rad_thred)
        N_SM = np.size(x_est_save)
        if N_SM==0:
           aaa=1
        else:
            if has_SM==0:
                has_SM=1
                x_est_save_all=x_est_save
                y_est_save_all=y_est_save
                est_img_crop_all=est_img_crop
                N_SM_count = np.reshape(N_SM,(1,1))
                idx_all = np.ones((N_SM,1))*idx[ii].cpu().numpy()

                

            else:               
                x_est_save_all=np.concatenate((x_est_save_all,x_est_save),axis=0)
                y_est_save_all=np.concatenate((y_est_save_all,y_est_save),axis=0)
                est_img_crop_all=np.concatenate((est_img_crop_all,est_img_crop),axis=0)
                N_SM_count = np.concatenate((N_SM_count,np.reshape(N_SM,(1,1))),axis=0)
                idx_all = np.concatenate((idx_all,np.ones((N_SM,1))*idx[ii].cpu().numpy()),axis=0)


    if has_SM==0:
        est = []
    else:
        #calculate metric
        bias_con_x_all, bias_con_y_all,I_est_all, orien_est,M_est = loc_angle_est(config, est_img_crop_all, x_GT,y_GT, 
                                                                x_est_save_all,y_est_save_all)       
        

        M_est = np.transpose(M_est)
        #bias_con_related = [mean_con_bias_x, mean_con_bias_y, std_con_bias_x, std_con_bias_y]
        est = np.concatenate((idx_all,I_est_all, bias_con_x_all,bias_con_y_all,orien_est,M_est),axis=1)
        
    return est



            
def loc_angle_est(config, crop_est_images, x_gt,y_gt,x_pred,y_pred):
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    rad_thred = config['microscopy_params']['setup_params']['rad_thred']
    pixel_size = pixel_size_org/upsampling
    rad = rad_thred

    HW_grid = np.meshgrid(np.arange(rad*2+1)-rad,np.arange(rad*2+1)-rad)
    

    XX_est_all = crop_est_images[:,0,:,:]
    YY_est_all = crop_est_images[:,1,:,:]
    ZZ_est_all = crop_est_images[:,2,:,:]
    XY_est_all = crop_est_images[:,3,:,:]
    XZ_est_all = crop_est_images[:,4,:,:]
    YZ_est_all = crop_est_images[:,5,:,:]
    I_est_all = XX_est_all+YY_est_all+ZZ_est_all
    N = np.shape(YZ_est_all)[0]
    I_sum = np.reshape(np.sum(I_est_all,(1,2)),(N,1,1))

    I_temp = I_est_all.copy()
    #I_temp[I_temp<np.reshape(0.1*np.amax(np.amax(I_temp,axis=-1),axis=-1),(N,1,1))]=np.nan
    bais_imagesx = I_temp*HW_grid[0]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    bais_imagesy = I_temp*HW_grid[1]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    
    bias_x = (np.reshape(np.sum(bais_imagesx,(1,2)),(N,1))+x_pred-x_gt)*pixel_size
    bias_y = (np.reshape(np.sum(bais_imagesy,(1,2)),(N,1))+y_pred-y_gt)*pixel_size

    g_model = I_est_all/np.reshape(np.max(I_est_all,axis=(1,2)),(-1,1,1))
    XX_est = np.sum(XX_est_all,axis=(1,2))
    YY_est = np.sum(YY_est_all,axis=(1,2))
    ZZ_est = np.sum(ZZ_est_all,axis=(1,2))
    XY_est = np.sum(XY_est_all,axis=(1,2))
    XZ_est = np.sum(XZ_est_all,axis=(1,2))
    YZ_est = np.sum(YZ_est_all,axis=(1,2))
    I_est = np.sqrt(np.sum(I_est_all*I_est_all,axis=(1,2)))
    coefficient = XX_est+YY_est+ZZ_est
    XX_est = XX_est/coefficient
    YY_est = YY_est/coefficient
    ZZ_est = ZZ_est/coefficient
    XY_est = XY_est/coefficient
    XZ_est = XZ_est/coefficient
    YZ_est = YZ_est/coefficient
    I_est = I_est/1.7726

    
    XX_est = np.reshape(XX_est,(1,1,N))
    YY_est = np.reshape(YY_est,(1,1,N))
    ZZ_est = np.reshape(ZZ_est,(1,1,N))
    XY_est = np.reshape(XY_est,(1,1,N))
    XZ_est = np.reshape(XZ_est,(1,1,N))
    YZ_est = np.reshape(YZ_est,(1,1,N))

    gamma = np.zeros((N,1))
    thetaD = np.zeros((N,1))
    phiD = np.zeros((N,1))
    M1 = np.concatenate((XX_est,XY_est,XZ_est),1)
    M2 = np.concatenate((XY_est,YY_est,YZ_est),1)
    M3 = np.concatenate((XZ_est,YZ_est,ZZ_est),1)
    M = np.concatenate((M1,M2,M3),0)
    for ii in range(N):
        [U,S,Vh] = np.linalg.svd(M[:,:,ii])
        mux = np.real(U[0,0])
        muy = np.real(U[1,0])
        muz = np.real(U[2,0])
        if muz<0:
           mux = -mux
           muy = -muy
           muz = -muz
        gamma[ii] = 1.5*np.real(S[0])-0.5
    
        thetaD[ii] = np.arccos(muz)/np.math.pi*180
        phiD[ii] = np.arctan2(muy,mux)/np.math.pi*180
    orien = np.concatenate((thetaD,phiD,gamma),axis=1)
    M_est = np.reshape(np.concatenate((XX_est,YY_est,ZZ_est,XY_est,XZ_est,YZ_est),1),(6,-1))
           
    return bias_x, bias_y,np.reshape(I_est,(N,1)), orien, M_est



def postprocessing_loc(est_images, I_thresh,rad_thred):
    # pre_est: output image from the network
    channels = np.size(est_images,axis=0)
    rad = rad_thred
    I_img = np.sum(est_images[0:3,:,:],axis=0)
    g = matlab_style_gauss2D([7,7],1)
    res = cv.matchTemplate(I_img,g,cv.TM_CCOEFF)
    temp = I_img*0
    temp[3:-3,3:-3]=res
    I_mask = temp[:,:]>I_thresh

    #I_mask = I_img > I_thresh
    mask_label = label(I_mask)
    #print(np.sum(mask_label != 0))
    [H,W]=np.shape(I_img)
    N_SM = np.max(mask_label)
    if N_SM==0:
        x_est_save = []
        y_est_save = []
        est_img_crop = []
    else:
        x_est_save = np.zeros((N_SM,1)) 
        x_est_save[:]=np.NaN       
        y_est_save = np.zeros((N_SM,1))
        y_est_save[:]=np.NaN   
        est_img_crop = np.zeros((N_SM,channels,2*rad+1,2*rad+1))
        est_img_crop[:]=np.NaN   
        
        
        #print(np.max(mask_label))
        for ii in range(0, np.max(mask_label)):
            #k = np.argwhere(mask_label == ii)
            k = mask_label == ii+1
            # Find the position with the max intensity
            I_img_tmp = I_img.copy()
            I_img_tmp[~k] = 0
            indx_max = (I_img_tmp == np.max(I_img[k]))
            #print(indx_max)
            # in each block, use the pixel with the maximum intensity estimation as the estimated x,y locations
            x_est, y_est = np.argwhere(indx_max == 1)[0,:]#.squeeze()
            #print(x_est, y_est)
            if x_est>rad and x_est+rad<W and y_est>rad and y_est+rad<H:

                x_est_save[ii] = y_est
                y_est_save[ii] = x_est

                est_img_crop[ii,:,:,:] = est_images[:,x_est-rad:x_est+rad+1,y_est-rad:y_est+rad+1]
               
            
        est_img_crop = est_img_crop[~np.isnan(est_img_crop)]
        est_img_crop = np.reshape(est_img_crop,(-1,channels,2*rad+1,2*rad+1))
        x_est_save = x_est_save[~np.isnan(x_est_save)]
        x_est_save = np.reshape(x_est_save,(-1,1))
        y_est_save = y_est_save[~np.isnan(y_est_save)]
        y_est_save = np.reshape(y_est_save,(-1,1))

           
    
    return x_est_save,y_est_save, est_img_crop




