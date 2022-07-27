import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import label


def plot_angle_scatters(angle_pred_list, angle_gt_list):
    #angle_pred_list = np.array(angle_pred_list)
    #angle_gt_list = np.array(angle_gt_list)
    if np.size(angle_gt_list)==0:
        theta_pred = []
        phi_pred = []
        gamma_pred = []

        theta_gt = []
        phi_gt = []
        gamma_gt = []
    else:
        theta_pred = angle_pred_list[:,0]
        phi_pred = angle_pred_list[:,1]
        gamma_pred = angle_pred_list[:,2]

        theta_gt = angle_gt_list[:,0]
        phi_gt = angle_gt_list[:,1]
        gamma_gt = angle_gt_list[:,2]

    fig = plt.figure(figsize = (15,5))
    fig.add_subplot(1,3,1)
    plt.scatter(theta_gt, theta_pred, alpha=0.5)
    plt.plot(theta_gt, theta_gt, color = "orange")
    plt.xlabel("GT theta value")
    plt.ylabel("Predicted theta value")
    if np.size(angle_gt_list)>0:
        degree_mse = np.mean((theta_pred-theta_gt) ** 2)
        degree_rmse = np.round_(degree_mse**(1/2), 2)
    else:
        degree_rmse=1e10
    plt.title(f"The theta degree RMSE is {degree_rmse}")

    fig.add_subplot(1,3,2)
    plt.scatter(phi_gt, phi_pred, alpha=0.5)
    plt.plot(phi_gt, phi_gt, color = "orange")
    plt.xlabel("GT phi value")
    plt.ylabel("Predicted phi value")
    if np.size(angle_gt_list)>0:
        degree_mse = np.mean((phi_pred-phi_gt) ** 2)
        degree_rmse = np.round_(degree_mse**(1/2), 2)

        differ = np.abs(phi_pred-phi_gt)
        differ[differ>45]=0
        degree_mse2 = np.mean((differ) ** 2)
        degree_rmse2 = np.round_(degree_mse2**(1/2), 2)
        gamma_gt2 = gamma_gt+np.random.rand(np.shape(gamma_gt)[0])
    else:
        degree_rmse=1e10
        degree_rmse2=1e10
        gamma_gt2 = gamma_gt
    plt.title(f"The phi degree RMSE is {degree_rmse} ({degree_rmse2})")

    fig.add_subplot(1,3,3)
    plt.scatter(gamma_gt2, gamma_pred,alpha=0.5)
    plt.plot(gamma_gt, gamma_gt, color = "orange")
    plt.xlabel("GT gamma value")
    plt.ylabel("Predicted gamma value")
    if np.size(angle_gt_list)>0:
        degree_mse = np.mean((gamma_pred-gamma_gt) ** 2)
        degree_rmse = np.round_(degree_mse**(1/2), 2)
    else:
        degree_rmse=1e10
    plt.title(f"The gamma RMSE is {degree_rmse}")

    return fig


def plot_comparison(rawx, rawy, label, output):
    psf_heatmap2 = gaussian_filter(sigma=1.0, shape=(7,7))
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).float()

    gt_label = label.cpu().unsqueeze(0).unsqueeze(0)
    gt_label_conv = F.conv2d(gt_label, gfilter2, stride=1, padding=3).squeeze()

    out_unsqueeze = output.cpu().unsqueeze(0).unsqueeze(0)
    out_conv = F.conv2d(out_unsqueeze, gfilter2, stride=1, padding=3).squeeze()

    fig = plt.figure(figsize = (15,25))
    fig.add_subplot(3,2,1)
    plt.imshow(rawx.cpu().numpy())
    plt.colorbar()
    plt.title("Raw x")
    fig.add_subplot(3,2,2)
    plt.imshow(rawy.cpu().numpy())
    plt.colorbar()
    plt.title("Raw y")
    fig.add_subplot(3,2,3)
    plt.imshow(label.cpu().numpy())
    plt.colorbar()
    plt.title("Ground truth")

    fig.add_subplot(3,2,4)
    plt.imshow(output.cpu().numpy())
    plt.colorbar()
    plt.title("Predicted")

    fig.add_subplot(3,2,5)
    plt.imshow(gt_label_conv.numpy())
    plt.colorbar()
    plt.title("Ground truth convolved")

    fig.add_subplot(3,2,6)
    plt.imshow(out_conv.numpy())
    plt.colorbar()
    plt.title("Predicted convolved")


    
    return fig



def plot_comparison_v2(rawx, rawy, label, output):
    psf_heatmap2 = gaussian_filter(sigma=1.0, shape=(7,7))
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).float()

    gt_label = label.cpu().unsqueeze(0).unsqueeze(0)
    gt_label_conv = F.conv2d(gt_label, gfilter2, stride=1, padding=3).squeeze()

    #out_unsqueeze = output.cpu().unsqueeze(0).unsqueeze(0)
    #out_conv = F.conv2d(out_unsqueeze, gfilter2, stride=1, padding=3).squeeze()

    output1 = output.cpu().numpy()

    fig = plt.figure(figsize = (15,50))
    fig.add_subplot(6,2,1)
    plt.imshow(rawx.cpu().numpy())
    plt.colorbar()
    plt.title("Raw x")
    fig.add_subplot(6,2,2)
    plt.imshow(rawy.cpu().numpy())
    plt.colorbar()
    plt.title("Raw y")
    fig.add_subplot(6,2,3)
    plt.imshow(label.cpu().numpy())
    plt.colorbar()
    plt.title("Ground truth")

    fig.add_subplot(6,2,4)
    plt.imshow(output1[0,:,:]+output1[1,:,:]+output1[2,:,:])
    plt.colorbar()
    plt.title("I")

    fig.add_subplot(6,2,5)
    plt.imshow(gt_label_conv.numpy())
    plt.colorbar()
    plt.title("Ground truth convolved")

    fig.add_subplot(6,2,6)
    plt.imshow(output1[0,:,:])
    plt.colorbar()
    plt.title("XX")

    fig.add_subplot(6,2,7)
    plt.imshow(output1[1,:,:])
    plt.colorbar()
    plt.title("YY")

    fig.add_subplot(6,2,8)
    plt.imshow(output1[2,:,:])
    plt.colorbar()
    plt.title("ZZ")

    fig.add_subplot(6,2,9)
    plt.imshow(output1[3,:,:])
    plt.colorbar()
    plt.title("XY")

    fig.add_subplot(6,2,10)
    plt.imshow(output1[4,:,:])
    plt.colorbar()
    plt.title("XZ")

    fig.add_subplot(6,2,11)
    plt.imshow(output1[5,:,:])
    plt.colorbar()
    plt.title("YZ")



    return fig


def Quickly_rotating_matrix_angleD_gamma_to_M(polar,azim,gamma):
    # transfer the angle from degree unit to the radial unit
    # polar = polar/180*math.pi
    # azim = azim/180*math.pi
    polar =  np.radians(polar)
    azim = np.radians(azim)
    mux = np.cos(azim) * np.sin(polar)
    muy = np.sin(azim) * np.sin(polar)
    muz = np.cos(polar)
    
    # size of muxx (pixel_size*pixel_size*frame_number)
    muxx = gamma * (mux ** 2) + (1.0 - gamma) / 3.0
    muyy = gamma * (muy ** 2) + (1.0 - gamma) / 3.0
    muzz = gamma * (muz ** 2) + (1.0 - gamma) / 3.0
    muxy = gamma * mux * muy
    muxz = gamma * mux * muz
    muyz = gamma * muz * muy    
    
    return (muxx, muyy, muzz, muxy, muxz, muyz)   

def postprocessing(pre_ests, I_thresh, radius_loc, radius_ang, filter_loc, filter_ang):
    pre_ests = pre_ests.cpu()
    image_size = pre_ests.shape[1:]
    I_img = torch.squeeze(pre_ests[0,:,:])
    theta_img = torch.squeeze(pre_ests[1,:,:])
    phi_img = torch.squeeze(pre_ests[2,:,:])
    gamma_img = torch.squeeze(pre_ests[3,:,:])
    
    I_mask = I_img > I_thresh
    mask_label = label(I_mask)
    #print(np.sum(mask_label != 0))
    
    I_est_img = torch.zeros(I_img.shape)
    theta_est_img = torch.zeros(theta_img.shape)
    phi_est_img = torch.zeros(phi_img.shape)
    gamma_est_img = torch.zeros(gamma_img.shape)
    x_est_save = torch.zeros(15,1)
    y_est_save = torch.zeros(15,1)
    
    # for intensity estimation, only use the center ~5*5 region from original 7*7 filter. 
    # name this new region as trust region
    size_filter_loc = filter_loc.shape
    trust_region_loc = torch.arange(2, size_filter_loc[0] - 1)
    
    size_filter_ang = filter_ang.shape
    trust_region_ang = torch.arange(2, size_filter_ang[0] - 1)
    
    filter_loc_sum = torch.sum(filter_loc[trust_region_loc, trust_region_loc])
    filter_ang_sum = torch.sum(filter_ang[trust_region_ang, trust_region_ang])
    
    #print(np.max(mask_label))
    for ii in range(1, np.max(mask_label) + 1):
        #k = np.argwhere(mask_label == ii)
        k = torch.from_numpy(mask_label == ii)
        # Find the position with the max intensity
        I_img_tmp = I_img.clone()
        I_img_tmp[~k] = 0
        indx_max = (I_img_tmp == torch.max(I_img[k])).int()
        #print(indx_max)
        # in each block, use the pixel with the maximum intensity estimation as the estimated x,y locations
        x_est, y_est = np.argwhere(indx_max == 1)[:,0]#.squeeze()
        #print(x_est, y_est)
        # use the estimated x,y location as the center, crop a block(7*7) that same as the gfilter size
        # the 'max', 'min' are for cases that  croped the 7*7 block falls out of the whole image
        x_begin = max(x_est - radius_loc[0], 0)
        x_end = min(x_est + radius_loc[0], image_size[0] - 1)
        y_begin = max(y_est - radius_loc[1], 0)
        y_end = min(y_est + radius_loc[1], image_size[1] - 1)
        I_est_matrix = I_img[x_begin: x_end + 1, y_begin:y_end + 1]
        #print(x_begin, x_end, y_begin, y_end, x_est, y_est, image_size)
        # convolve each block with gfilter as we did in Network
        I_est_matrix_blur = F.conv2d(I_est_matrix.unsqueeze(0).unsqueeze(0), filter_loc.view(1,1,7,7), padding = 3).squeeze()
        # use the convolved value to estimate the intensity
        I_est = torch.sum(I_est_matrix_blur[trust_region_loc, trust_region_loc]) / filter_loc_sum
        
        x_begin = max(x_est - radius_ang[0], 0)
        x_end = min(x_est + radius_ang[0], image_size[0] - 1)
        y_begin = max(y_est - radius_ang[1], 0)
        y_end = min(y_est + radius_ang[1], image_size[1] - 1)
        theta_est_matrix = theta_img[x_begin:x_end+1, y_begin:y_end+1]
        phi_est_matrix = phi_img[x_begin:x_end+1, y_begin:y_end+1]
        gamma_est_matrix = gamma_img[x_begin:x_end+1, y_begin:y_end+1]
        theta_est = torch.sum(theta_est_matrix[trust_region_ang, trust_region_ang]) / filter_ang_sum
        phi_est = torch.sum(phi_est_matrix[trust_region_ang, trust_region_ang]) / filter_ang_sum
        gamma_est = torch.sum(gamma_est_matrix[trust_region_ang, trust_region_ang]) / filter_ang_sum
        
        # put the post processed intensity value back to the estimation location pixel
        I_est_img[x_est, y_est] = I_est
        theta_est_img[x_est, y_est] = theta_est
        phi_est_img[x_est, y_est] = phi_est
        gamma_est_img[x_est, y_est] = gamma_est
        # x_est_save[ii] = x_est
        # y_est_save[ii] = y_est
    
    post_ests = torch.stack([I_est_img, theta_est_img, phi_est_img, gamma_est_img], 0)
    
    return post_ests#,x_est_save,y_est_save

def plot_mean(mean_phi,mean_theta,mean_omega, phi_sim,theta_sim, omega_sim, phi_sim1,xlimrange):
    fig = plt.figure(figsize = (18,20))
    fig.add_subplot(3,3,1)
    plt.plot(phi_sim1, phi_sim, '--')
    plt.plot(phi_sim1, mean_phi[0,:,0], color = "blue")
    plt.plot(phi_sim1, mean_phi[1,:,0], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean phi")
    plt.title("Omega 4")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,2)
    plt.plot(phi_sim1, phi_sim, '--')
    plt.plot(phi_sim1, mean_phi[0,:,1], color = "blue")
    plt.plot(phi_sim1, mean_phi[1,:,1], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean phi")
    plt.title("Omega 2")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,3)
    plt.plot(phi_sim1, phi_sim, '--')
    plt.plot(phi_sim1, mean_phi[0,:,2], color = "blue")
    plt.plot(phi_sim1, mean_phi[1,:,2], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean phi")
    plt.title("Omega 0")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,4)
    plt.plot(phi_sim1, np.linspace(theta_sim[0], theta_sim[0], len(phi_sim)), '--', color = "blue")
    plt.plot(phi_sim1, np.linspace(theta_sim[1], theta_sim[1], len(phi_sim)), '--', color = "orange")
    plt.plot(phi_sim1, mean_theta[0,:,0], color = "blue")
    plt.plot(phi_sim1, mean_theta[1,:,0], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean theta")
    plt.title("Omega 4")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,5)
    plt.plot(phi_sim1, np.linspace(theta_sim[0], theta_sim[0], len(phi_sim)), '--', color = "blue")
    plt.plot(phi_sim1, np.linspace(theta_sim[1], theta_sim[1], len(phi_sim)), '--', color = "orange")
    plt.plot(phi_sim1, mean_theta[0,:,1], color = "blue")
    plt.plot(phi_sim1, mean_theta[1,:,1], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean theta")
    plt.title("Omega 2")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,6)
    plt.plot(phi_sim1, np.linspace(theta_sim[0], theta_sim[0], len(phi_sim)), '--', color = "blue")
    plt.plot(phi_sim1, np.linspace(theta_sim[1], theta_sim[1], len(phi_sim)), '--', color = "orange")
    plt.plot(phi_sim1, mean_theta[0,:,2], color = "blue")
    plt.plot(phi_sim1, mean_theta[1,:,2], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean theta")
    plt.title("Omega 0")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,7)
    plt.plot(phi_sim1, np.linspace(omega_sim[0], omega_sim[0], len(phi_sim)), '--')
    plt.plot(phi_sim1, np.linspace(omega_sim[0], omega_sim[0], len(phi_sim)), '--')
    plt.plot(phi_sim1, mean_omega[0,:,0], color = "blue")
    plt.plot(phi_sim1, mean_omega[1,:,0], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean omega")
    plt.title("Omega 4")
    plt.xlim(xlimrange)

    fig.add_subplot(3,3,8)
    plt.plot(phi_sim1, np.linspace(omega_sim[1], omega_sim[1], len(phi_sim)), '--')
    plt.plot(phi_sim1, np.linspace(omega_sim[1], omega_sim[1], len(phi_sim)), '--')
    plt.plot(phi_sim1, mean_omega[0,:,1], color = "blue")
    plt.plot(phi_sim1, mean_omega[1,:,1], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean omega")
    plt.title("Omega 2")
    plt.xlim(xlimrange)
    
    fig.add_subplot(3,3,9)
    plt.plot(phi_sim1, np.linspace(omega_sim[2], omega_sim[2], len(phi_sim)), '--')
    plt.plot(phi_sim1, np.linspace(omega_sim[2], omega_sim[2], len(phi_sim)), '--')
    plt.plot(phi_sim1, mean_omega[0,:,2], color = "blue")
    plt.plot(phi_sim1, mean_omega[1,:,2], color = "orange")
    plt.xlabel("phi")
    plt.ylabel("mean omega")
    plt.title("Omega 0")
    plt.xlim(xlimrange)
    
    plt.show()

    return fig

def gaussian_filter(sigma, shape):
    x = np.arange(-(shape[0] - 1) // 2, (shape[0] - 1) // 2 + 1)
    y = np.arange(-(shape[1] - 1) // 2, (shape[1] - 1) // 2 + 1)

    X,Y = np.meshgrid(x,y)
    weights = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    weights = weights / np.max(weights)

    return weights

def process_val_result(recovery):

    # The validation data generate theta from [70,]
    theta_sim = [70,90]
    phi_sim1 = list(np.arange(0, 359, 10))
    phi_sim = list(np.remainder(phi_sim1,180.01))
    gamma_sim = [0.5731,0.7739,1]
    omega_sim = [4,2,0]

    I_thresh = 10 # thresh to get the mask for localization estimation; adjust these value based on your SNR

    # two filters used in the training
    sigma_loc = 1
    shape_loc = np.array([7,7])
    radius_loc = (shape_loc - 1) // 2
    filter_loc = torch.from_numpy(gaussian_filter(sigma_loc, shape_loc)).float()

    shape_ang = np.array([7,7])
    radius_ang = (shape_ang - 1) // 2
    filter_ang = torch.from_numpy(np.ones(shape_ang)).float()

    frame_numb = 0
    frame_per_state = 10 #20   # each orientation state simulated 20 images; 

    mean_theta = np.zeros([len(theta_sim), len(phi_sim), len(omega_sim)])
    mean_phi = np.zeros([len(theta_sim), len(phi_sim), len(omega_sim)])
    mean_omega = np.zeros([len(theta_sim), len(phi_sim), len(omega_sim)])
    # validation loss
    MSE_M = np.zeros([len(theta_sim), len(phi_sim), len(omega_sim)])

    # match the for-loop order with generated validatation data
    for ii in range(len(theta_sim)):
        for jj in range(len(phi_sim)):
            for kk in range(len(omega_sim)):
                theta_est = np.empty([100,frame_per_state])
                theta_est[:] = np.NaN # Initial everything to be nan
                phi_est = np.empty([100,frame_per_state])
                phi_est[:] = np.NaN
                omega_est = np.empty([100,frame_per_state])
                omega_est[:] = np.NaN
                gamma_est = np.empty([100,frame_per_state])
                gamma_est[:] = np.NaN
                I_est = np.empty([100,frame_per_state])
                I_est[:] = np.NaN
                
                gt_theta = theta_sim[ii]
                gt_phi = phi_sim[jj]
                gt_omega = omega_sim[kk]
                gt_gamma = gamma_sim[kk]
                
                for ll in range(frame_per_state):
                    frame_numb = ii * len(phi_sim) * len(omega_sim) * frame_per_state + jj * len(omega_sim) * frame_per_state + kk * frame_per_state + ll
                    pre_ests = recovery[frame_numb,:,:,:].squeeze() # prediction result
                    
                    post_ests = postprocessing(pre_ests, I_thresh, radius_loc, radius_ang, filter_loc, filter_ang)

                    i_mask = post_ests[0,:,:].squeeze()
                    i_mask[i_mask > 1] = 1
                    # change to list of estimation
                    th_ch = post_ests[1,:,:].squeeze()
                    phi_ch = post_ests[2,:,:].squeeze()
                    gm_ch = post_ests[3,:,:].squeeze()
                    
                    n_sm = torch.sum(i_mask == 1) #len(th_ch(i_mask==1));
                    theta_est[:n_sm, ll] = th_ch[i_mask == 1]
                
                    #phi_est1 = phi_ch[i_mask==1]
                    #phi_est2 = 180 - phi_est1
                    #[~,ind] = min([abs(phi_est1-phi_sim(jj)),abs(phi_est2-phi_sim(jj))].');
                    #temp = [phi_est1,phi_est2].';
                    phi_est[:n_sm,ll] = phi_ch[i_mask==1] # phi_est1
                    
                    gamma_est_temp = gm_ch[i_mask == 1]
                    gamma_est_temp = np.clip(gamma_est_temp, 0, 1)
                    alpha = np.arccos(np.sqrt(2 * gamma_est_temp + 1/4) - 1/2)
                    omega_est[:n_sm,ll] = 8 * np.pi * np.sin(alpha / 2) ** 2
                    gamma_est[:n_sm,ll] = gamma_est_temp
                
                muxx_est,muyy_est,muzz_est,muxy_est,muxz_est,muyz_est = Quickly_rotating_matrix_angleD_gamma_to_M(theta_est,phi_est,gamma_est)
                muxx_gt,muyy_gt,muzz_gt,muxy_gt,muxz_gt,muyz_gt = Quickly_rotating_matrix_angleD_gamma_to_M(gt_theta,gt_phi,gt_gamma)
                # MSE_M[ii,jj,kk] = np.nansum((muxx_est - muxx_gt) ** 2) + np.nansum((muyy_est - muyy_gt) ** 2) \
                #                     + np.nansum((muzz_est - muzz_gt) ** 2) + np.nansum((muxy_est - muxy_gt) ** 2) \
                #                     + np.nansum((muxz_est - muxz_gt) ** 2) + np.nansum((muyz_est - muyz_gt) ** 2)
                MSE_M[ii,jj,kk] = np.nanmean(np.array([muxx_est - muxx_gt, muyy_est - muyy_gt,muzz_est - muzz_gt,
                                            muxy_est - muxy_gt, muxz_est - muxz_gt, muyz_est - muyz_gt]) ** 2)

                # theta_est[theta_est == 0] = np.nan
                # phi_est[phi_est == 0] = np.nan
                # omega_est[omega_est == 0] = np.nan
                mean_theta[ii,jj,kk] = np.nanmean(theta_est)
                mean_phi[ii,jj,kk] = np.nanmean(phi_est)
                mean_omega[ii,jj,kk] = np.nanmean(omega_est)

    # final validation loss
    MSE_M_final = np.nanmean(MSE_M)

    # compare the mean value to the ground truth
    xlimrange = [0,360]
    fig = plot_mean(mean_phi,mean_theta,mean_omega, phi_sim,theta_sim, omega_sim, phi_sim1,xlimrange)
    return MSE_M_final, fig


def plot_zoom_in(target, prediction, n_sm = 1):
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    if n_sm == 1:
        max_loc = np.argmax(target)
        x_GT = (max_loc / target.shape[1]).astype(int)
        y_GT = (max_loc % target.shape[1]).astype(int)
        fig = plt.figure(figsize = (15,10))
        fig.add_subplot(1,2,1)
        plt.imshow(target[x_GT-5:x_GT+6, y_GT-5:y_GT+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Ground Truth")
        fig.add_subplot(1,2,2)
        plt.imshow(prediction[x_GT-5:x_GT+6, y_GT-5:y_GT+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Prediction")
    else:
        x_GT, y_GT = find_gt(target)
        x_GT = x_GT.flatten().astype(int)
        y_GT = y_GT.flatten().astype(int)
        fig = plt.figure(figsize = (15,20))
        fig.add_subplot(2,2,1)
        plt.imshow(target[x_GT[0]-5:x_GT[0]+6, y_GT[0]-5:y_GT[0]+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Ground Truth of first SM")
        fig.add_subplot(2,2,2)
        plt.imshow(prediction[x_GT[0]-5:x_GT[0]+6, y_GT[0]-5:y_GT[0]+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Prediction of first SM")
        fig.add_subplot(2,2,3)
        plt.imshow(target[x_GT[1]-5:x_GT[1]+6, y_GT[1]-5:y_GT[1]+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Ground Truth of second SM")
        fig.add_subplot(2,2,4)
        plt.imshow(prediction[x_GT[1]-5:x_GT[1]+6, y_GT[1]-5:y_GT[1]+6])
        plt.scatter(5,5, marker="x", color="black", s=100)
        plt.colorbar()
        plt.title("Prediction of second SM")
    
    return fig




def eval_val_metric_zoom(data, label, output, n_sm = 1): # Added for only localization
    rawx = data[:,0,:,:]
    rawy = data[:,1,:,:]
    

    estimated_loc = output[:,0,:,:]
    gt_loc = label[:,0,:,:] 
    total_num = output.shape[0]
    W = output.shape[-1]
    crop = 50

    samp_to_show = int(torch.randint(0,total_num,[1]))
    fig = plot_comparison(rawx[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                        rawy[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                        gt_loc[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                        estimated_loc[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)])
    #fig2 = plot_zoom_in(gt_loc[samp_to_show], estimated_loc[samp_to_show], n_sm = n_sm)

    return fig
