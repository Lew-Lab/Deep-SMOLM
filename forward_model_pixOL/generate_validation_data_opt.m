%% 2020/02/20 Ting: correct the noise model; 
%                   generate a small size basis matrix
%                   change to high SNR situation: signal 1000, background 5
%% 2020/02/22 Asheq: Modified saving the x and y channel psf map without
% noise

%%

% clear;
% clc;
% %%
% save_folder = 'simulate_image_correct_noise_Feb22\'; 
% % generate a small size basis and upsampling it
% 
% image_size = 17;  % the pixel size of the simulation image (feel free to change it)
% upsampling_ratio  = 6;
% pmask = 'clear plane.bmp';
% basis_matrix_SD_s1 = forward_model(pmask, image_size);
% imsize = size(basis_matrix_SD_s1);
% basis_matrix_SD_temp = reshape(basis_matrix_SD_s1,image_size,image_size*2,6);
% %
% for i = 1:6
%   basis_matrix_SD(:,:,i) = imresize(basis_matrix_SD_temp(:,:,i),[image_size,image_size*2]*upsampling_ratio,'box');
% end
% 
% save([save_folder,'basis_matrix_SD.mat'],'basis_matrix_SD');



%% image generation
clear;
clc;

% give the save address for generated data
% ********************************
save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data/validation_SNR380vs5_min7_U8_2021Jan7/'; 
% ********************************
image_size = 60;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'opt_v11.bmp';
basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.6; %in unit of um

%% user defined parameters
theta_list = [70,90];
phi_list = [0:10:360];
gamma_list = [0.5731,0.7739,1];
num_each_state = 20;



theta_list_length = length(theta_list);
phi_list_length = length(phi_list);
gamma_list_length = length(gamma_list);

n_images = theta_list_length*phi_list_length*gamma_list_length*num_each_state; % the simulated image numbers (feel free to change it)


signal= 380; %(feel free to change it)
background=2 ; %(feel free to change it)
signal_sigma = 80;
SM_num_range = 8;
SM_num_min = 7;



zz=0;

for ii = 1:theta_list_length
    for jj = 1:phi_list_length
        for kk = 1:gamma_list_length
            for ll = 1:num_each_state
                zz=zz+1
                x_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation
                y_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation
                x_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation (phyiscal distance)
                y_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation (phyiscal distance)
                thetaD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the thetaD
                phiD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the phi
                gamma_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the gamma 
                I_grd = nan(SM_num_range+SM_num_min,1);
                
                val_image_with_poission = zeros(2,image_size,image_size);
                val_image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
                val_image_GT_up = zeros(4,image_size*upsampling_ratio,image_size*upsampling_ratio);

                n_SMs = floor(rand(1)*SM_num_range+SM_num_min); % number of single molecules
                thetaD_SMs = theta_list(ii)*ones(1,n_SMs); %theta angle of SMs, note theta is in the range of (0,90) degree
                phiD_SMs = phi_list(jj)*ones(1,n_SMs); %phi angle of SMs, note phi is in the range of (0,180) degree
                gamma_SMs = gamma_list(kk)*ones(1,n_SMs); %gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)
                
                x_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %x location, in unit of pixles
                y_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %y location, in unit of pixles
                signal_SMs = normrnd(signal,signal_sigma,[1,n_SMs]);
                x_SMs_phy = x_SMs*pixel_size;
                y_SMs_phy = y_SMs*pixel_size;
                
                % save the list of the ground truth
                x_grd(1:n_SMs) = x_SMs.'; y_grd(1:n_SMs) = y_SMs.';  x_phy(1:n_SMs) = x_SMs_phy.'; y_phy(1:n_SMs) = y_SMs_phy.'; 
                thetaD_grd(1:n_SMs) = thetaD_SMs.'; phiD_grd(1:n_SMs)=phiD_SMs.'; 
                gamma_grd(1:n_SMs) = gamma_SMs.'; I_grd(1:n_SMs) = signal_SMs.'; 


                

                %% forward model
                [muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
                M = [muxx;muyy;muzz;muxy;muxz;muyz];
                I_SMs = basis_matrix_opt*M;
                I_SMs = reshape(I_SMs,image_size,image_size*2,n_SMs);
                I_SMsx = I_SMs(1:image_size,1:image_size,:);
                I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
                I_SMsy = flip(I_SMsy,2);
                %% generate the basis image
                I = ones(image_size,image_size*2)*background;
                Ix = I(1:image_size,1:image_size);
                Iy = I(1:image_size,image_size+1:image_size*2);
                % I = imresize(I,size(I)*upsampling_ratio,'nearest');
                I_basis = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
                %I_basis = imresize(I_basis,size(I_basis)*upsampling_ratio,'nearest');

                % four channels
                I_intensity_up = I_basis;
                I_theta_up = I_intensity_up;
                I_phi_up = I_intensity_up;
                I_omega_up = I_intensity_up;
                I_gamma_up = I_intensity_up;
                I_dx_up = I_intensity_up;
                I_dy_up = I_intensity_up;
                I_basis(round((size(I_basis,1)+1)/2),round((size(I_basis,2)+1)/2)) = 1;

                for i = 1:n_SMs
                Ix = Ix+imtranslate(I_SMsx(:,:,i),[x_SMs(i),y_SMs(i)])*signal_SMs(i);
                Iy = Iy+imtranslate(I_SMsy(:,:,i),[x_SMs(i),y_SMs(i)])*signal_SMs(i);
                I_intensity_up = I_intensity_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*signal_SMs(i);
                I_theta_up = I_theta_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*thetaD_SMs(i);
                I_phi_up = I_phi_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*phiD_SMs(i);
                I_gamma_up = I_gamma_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*gamma_SMs(i);
                I_dx_up = I_dx_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*(x_SMs(i)*upsampling_ratio-round(x_SMs(i)*upsampling_ratio));
                I_dy_up = I_dy_up+imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)])*(y_SMs(i)*upsampling_ratio-round(y_SMs(i)*upsampling_ratio));
                end

                I_poissx = poissrnd(Ix); % if you need multiple realization for a single ground truth, modify here
                %imagesc(I_poiss); axis image;
                I_poissy = poissrnd(Iy);
                I_poissx_up = imresize(I_poissx,[image_size,image_size]*upsampling_ratio,'box');  
                I_poissy_up = imresize(I_poissy,[image_size,image_size]*upsampling_ratio,'box'); 
                Ix_up = imresize(Ix,[image_size,image_size]*upsampling_ratio,'box');  
                Iy_up = imresize(Iy,[image_size,image_size]*upsampling_ratio,'box'); 

                %save ground truth and image
                val_image_with_poission(1,:,:) = I_poissx;
                val_image_with_poission(2,:,:) = I_poissy;
                val_image_with_poission_up(1,:,:) = I_poissx_up;
                val_image_with_poission_up(2,:,:) = I_poissy_up;
                val_image_GT_up(1,:,:) = I_intensity_up;
                val_image_GT_up(2,:,:) = I_theta_up;
                val_image_GT_up(3,:,:) = I_phi_up;
                val_image_GT_up(4,:,:) = I_gamma_up;
                val_GT_list(1,:)=x_phy;
                val_GT_list(2,:)=y_phy;
                val_GT_list(3,:)=I_grd;
                val_GT_list(4,:)=thetaD_grd;
                val_GT_list(5,:)=phiD_grd;
                val_GT_list(6,:)=gamma_grd;

                val_image_with_poission_bkgdRmvd = val_image_with_poission-background;
                val_image_with_poission_bkgdRmvd_up = val_image_with_poission_up-background;


                save([save_folder,'val_image_with_poission',num2str(zz),'.mat'],'val_image_with_poission');
                save([save_folder,'val_image_with_poission_up',num2str(zz),'.mat'],'val_image_with_poission_up');
                save([save_folder,'val_image_with_poission_bkgdRmvd',num2str(zz),'.mat'],'val_image_with_poission_bkgdRmvd');
                save([save_folder,'val_image_with_poission_bkgdRmvd_up',num2str(zz),'.mat'],'val_image_with_poission_bkgdRmvd_up');
                save([save_folder,'val_image_GT_up',num2str(zz),'.mat'],'val_image_GT_up');
                save([save_folder,'val_GT_list',num2str(zz),'.mat'],'val_GT_list');
        
            end
        end
    end
end
