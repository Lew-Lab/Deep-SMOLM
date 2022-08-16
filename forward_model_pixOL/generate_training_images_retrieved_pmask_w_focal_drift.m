%% description: for generating training data using calibrated phase mask & including the focal plane driftion 

clear;
clc;


%% parameter of the microscopy

% give the save address for generated data
% ********************************
save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/training_20220720_pixOL_retrieved_SNR2000vs6_gamma_linear_photon_poisson_distribution_cubic_model/'; 
% ********************************
image_size = 60;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
% give the name of calibrated phase mask for generated data
% ********************************
pmask_retrieve_name = '20220528_pixOLcom_retrieved.mat';
% ********************************
zf_set = linspace(-150,150,40);
basis_matrix_opt_set = forward_model_opt_3D_retrieved_w_z(pmask, image_size,pmask_retrieve_name,zf_set*10^-9);
%basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.6; %in unit of um

%% gaussian filter
h_shape = [7,7];
h_sigma = 1;
[x,y] = meshgrid([-(h_shape(1)-1)/2:(h_shape(1)-1)/2]);
h = exp(-(x.^2+y.^2)/(2*h_sigma^2));
h = h./max(max(h));


%% user defined parameters

n_images = 1; % the simulated image numbers (feel free to change it)
signal= 1000; %(feel free to change it)
background_avg=6; %(feel free to change it)
SM_num_range = 8;
SM_num_min = 7;


for ii = (2337:2337+32)+32 %each 4 images, and total 2000*4 images
   
if rem(ii,100)==0
   ii
end
x_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation
y_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation
x_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation (phyiscal distance)
y_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation (phyiscal distance)
thetaD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the thetaD
phiD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the phi
gamma_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the gamma 
I_grd = nan(SM_num_range+SM_num_min,1);

image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = floor(rand(1)*SM_num_range+SM_num_min); % number of single molecules

%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_gamma_linear_distribution(n_SMs);


x_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %x location, in unit of pixles
y_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %y location, in unit of pixles
temp = (poissrnd(3,1,100000)+normrnd(0,1,1,100000)-0.5)*350; temp(temp<100)=[];  %mean(temp)
%temp=generateSignal_distribution(); temp(temp<100)=[];


signal_SMs = temp(1:n_SMs)*2;
x_SMs_phy = x_SMs*pixel_size;
y_SMs_phy = y_SMs*pixel_size;
z_SMs_phy = (rand(1,n_SMs)*300-150);

% save the list of the ground truth
x_grd(1:n_SMs) = x_SMs.'; y_grd(1:n_SMs) = y_SMs.';  x_phy(1:n_SMs) = x_SMs_phy.'; y_phy(1:n_SMs) = y_SMs_phy.'; 
thetaD_grd(1:n_SMs) = thetaD_SMs.'; phiD_grd(1:n_SMs)=phiD_SMs.'; 
gamma_grd(1:n_SMs) = gamma_SMs.'; I_grd(1:n_SMs) = signal_SMs.'; 

% adding variation to background and background across two channels
background = rand(1)*2-1+background_avg;
bkg_img = [ones(image_size,image_size)*background,ones(image_size,image_size)*background*(rand(1)*0.2+1.1)];
%% forward imaging system & basis image caculation
[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];
I_SMs = [];
for dd = 1:n_SMs
    [~,set_indx] = min(abs(z_SMs_phy(dd)-zf_set));
    basis_matrix_opt = basis_matrix_opt_set(:,:,set_indx); 
    I_SMs(:,dd) = basis_matrix_opt*M(:,dd);
end
I_SMs = reshape(I_SMs,image_size,image_size*2,n_SMs);
I_SMsx = I_SMs(1:image_size,1:image_size,:);
I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
I_SMsy = flip(I_SMsy,2);
%% create GT image and SMLM image
I = bkg_img;
Ix = I(1:image_size,1:image_size);
Iy = I(1:image_size,image_size+1:image_size*2);
I_basis = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);


I_intensity_gaussian = I_intensity_up;
I_sXX = I_intensity_up;
I_sYY = I_intensity_up;
I_sZZ = I_intensity_up;
I_sXY = I_intensity_up;
I_sXZ = I_intensity_up;
I_sYZ = I_intensity_up;


h_basis = I_basis;
h_basis(round((size(I_basis,1)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2],round((size(I_basis,2)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2]) = h;
I_basis(round((size(I_basis,1)+1)/2),round((size(I_basis,2)+1)/2)) = 1;


for i = 1:n_SMs
Ix = Ix+imtranslate(I_SMsx(:,:,i),[x_SMs(i),y_SMs(i)],'bicubic')*signal_SMs(i);
Iy = Iy+imtranslate(I_SMsy(:,:,i),[x_SMs(i),y_SMs(i)],'bicubic')*signal_SMs(i);


temp = imtranslate(h_basis,[(x_SMs(i)*upsampling_ratio),(y_SMs(i)*upsampling_ratio)],'bicubic');
I_intensity_gaussian = I_intensity_gaussian+temp*signal_SMs(i);

I_sXX = I_sXX+temp*signal_SMs(i)*muxx(i);
I_sYY = I_sYY+temp*signal_SMs(i)*muyy(i);
I_sZZ = I_sZZ+temp*signal_SMs(i)*muzz(i);
I_sXY = I_sXY+temp*signal_SMs(i)*muxy(i);
I_sXZ = I_sXZ+temp*signal_SMs(i)*muxz(i);
I_sYZ = I_sYZ+temp*signal_SMs(i)*muyz(i);


end

%% save ground truth and single-molecule images
image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;

image_GT_up(1,:,:) = I_intensity_gaussian;
image_GT_up(2,:,:) = I_XX;
image_GT_up(3,:,:) = I_YY;
image_GT_up(4,:,:) = I_ZZ;
image_GT_up(5,:,:) = I_XY;
image_GT_up(6,:,:) = I_XZ;
image_GT_up(7,:,:) = I_YZ;

image_GT_up(8,:,:) = I_sXX;
image_GT_up(9,:,:) = I_sYY;
image_GT_up(10,:,:) = I_sZZ;
image_GT_up(11,:,:) = I_sXY;
image_GT_up(12,:,:) = I_sXZ;
image_GT_up(13,:,:) = I_sYZ;


GT_list(1,:)=ones(size(x_phy))*ii;
GT_list(2,:)=x_phy;
GT_list(3,:)=y_phy;
GT_list(4,:)=I_grd;
GT_list(5,:)=thetaD_grd;
GT_list(6,:)=phiD_grd;
GT_list(7,:)=gamma_grd;
img_bkg = background;

save([save_folder,'img_bkg',num2str(ii),'.mat'],'img_bkg');
save([save_folder,'image_GT_up',num2str(ii),'.mat'],'image_GT_up');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');



end