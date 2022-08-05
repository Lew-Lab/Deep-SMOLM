% clc;
% clear;


%% parameter of the microscopy

% give the save address for generated data
% ********************************

%save_folder = '..\Examples\training_data\example1_100images'; 
save_folder = '/home/wut/Documents/Deep-SMOLM/data/vortex_PSF_data/training_20220804_vortex_500_3.5/';
% ********************************
image_size = 60;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
%pmask = 'pixOL_v12.bmp';
pmask = 'vortex_v2.bmp';
pmask_retrieve_name = '20220214_pixOL_com_retrieve.mat';
%basis_matrix_opt = forward_model_opt_retrieved(pmask, image_size,pmask_retrieve_name);
basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.5; %in unit of um
%% gaussian filter
h_shape = [7,7];
h_sigma = 1;
[x,y] = meshgrid([-(h_shape(1)-1)/2:(h_shape(1)-1)/2]);
h = exp(-(x.^2+y.^2)/(2*h_sigma^2));
h = h./max(max(h));


%% user defined parameters

n_images = 30000; % the simulated image numbers (feel free to change it); 30K is used for training Deep-SMOLM
signal= 500; %(feel free to change it)
background_avg=3.5; %(feel free to change it)

SM_num_range = 8;
SM_num_min = 7;


for ii = 1:n_images %each 4 images, and total 2000*4 images

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


n_SMs = floor(rand(1)*SM_num_range+SM_num_min); % number of single molecules
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_gamma_linear_distribution(n_SMs);


x_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %x location, in unit of pixles
y_SMs = (0.10+0.8*rand(1,n_SMs))*image_size-(image_size)/2; %y location, in unit of pixles
temp = (poissrnd(3,1,100000)+normrnd(0,1,1,100000)-0.5)*350; temp(temp<100)=[];  %create broad signal distribution with mean of 1000

signal_SMs = temp(1:n_SMs)*signal/1000; % assign signal to emitters
x_SMs_phy = x_SMs*pixel_size;
y_SMs_phy = y_SMs*pixel_size;

% save the list of the ground truth
x_grd(1:n_SMs) = x_SMs.'; y_grd(1:n_SMs) = y_SMs.';  x_phy(1:n_SMs) = x_SMs_phy.'; y_phy(1:n_SMs) = y_SMs_phy.'; 
thetaD_grd(1:n_SMs) = thetaD_SMs.'; phiD_grd(1:n_SMs)=phiD_SMs.'; 
gamma_grd(1:n_SMs) = gamma_SMs.'; I_grd(1:n_SMs) = signal_SMs.'; 


background = background_avg;

%% forward imaging system

[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];
I_SMs = basis_matrix_opt*M;
I_SMs = reshape(I_SMs,image_size,image_size*2,n_SMs);
I_SMsx = I_SMs(1:image_size,1:image_size,:);
I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
I_SMsy = flip(I_SMsy,2);
%% generate the basis image
I = ones(image_size,image_size*2)*background;   
bkg_img = I;
Ix = I(1:image_size,1:image_size);
Iy = I(1:image_size,image_size+1:image_size*2);

% 6 channels
I_intensity_gaussian = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sXX = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sYY = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sZZ = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sXY = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sXZ = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
I_sYZ = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);

h_basis = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
h_basis(round((size(h_basis,1)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2],round((size(h_basis,2)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2]) = h;


%generate ground truth image and single moleucle image
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

image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;

image_GT_up(1,:,:) = I_intensity_gaussian;

image_GT_up(2,:,:) = I_sXX;
image_GT_up(3,:,:) = I_sYY;
image_GT_up(4,:,:) = I_sZZ;
image_GT_up(5,:,:) = I_sXY;
image_GT_up(6,:,:) = I_sXZ;
image_GT_up(7,:,:) = I_sYZ;


GT_list(1,:)=ones(size(x_phy))*ii;
GT_list(2,:)=x_phy;
GT_list(3,:)=y_phy;
GT_list(4,:)=I_grd;
GT_list(5,:)=thetaD_grd;
GT_list(6,:)=phiD_grd;
GT_list(7,:)=gamma_grd;
img_bkg = bkg_img;




save([save_folder,'img_bkg',num2str(ii),'.mat'],'img_bkg');
save([save_folder,'image_GT_up',num2str(ii),'.mat'],'image_GT_up');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');



end