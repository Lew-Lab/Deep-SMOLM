%% generate validation image that only contains 1 SM at a defined 2D location but with random orientation
% this data is used for generating the figure 2(a) in Deep-SMOLM paper



%% image generation
clear;
clc;

% give the save address for generated data
% ********************************
save_folder = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data for Deep-SMOLM\validation_20220114_SNR1000vs2_1SM_x1y0upsampled\'; 
% ********************************
image_size = 32;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.6; %in unit of um

definedX = 1; %defined 2D position of emitter, unit [upsampled grid spacing=9.75]
definedY = 0; 
%% gaussian filter

h_shape = [7,7];
h_sigma = 1;
[x,y] = meshgrid([-(h_shape(1)-1)/2:(h_shape(1)-1)/2]);
h = exp(-(x.^2+y.^2)/(2*h_sigma^2));
h = h./max(max(h));
%% user defined parameters

n_images = 1; % the simulated image numbers (feel free to change it)
signal= 1000; %(feel free to change it)
background=2 ; %(feel free to change it)


for ii = 1:2000  %each 4 images, and total 2000*4 images
if rem(ii,100)==0
   ii
end


image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 1; % number of single molecules
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs);


x_SMs = definedX/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio; %x location, in unit of pixles
y_SMs = definedY/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio;%y location, in unit of pixles
% temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<100)=[];
% signal_SMs = temp(1:n_SMs);
signal_SMs = 1000;
x_SMs_phy = x_SMs*pixel_size;
y_SMs_phy = y_SMs*pixel_size;

% save the list of the ground truth
x_grd= x_SMs.'; y_grd= y_SMs.';  x_phy= x_SMs_phy.'; y_phy= y_SMs_phy.'; 
thetaD_grd= thetaD_SMs.'; phiD_grd=phiD_SMs.'; 
gamma_grd = gamma_SMs.'; I_grd = signal_SMs.'; 




%% forward imaging system & basis image caculation


[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];
I_SMs = basis_matrix_opt*M;
I_SMs = reshape(I_SMs,image_size,image_size*2,n_SMs);
I_SMsx = I_SMs(1:image_size,1:image_size,:);
I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
I_SMsy = flip(I_SMsy,2);
%% create GT image and SMLM image
I = ones(image_size,image_size*2)*background;
bkg_img = I;
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

I_poissx = poissrnd(Ix); 
I_poissy = poissrnd(Iy);

%save ground truth and image
image_with_poission(1,:,:) = I_poissx;
image_with_poission(2,:,:) = I_poissy;
image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;

image_GT_up(1,:,:) = I_intensity_gaussian;
image_GT_up(2,:,:) = I_sXX;
image_GT_up(3,:,:) = I_sYY;
image_GT_up(4,:,:) = I_sZZ;
image_GT_up(5,:,:) = I_sXY;
image_GT_up(6,:,:) = I_sXZ;
image_GT_up(7,:,:) = I_sYZ;
img_bkg = bkg_img;

GT_list(1,:)=x_phy;
GT_list(2,:)=y_phy;
GT_list(3,:)=I_grd;
GT_list(4,:)=thetaD_grd;
GT_list(5,:)=phiD_grd;
GT_list(6,:)=gamma_grd;


save([save_folder,'img_bkg',num2str(ii),'.mat'],'img_bkg');
save([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
save([save_folder,'image_GT_up',num2str(ii),'.mat'],'image_GT_up');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');



end