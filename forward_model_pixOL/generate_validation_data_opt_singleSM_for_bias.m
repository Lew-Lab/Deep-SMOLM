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
save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/validation_20220629_SNR1000vs2_1SM_atCenter_for_bias_library/'; 
% ********************************
image_size = 32;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
basis_matrix_opt = forward_model_opt(pmask, image_size);
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
background=2 ; %(feel free to change it)
%signal_sigma = 80;
[thetaD_set,phiD_set,gamma_set] = generate_angleD();

for ii = 1:length(thetaD_set) %each 4 images, and total 2000*4 images
if rem(ii,100)==0
   ii
end


image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 1; % number of single molecules

thetaD_SMs = thetaD_set(ii);
phiD_SMs = phiD_set(ii);
gamma_SMs = gamma_set(ii);
%gamma_SMs(:)=1;
%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)


x_SMs = 0;%0/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio; %x location, in unit of pixles
y_SMs = 0; %0/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio;%y location, in unit of pixles
%temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<100)=[];

signal_SMs = 1000;
x_SMs_phy = x_SMs*pixel_size;
y_SMs_phy = y_SMs*pixel_size;

% save the list of the ground truth
x_grd= x_SMs.'; y_grd= y_SMs.';  x_phy= x_SMs_phy.'; y_phy= y_SMs_phy.'; 
thetaD_grd= thetaD_SMs.'; phiD_grd=phiD_SMs.'; 
gamma_grd = gamma_SMs.'; I_grd = signal_SMs.'; 




%% forward imaging system


[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];
I_SMs = basis_matrix_opt*M;
I_SMs = reshape(I_SMs,image_size,image_size*2,n_SMs);
I_SMsx = I_SMs(1:image_size,1:image_size,:);
I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
I_SMsy = flip(I_SMsy,2);

Ix = signal_SMs*I_SMsx+background;
Iy = signal_SMs*I_SMsy+background;


%save ground truth and image

image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;

GT_list(1)=x_phy;
GT_list(2)=y_phy;
GT_list(3)=I_grd;
GT_list(4)=thetaD_grd;
GT_list(5)=phiD_grd;
GT_list(6)=gamma_grd;
GT_list(7:12) = M;


save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');




end

%%



function [thetaD_set,phiD_set,gamma_set] = generate_angleD()

% generate random angular combination from uniformly sampled space
x1 = linspace(-1,1,50);
x2 = linspace(-1,1,50);
[x1,x2] = meshgrid(x1,x2);


mux = 2*x1.*sqrt(1-x1.^2-x2.^2);
muy = 2*x2.*sqrt(1-x1.^2-x2.^2);
muz = 1-2*(x1.^2+x2.^2);

indx =  muz<0 | x1.^2+x2.^2>1;
mux(indx)=[];
muy(indx)=[];
muz(indx)=[];

thetaD = acos(muz)/pi*180;
phiD = atan2(muy,mux)/pi*180;


x = 0:0.01:1;
Fx = x.^2;
F_dist = makedist('PiecewiseLinear', 'x', x, 'Fx', Fx);
gamma = random(F_dist, 100000,1);
gamma_sort = sort(gamma);
gamma_set = gamma_sort(1:20000:100000);
gamma_set1 = [gamma_set;1];

[thetaD_set,gamma_set] = meshgrid(thetaD,gamma_set1);
[phiD_set,gamma_set] = meshgrid(phiD,gamma_set1);
thetaD_set = thetaD_set(:);
phiD_set  = phiD_set(:);
gamma_set = gamma_set(:);
end