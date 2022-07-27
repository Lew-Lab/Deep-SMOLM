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
save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/validation_20220722_2SM_fixed_v2_seperation1to20_signal500_gamma1/'; 
% ********************************
image_size = 56;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.6; %in unit of um

%distance_differ_set = linspace(1,1000,30)/pixel_size; %in unit of pixel
distance_differ_set = 0/pixel_size;
frame_per_state = 1000;
%% user defined parameters
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
signal_sigma = 80;

kk=0;
%
for ii = 1:frame_per_state*length(distance_differ_set)  %each 4 images, and total 2000*4 images
if rem(ii-1,frame_per_state)==0
   ii
   kk=kk+1;
   kk
end


image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 2; % number of single molecules
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs);
gamma_SMs(:)=1;
%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)


x_SMs1 = (1*rand(1)-1/2); %x location, in unit of pixles
y_SMs1 = (1*rand(1)-1/2);%y location, in unit of pixles
angle = rand(1)*360;
x_dif = distance_differ_set(kk)*cosd(angle);
y_dif = distance_differ_set(kk)*sind(angle);
x_SMs2 = x_SMs1+x_dif;
y_SMs2 = y_SMs1+y_dif;
x_mean = (x_SMs1+x_SMs2)/2;
y_mean = (y_SMs1+y_SMs2)/2;
x_SMs1 = x_SMs1-x_mean;
x_SMs2 = x_SMs2-x_mean;
y_SMs1 = y_SMs1-y_mean;
y_SMs2 = y_SMs2-y_mean;

x_SMs = [x_SMs1,x_SMs2];
y_SMs = [y_SMs1,y_SMs2];

temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<100)=[];
%signal_SMs = temp(1:n_SMs);
signal_SMs = [signal,signal];
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

cut = 0.05;
Ix = I_SMs(1:end,1:56,1);
Iy = I_SMs(1:end,(1:56)+56,1);
Ix = Ix+imtranslate(Ix(:,:),[x_SMs(1),y_SMs(1)],'bicubic')*signal_SMs(1);
Iy = Iy+imtranslate(Iy(:,:),[x_SMs(1),y_SMs(1)],'bicubic')*signal_SMs(1);
I_SM1 = [Ix,Iy];
indx1 = [Ix,Iy]; indx1(indx1>cut*max(I_SM1,[],'all'))=1; indx1(indx1~=1)=0;
Ix = I_SMs(1:end,1:56,2);
Iy = I_SMs(1:end,(1:56)+56,2);
Ix = Ix+imtranslate(Ix(:,:),[x_SMs(2),y_SMs(2)],'bicubic')*signal_SMs(2);
Iy = Iy+imtranslate(Iy(:,:),[x_SMs(2),y_SMs(2)],'bicubic')*signal_SMs(2);
I_SM2 = [Ix,Iy];
indx2 = [Ix,Iy]; indx2(indx2>cut*max(I_SM2,[],'all'))=1; indx2(indx2~=1)=0;
overlap(ii) = sum(indx1.*indx2,'all')/(sum((indx1+indx2)/2,'all'));

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
temp1 = imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)]);
I_intensity_up = I_intensity_up+temp1*signal_SMs(i);
I_theta_up = I_theta_up+temp1*thetaD_SMs(i);
I_phi_up = I_phi_up+temp1*phiD_SMs(i);
I_gamma_up = I_gamma_up+temp1*gamma_SMs(i);
%I_dx_up = I_dx_up+temp1*(x_SMs(i)*upsampling_ratio-round(x_SMs(i)*upsampling_ratio));
%I_dy_up = I_dy_up+temp1*(y_SMs(i)*upsampling_ratio-round(y_SMs(i)*upsampling_ratio));

temp = imtranslate(h_basis,[(x_SMs(i)*upsampling_ratio),(y_SMs(i)*upsampling_ratio)],'bicubic');
I_intensity_gaussian = I_intensity_gaussian+temp*signal_SMs(i);
I_sXX = I_sXX+temp*signal_SMs(i)*muxx(i);
I_sYY = I_sYY+temp*signal_SMs(i)*muyy(i);
I_sZZ = I_sZZ+temp*signal_SMs(i)*muzz(i);
I_sXY = I_sXY+temp*signal_SMs(i)*muxy(i);
I_sXZ = I_sXZ+temp*signal_SMs(i)*muxz(i);
I_sYZ = I_sYZ+temp*signal_SMs(i)*muyz(i);
end

I_poissx = poissrnd(Ix); % if you need multiple realization for a single ground truth, modify here
%imagesc(I_poiss); axis image;
I_poissy = poissrnd(Iy);
I_poissx_up = imresize(I_poissx,[image_size,image_size]*upsampling_ratio,'box');  
I_poissy_up = imresize(I_poissy,[image_size,image_size]*upsampling_ratio,'box'); 
Ix_up = imresize(Ix,[image_size,image_size]*upsampling_ratio,'box');  
Iy_up = imresize(Iy,[image_size,image_size]*upsampling_ratio,'box'); 

%save ground truth and image
image_with_poission(1,:,:) = I_poissx;
image_with_poission(2,:,:) = I_poissy;

image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;
GT_list(1,:)=ones(size(x_phy))*ii;
GT_list(2,:)=x_phy;
GT_list(3,:)=y_phy;
GT_list(4,:)=I_grd;
GT_list(5,:)=thetaD_grd;
GT_list(6,:)=phiD_grd;
GT_list(7,:)=gamma_grd;


% I_noise(ii,:,:) = [I_poissx,I_poissy];
% I_noiseless_save(ii,:,:) =[Ix,Iy];
% distance_differ_save(ii) = distance_differ_set(ii);
% GT_list_save(ii,:,:)=GT_list;
save([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');



end

%save('2SMs_image_demo.mat','I_noise','I_noiseless_save','distance_differ_save','GT_list_save');