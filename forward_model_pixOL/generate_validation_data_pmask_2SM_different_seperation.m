%% generate images that containing 2 emitters at defined seperation
% these images are used for generating fig 2(e-h) in Deep-SMOLM paper


clear;
clc;

%%
% give the save address for generated data
% ********************************
save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/validation_20220722_2SM_fixed_v2_seperation1to20_signal500_gamma1/'; 
% ********************************
image_size = 56;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
basis_matrix_opt = forward_model_opt(pmask, image_size);
pixel_size = 58.6; %in unit of um

% the defined distance
distance_differ_set = linspace(1,1000,30)/pixel_size; %in unit of pixel
%distance_differ_set = 0/pixel_size;

% at each speration, the algorithm will generate multiple frames 
frame_per_state = 1000;

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

kk=0;
%
for ii = 1:frame_per_state*length(distance_differ_set)  
if rem(ii-1,frame_per_state)==0
   ii
   kk=kk+1;
end


image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 2; % number of single molecules

%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs);
gamma_SMs(:)=1;


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

%temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<100)=[];
%signal_SMs = temp(1:n_SMs);
signal_SMs = [signal,signal];
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

%------------------------caculate the overlapping percentage----------------
% cut = 0.05;
% Ix = I_SMs(1:end,1:56,1);
% Iy = I_SMs(1:end,(1:56)+56,1);
% Ix = Ix+imtranslate(Ix(:,:),[x_SMs(1),y_SMs(1)],'bicubic')*signal_SMs(1);
% Iy = Iy+imtranslate(Iy(:,:),[x_SMs(1),y_SMs(1)],'bicubic')*signal_SMs(1);
% I_SM1 = [Ix,Iy];
% indx1 = [Ix,Iy]; indx1(indx1>cut*max(I_SM1,[],'all'))=1; indx1(indx1~=1)=0;
% Ix = I_SMs(1:end,1:56,2);
% Iy = I_SMs(1:end,(1:56)+56,2);
% Ix = Ix+imtranslate(Ix(:,:),[x_SMs(2),y_SMs(2)],'bicubic')*signal_SMs(2);
% Iy = Iy+imtranslate(Iy(:,:),[x_SMs(2),y_SMs(2)],'bicubic')*signal_SMs(2);
% I_SM2 = [Ix,Iy];
% indx2 = [Ix,Iy]; indx2(indx2>cut*max(I_SM2,[],'all'))=1; indx2(indx2~=1)=0;
% overlap(ii) = sum(indx1.*indx2,'all')/(sum((indx1+indx2)/2,'all'));

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


%% save ground truth and single-molecule images
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



save([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');



end
