function [basis_matrix,mask,BFP_matrix] = basis_matrix_pixel_based_v2(Microscopy)

name=Microscopy.mask;
rot=Microscopy.rot;
wavelength=Microscopy.wavelength;
bfp_radius=Microscopy.bfp_radius;
n1=Microscopy.n1;
n2=Microscopy.n2;
nh=Microscopy.nh;
NA=Microscopy.NA;
Magnitude=Microscopy.Magnitude;
sampling_size=Microscopy.sampling_size;
image_size=Microscopy.image_size;
pix_size=Microscopy.pix_size;
upsamping=Microscopy.upsampling;
pixelSizeUpsampling=Microscopy.pixelSizeUpsampling;

%MaskResized=mask_resize(name,bfp_radius,rot,sampling_size);

%1. generate the mask
angle_temp = imread(name);
angle_temp = rot90(angle_temp,rot);
angle_1 = ones(sampling_size, sampling_size)*127;
angle_temp = im2double(angle_temp,'indexed');
angleResize = imresize(angle_temp,upsamping);

center = round(sampling_size/2);
[psf_radus aa] = size(angleResize);
angle_1(center-psf_radus/2+1:center+psf_radus/2,center-psf_radus/2+1:center+psf_radus/2)=angleResize(:,:);
%angle_1(219-48:378+48,219-48:378+48)=angleResize(:,:);
angle_1 = ((angle_1/255)*2*pi)-pi;
mask = exp(1i*angle_1);


%2.a generate the basis image
[basisImagex,basisImagey,BFP_image_x,BFP_image_y] = simDipole_v5(0,0,0,mask,sampling_size,wavelength,n1,n2,nh,NA,Magnitude,pix_size);
%[basisImagex,basisImagey,BFP_image_x,BFP_image_y] = simDipole_190703(0,0,0,mask,sampling_size,wavelength,n1,n2,NA,Magnitude);


%intensity = sum(sum(sum(basisImagex+basisImagey)))/3;
basisx = basisImagex(sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,:);
basisy = basisImagey(sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,:);

[a,b]=size(BFP_image_x);

BFP_x = BFP_image_x(a/2-bfp_radius+1:a/2+bfp_radius,a/2-bfp_radius+1:a/2+bfp_radius,:);
BFP_y = BFP_image_y(a/2-bfp_radius+1:a/2+bfp_radius,a/2-bfp_radius+1:a/2+bfp_radius,:);


intensity = 1/3*(sum(sum(basisx(:,:,1)))+sum(sum(basisy(:,:,1))))+1/3*(sum(sum(basisx(:,:,2)))+sum(sum(basisy(:,:,2))))+1/3*(sum(sum(basisx(:,:,3)))+sum(sum(basisy(:,:,3))));
%intensity =1.1077e+10;
basisx = basisx./intensity;   %normaliza basis images
basisy = basisy./intensity;

%2.b reshape the image to (75*75+75*75) by 6 matrix
basis_matrix = zeros(image_size*image_size+image_size*image_size,6);

[m,n]=size(BFP_x);
BFP_matrix = zeros(m*m+m*m,6);

for i = 1:6  
    A = reshape(basisx(:,:,i),image_size*image_size,1);
    B = reshape(basisy(:,:,i),image_size*image_size,1);
    basis_matrix(:,i) = cat(1,A,B);
    
    A1 = reshape(BFP_x(:,:,i),m*m,1);
    B1 = reshape(BFP_y(:,:,i),m*m,1);
    BFP_matrix(:,i) = cat(1,A1,B1);
end

end