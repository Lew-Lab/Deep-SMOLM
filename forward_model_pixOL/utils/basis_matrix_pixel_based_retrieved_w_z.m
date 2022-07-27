function [basis_matrix] = basis_matrix_pixel_based_retrieved_w_z(Microscopy)


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


zf = Microscopy.z;
z2 = Microscopy.z2;
xy_ind = Microscopy.xy_ind;



%2.a generate the basis image
[basisImagex,basisImagey,BFP_image_x,BFP_image_y,basisImagex_dx,basisImagey_dx,basisImagex_dy,basisImagey_dy] = simDipole_v5_retrieved_w_z(zf,z2,0,sampling_size,wavelength,n1,n2,nh,NA,Magnitude,pix_size,xy_ind,Microscopy.pmask_retrieve_name);
%intensity = sum(sum(sum(basisImagex+basisImagey)))/3;
basisx = basisImagex(sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,:);
basisy = basisImagey(sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,sampling_size/2-image_size/2+1:sampling_size/2+image_size/2,:);

[a,b]=size(BFP_image_x);

BFP_x = BFP_image_x(a/2-bfp_radius+1:a/2+bfp_radius,a/2-bfp_radius+1:a/2+bfp_radius,:);
BFP_y = BFP_image_y(a/2-bfp_radius+1:a/2+bfp_radius,a/2-bfp_radius+1:a/2+bfp_radius,:);


intensity = 1/3*(sum(sum(basisx(:,:,1)))+sum(sum(basisy(:,:,1))))+1/3*(sum(sum(basisx(:,:,2)))+sum(sum(basisy(:,:,2))))+1/3*(sum(sum(basisx(:,:,3)))+sum(sum(basisy(:,:,3))));
%intensity =1.1077e+10;
basisx = basisx./intensity;   %normaliza basis images
basisy = basisy./intensity*Microscopy.y2xratio;

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


