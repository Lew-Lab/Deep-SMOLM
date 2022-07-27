function basis_matrix_opt_set = forward_model_opt_3D_retrieved_w_z(pmask, image_size,pmask_retrieve_name,zf_set)
Microscopy=struct();
Microscopy.wavelength = 610e-9; 
Microscopy.n1 = 1.518;
Microscopy.n2 = 1.334;
Microscopy.nh = 1.518;
Microscopy.NA = 1.4;
Microscopy.pixelSizeUpsampling = 1;
Microscopy.upsampling = 1;
Microscopy.pix_size=6500/Microscopy.pixelSizeUpsampling;
Microscopy.bfp_radius = 80*Microscopy.upsampling;
Microscopy.Magnitude = 111.1111;
Microscopy.sampling_size = 597;%round(1.541e5*Microscopy.wavelength*Microscopy.Magnitude*Microscopy.bfp_radius/Microscopy.NA)*Microscopy.pixelSizeUpsampling-1;

Microscopy.image_size =image_size*Microscopy.pixelSizeUpsampling;
%
Microscopy.mask=pmask; Microscopy.rot=0;
Microscopy.pmask_retrieve_name=pmask_retrieve_name;
Microscopy.y2xratio=1.145;

for ii = 1:length(zf_set)
Microscopy.z = zf_set(ii);  % focal length
Microscopy.z2 = 0; %SM's Z position
Microscopy.zh = 0;
Microscopy.xy_ind = 0;
[basis_matrix_opt] = basis_matrix_pixel_based_retrieved_w_z(Microscopy);
basis_matrix_opt_set(:,:,ii) = basis_matrix_opt;
end
end


