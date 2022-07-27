function basis_matrix_opt = forward_model_opt_retrieved(pmask, image_size,pmask_retrieve_name)
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
[basis_matrix_opt,mask_opt] = basis_matrix_pixel_based_v2_retrived_opt(Microscopy);
end