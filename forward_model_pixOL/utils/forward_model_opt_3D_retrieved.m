function imgPara = forward_model_opt_3D_retrieved(pmask, image_size,NFP,z_range_phy,pixel_size_xy,pixel_size_z,retrieved_mask)

maskName = fullfile('phasemask', pmask);
emitter_wavelength = 610; %nm
refractiveIndx=1.515;% objective refractive index
sampleRefractiveIndx=1.314;% sample refractive index

left_to_right_trans_ratio = 1.114; %transmit_ratio_L2R; for Di03-R488/561,523/610 bandpass; y channel.x channel
%1.5350;%transmit_ratio_L2R; for Di03-R488/561, 593/46; y channel.x channel
zerothOrder = [0,0];%zerothOrder_RL;
%construct phasemaskpara
phasemaskpara.zeroorder = zerothOrder;
phasemaskpara.maskname = maskName;
phasemaskpara.calibrated_mask = retrieved_mask;

imgPara.pix_sizex = pixel_size_xy; %the griding unit for RoSEO3D, in unit of nm
imgPara.pix_sizez = pixel_size_z;
imgPara.axial_grid_points = [(floor(z_range_phy(1)./pixel_size_z)-1):1:(ceil(z_range_phy(2)./pixel_size_z)+1)]*pixel_size_z;
imgPara.number_axial_pixels = length(imgPara.axial_grid_points);


%
n1 = Nanoscope_beads('imageSize', image_size*3+4,...
    'ADcount', 1,...
    'emissWavelength', emitter_wavelength, ...
    'refractiveIndx',refractiveIndx,...
    'sampleRefractiveIndx',sampleRefractiveIndx,...
    'phasemaskpara', phasemaskpara);
% create PSF matrix accounting for channel transmission ratio

%
[PSFx, PSFy] = n1.createPSFstruct3D(n1,...
    'ytoxchanneltransratio', left_to_right_trans_ratio,...
    'normal_focal_plane',NFP,...
    'axial_grid_points',imgPara.axial_grid_points); %distance is is nm

% build the image para structure
Bx = cat(4,PSFx.XXx,PSFx.YYx,PSFx.ZZx,PSFx.XYx,PSFx.XZx,PSFx.YZx,...
     PSFx.XXxdx,PSFx.YYxdx,PSFx.ZZxdx,...
     PSFx.XXxdy,PSFx.YYxdy,PSFx.ZZxdy,...
     PSFx.XXxdz,PSFx.YYxdz,PSFx.ZZxdz);
By = cat(4,PSFy.XXy,PSFy.YYy,PSFy.ZZy,PSFy.XYy,PSFy.XZy,PSFy.YZy,...
     PSFy.XXydx,PSFy.YYydx,PSFy.ZZydx,...
     PSFy.XXydy,PSFy.YYydy,PSFy.ZZydy,...
     PSFy.XXydz,PSFy.YYydz,PSFy.ZZydz);

imgPara.img_size = image_size;
imgPara.Bx = Bx;
imgPara.By = By;
end