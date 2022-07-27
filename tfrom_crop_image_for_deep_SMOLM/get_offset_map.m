%fileName = 'F:\data\20210410 3D lipid+ATTO647N\';
fileName = 'E:\Experimental_data\20220528 amyloid fibril\';
ID = 7;
offsetName = [fileName,'_',num2str(ID),'\_',num2str(ID),'_MMStack_Default.ome.tif'];
Nimg = 100;

offsetR = Tiff(offsetName,'r');
for i=1:Nimg
    setDirectory(offsetR,i);
    offset(:,:,i) = double(offsetR.read);

end
offset_mean = mean(offset,3);
offset = offset_mean;
save([fileName 'processed data\offSet_for_A1_LCD.mat'],'offset')
%offset_ROI=offset_mean;imwrite(uint16(offset_ROI),'offset_ROI572_763.tif');


%% same whole offset

SMLM_img = single(offset_mean);
tagstruct.ImageLength = size(SMLM_img,1);
tagstruct.ImageWidth = size(SMLM_img,2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;


%
t = Tiff([fileName 'processed data\offSet.tif'],'w');
t.setTag(tagstruct);
t.write(SMLM_img);
t.close();

%% crop offset and save offset
ROI_centery = [194,414]; 
ROI_centerx = [173,1529]; 
D = 201;
R = (D-1)/2;
SMLM_img_ROIy = (SMLM_img(ROI_centery(1)-R:ROI_centery(1)+R,ROI_centery(2)-R:ROI_centery(2)+R));
SMLM_img_ROIx = (SMLM_img(ROI_centerx(1)-R:ROI_centerx(1)+R,ROI_centerx(2)-R:ROI_centerx(2)+R));
SMLM_img_ROI = single([SMLM_img_ROIx,fliplr(SMLM_img_ROIy)]);

tagstruct.ImageLength = size(SMLM_img_ROI,1);
tagstruct.ImageWidth = size(SMLM_img_ROI,2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%
t = Tiff(['F:\data\20210813 Ddx4\offset_subtracted\FOV_x173_1529_y194_414\offSet_ROI.tif'],'w');
t.setTag(tagstruct);
t.write(SMLM_img_ROI);
t.close();
