
dataN =2;
fileFolder = 'E:\Experimental_data\20220528 amyloid fibril\';
SMLMName = ['_',num2str(dataN),'\_',num2str(dataN),'_MMStack_Default.ome.tif'];


Nimg = 50;

%
ROI_centery = [92,343]; 
ROI_centerx = [76,1424]; 
D = 41;
R = (D-1)/2;
load([fileFolder,'processed data\offset.mat']);
offset_ROIy = double(offset(ROI_centery(1)-R:ROI_centery(1)+R,ROI_centery(2)-R:ROI_centery(2)+R));
offset_ROIx = double(offset(ROI_centerx(1)-R:ROI_centerx(1)+R,ROI_centerx(2)-R:ROI_centerx(2)+R));
%

SMLM_imgR = Tiff([fileFolder,SMLMName],'r');
for i=1:Nimg

    setDirectory(SMLM_imgR,i);
    SMLM_img = double(SMLM_imgR.read);
    SMLM_img_ROIy = double(SMLM_img(ROI_centery(1)-R:ROI_centery(1)+R,ROI_centery(2)-R:ROI_centery(2)+R))-offset_ROIy;
    SMLM_img_ROIx = double(SMLM_img(ROI_centerx(1)-R:ROI_centerx(1)+R,ROI_centerx(2)-R:ROI_centerx(2)+R))-offset_ROIx;
    SMLM_img_save(:,:,i) = [SMLM_img_ROIx,SMLM_img_ROIy];
    
end
save([fileFolder,'saved_beads_for_in_focus_pixOL_retrieval\',num2str(dataN),'_beads1_L',num2str(ROI_centery(1)),'_',num2str(ROI_centery(2)),...
    '_R_',num2str(ROI_centerx(1)),'_',num2str(ROI_centerx(2)),...
     '_wo_offset_unfliped.mat'],"SMLM_img_save")

figure(); imagesc(SMLM_img_save(:,:,1)); axis image; colorbar;
%% concatenate all beads together
%go to folder that saved your beads data
%beads_img = [];
beads_img = cat(3,beads_img,SMLM_img_save);

%save([fileFolder,'saved_beads_for_in_focus_pixOL_retrieval\','combined_beads_20220429.mat'],"beads_img");

