
for dataN =[17]
fileFolder = 'E:\Experimental_data\20220530 amyloid fibril\';
SMLMName = ['_',num2str(dataN),'\_',num2str(dataN),'_MMStack_Default.ome.tif'];
load(strcat('E:\Experimental_data\20220530 amyloid fibril\','\processed data8 data9_16\saved_beads_loc_for_tform\tformx2y_y_center_505_200_FoV_250'));
tform1 = tformx2y; tform_center1 = [505,200];
load(strcat('E:\Experimental_data\20220530 amyloid fibril\','\processed data8 data9_16\saved_beads_loc_for_tform\tformx2y_y_center_305_200_FoV_250'));
tform2 = tformx2y; tform_center2 = [305,200];


ROI_centerY = [410,200];
W = 1748/2;
ROI_centerX = transformPointsInverse(tformx2y,[W,0]+[-ROI_centerY(1),ROI_centerY(2)])+[W,0];

Nimg = 1000;
FoV = [420,220];  %[x,y]
N_FoV = [8,4]; %[x,y]
FoV_each = 80;

center_x = FoV(1)/N_FoV(1)/2*[-N_FoV(1)+1:2:N_FoV(1)-1];
center_y = FoV(2)/N_FoV(2)/2*[-N_FoV(2)+1:2:N_FoV(2)-1];
[center_X,center_Y] = meshgrid(center_x,center_y);
center_X = center_X(:);
center_Y = center_Y(:);


%% check the cropping
SMLM_img1 = 0;
figure();
SMLM_imgR = Tiff([fileFolder,SMLMName],'r');
ROI_centerX_save = [];

for i=1:round(Nimg/10)

    setDirectory(SMLM_imgR,i);
    SMLM_img1 = SMLM_img1+double(SMLM_imgR.read);
end
imagesc(SMLM_img1); axis image; hold on;
SMLM_img2 = nan(size(SMLM_img1));
for ii = 1:length(center_X)

%count = count+1;
range = round(-(FoV_each-1)/2)+1:1:round((FoV_each-1)/2);

ROI_centerY_cur = round(ROI_centerY+[center_X(ii),center_Y(ii)]);
ROI_Y_all_cur = [ROI_centerY_cur(1)+range.',ROI_centerY_cur(2)+range.'];
distance1 = sqrt(sum((ROI_centerY_cur-tform_center1).^2));
distance2 = sqrt(sum((ROI_centerY_cur-tform_center2).^2));
if distance1<distance2
    tformx2y=tform1;
else
    tformx2y=tform2;
end
ROI_centerX_cur = round(transformPointsInverse(tformx2y,[W,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[W,0]);
ROI_X_all_cur = (transformPointsInverse(tformx2y,[W,0]+[-ROI_Y_all_cur(:,1),ROI_Y_all_cur(:,2)])+[W,0]);
[ROI_X_all_curX,ROI_X_all_curY] = meshgrid(ROI_X_all_cur(end:-1:1,1),ROI_X_all_cur(:,2));
ROI_X_all_cur_intX = floor(ROI_X_all_curX);
ROI_X_all_cur_intY = floor(ROI_X_all_curY);

SMLM_img_ROIy = SMLM_img1(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range);
SMLM_img_ROIx = SMLM_img1(round(ROI_X_all_cur(:,2)),round(ROI_X_all_cur(end:-1:1,1)));

ROI_centerX_save = [ROI_centerX_save;ROI_centerX_cur];
%SMLM_img = [SMLM_img_ROIx,fliplr(SMLM_img_ROIy)];
SMLM_img_ROIy(1,:)=nan; SMLM_img_ROIy(:,end)=nan;
SMLM_img_ROIy(end,:)=nan; SMLM_img_ROIy(:,1)=nan;
SMLM_img2(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range) = (SMLM_img_ROIy)+fliplr(SMLM_img_ROIx);

%figure(); imagesc(SMLM_img); axis image;
rectangle('Position',[ROI_centerY_cur(1)-FoV(1)/N_FoV(1)/2,ROI_centerY_cur(2)-FoV(2)/N_FoV(2)/2,FoV(1)/N_FoV(1),FoV(2)/N_FoV(2)],'EdgeColor','r');
rectangle('Position',[ROI_centerX_cur(1)-FoV(1)/N_FoV(1)/2,ROI_centerX_cur(2)-FoV(2)/N_FoV(2)/2,FoV(1)/N_FoV(1),FoV(2)/N_FoV(2)],'EdgeColor','r');

end

SMLM_img2 = fliplr(SMLM_img2);
SMLM_img2 = SMLM_img2(79:323,1115:1562);
figure(); imagesc(SMLM_img2); axis image
caxis([min(SMLM_img2,[],'all'),max(SMLM_img2,[],'all')/1.2]);
%% cropping

for ii = 1:length(center_X)
%if ii==6
%count = count+1;
range = round(-(FoV_each-1)/2)+1:1:round((FoV_each-1)/2);
SMLM_save_Nmae = ['processed data8 data9_16\data',num2str(dataN),'_centerY_y',num2str(ROI_centerY(1)),'_x_',num2str(ROI_centerY(2)),'_','FoV',num2str(FoV(1)),'_',num2str(FoV(2)),'_',num2str(ii),'th_FoV','.tif'];

ROI_centerY_cur = round(ROI_centerY+[center_X(ii),center_Y(ii)]);
ROI_Y_all_cur = [ROI_centerY_cur(1)+range.',ROI_centerY_cur(2)+range.'];
distance1 = sqrt(sum((ROI_centerY_cur-tform_center1).^2));
distance2 = sqrt(sum((ROI_centerY_cur-tform_center2).^2));
if distance1<distance2
    tformx2y=tform1;
else
    tformx2y=tform2;
end

ROI_centerX_cur = round(transformPointsInverse(tformx2y,[W,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[W,0]);
ROI_X_all_cur = (transformPointsInverse(tformx2y,[W,0]+[-ROI_Y_all_cur(:,1),ROI_Y_all_cur(:,2)])+[W,0]);
[ROI_X_all_curX,ROI_X_all_curY] = meshgrid(ROI_X_all_cur(end:-1:1,1),ROI_X_all_cur(:,2));



SMLM_imgR = Tiff([fileFolder,SMLMName],'r');

%shift = [0,-2]; %[x,y]
for i=1:Nimg

    setDirectory(SMLM_imgR,i);
    SMLM_img1 = double(SMLM_imgR.read);
%     SMLM_img_ROIy = uint16(SMLM_img1(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range));
%     SMLM_img_ROIx = uint16(SMLM_img1(ROI_centerX_cur(2)+range,ROI_centerX_cur(1)+range));

SMLM_img_ROIy = SMLM_img1(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range);
SMLM_img_ROIx = SMLM_img1(round(ROI_X_all_cur(:,2)),round(ROI_X_all_cur(end:-1:1,1)));

SMLM_img = [uint16(SMLM_img_ROIx),uint16(fliplr(SMLM_img_ROIy))];
    


    if i==1
     imwrite(SMLM_img,[fileFolder,SMLM_save_Nmae])
    else
    imwrite(SMLM_img,[fileFolder,SMLM_save_Nmae],'WriteMode','append')
    end
end
end

end
%end
