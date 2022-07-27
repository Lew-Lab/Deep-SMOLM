ID = [14:16];

for kk = 1:length(ID)
    kk
fileFolder = ['E:\Experimental_data\20220530 amyloid fibril\'];
SMLMName = ['_',num2str(ID(kk)),'\_',num2str(ID(kk)),'_MMStack_Default.ome.tif'];
SMLM_offsetName = [fileFolder,'\processed data\offSet.mat'];
Nimg = 1000;

load(SMLM_offsetName);
SMLMR = Tiff([fileFolder,SMLMName],'r');
for i=1:Nimg
    setDirectory(SMLMR,i);
    SM_img(:,:,i) = double(SMLMR.read);

end
imgSzx = size(SM_img,2)/2;
imgSzy = size(SM_img,1);

SM_img_offset =offset;
SM_img = SM_img-SM_img_offset;
%%
fileName1 = [[fileFolder,'processed data\BKG_list_data9_16\data'],  num2str(ID(kk)) '_xch.csv'];
data = readtable(fileName1);

%
x_X = data.x_nm_./1;
y_X = data.y_nm_./1;
signal_X = data.intensity_photon_;
frameN_X = data.frame;
%figure(); scatter(x,y); axis image
%%

fileName1 = [[fileFolder,'processed data\BKG_list_data9_16\data'],  num2str(ID(kk)) '_ych.csv'];
data = readtable(fileName1);

%
x_Y = data.x_nm_./1;
y_Y = data.y_nm_./1;
signal_Y = data.intensity_photon_;
frameN_Y = data.frame;
%%
SM_img_subtractX = SM_img(:,(1:imgSzx)+imgSzx,:);
SM_img_subtractY = SM_img(:,(1:imgSzx),:); 
%clear SM_img
%%
count = 0;
for i = 1:Nimg
    %figure(); imagesc(SM_img_subtract(:,:,i)); axis image
    for j = 1:sum(frameN_X==i)
        count = count+1;
        rangex = [max(1,-10+round(x_X(count))):min(10+round(x_X(count)),imgSzx)];
        rangey = [max(1,-10+round(y_X(count))):min(10+round(y_X(count)),imgSzy)];
        SM_img_subtractX(rangey,rangex,i) = nan; 
        
    end
    %figure(); imagesc(SM_img_subtractX(:,:,i)); axis image
end
%%
count = 0;
for i = 1:Nimg
    %figure(); imagesc(SM_img_subtract(:,:,i)); axis image
    for j = 1:sum(frameN_Y==i)
        count = count+1;
        rangex = [max(1,-10+round(x_Y(count))):min(10+round(x_Y(count)),imgSzx)];
        rangey = [max(1,-10+round(y_Y(count))):min(10+round(y_Y(count)),imgSzy)];
        SM_img_subtractY(rangey,rangex,i) = nan; 
       
    end
    %figure(); imagesc(SM_img_subtractY(:,:,i)); axis image
end
%% background in X channel
% save bkg frame every 50 frames
count = 0;
[H,W,L]=size(SM_img_subtractY);
center_size = 100;
range = round([-center_size/2,center_size/2]);
for i=1:50:Nimg
    count = count+1;
    indx_start = max(1,i-50);
    indx_end = min(i+50,Nimg);
    back_cur = nanmean(SM_img_subtractX(:,:,indx_start:indx_end),3);
    back_cur(back_cur<0)=0;
    back_cur(isnan(back_cur))=nanmean(back_cur(round(H/2)+range,round(W/2)+range),'all');
    SMLM_bkgX(:,:,count) = imgaussfilt(back_cur(:,:,:),15);
    

end
%clear SM_img_subtractX
%% background in Y channel
count = 0;
for i=1:50:Nimg
    count = count+1;
    indx_start = max(1,i-50);
    indx_end = min(i+50,Nimg);
    back_cur = nanmean(SM_img_subtractY(:,:,indx_start:indx_end),3);
    back_cur(back_cur<0)=0;
    back_cur(isnan(back_cur))=nanmean(back_cur(round(H/2)+range,round(W/2)+range),'all');
    SMLM_bkgY(:,:,count) = imgaussfilt(back_cur(:,:,:),15);

end
%clear SM_img_subtractY
%% crop FoV; copy from the the crop_save_image.m code
%------------ copy from crop_save_image.m code----------

%load(strcat('E:\Experimental_data\20220521 amyloid fibril\','processed data3\saved_beads_loc_for_tform\tformx2y_y_center_400_230_FoV_300.mat'));
%load(strcat('E:\Experimental_data\20220530 beads\','processed data\saved_beads_loc_for_tform\tformx2y_y_center_400_230_FoV_300.mat'));
load(strcat('E:\Experimental_data\20220530 beads\','processed data\saved_beads_loc_for_tform\tformx2y_y_center_410_200_FoV_500.mat'));

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
count = 0;


for ii = 1:length(center_x)
for jj = 1:length(center_y)

count = count+1;



range = round(-(FoV_each-1)/2)+1:1:round((FoV_each-1)/2);
SMLM_save_Nmae = ['processed data\data',num2str(ID(kk)),'_bkg_centerY_y',num2str(ROI_centerY(1)),'_x_',num2str(ROI_centerY(2)),'_','FoV',num2str(FoV(1)),'_',num2str(FoV(2)),'_',num2str(count),'th_FoV','.mat'];

ROI_centerY_cur = round(ROI_centerY+[center_x(ii),center_y(jj)]);
ROI_centerX_cur = round(transformPointsInverse(tformx2y,[W,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[W,0]);



for i=1:size(SMLM_bkgX,3)

    
    SMLM_img = [SMLM_bkgY(:,:,i),SMLM_bkgX(:,:,i)];
    SMLM_img_ROIy = SMLM_img(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range);
    SMLM_img_ROIx = SMLM_img(ROI_centerX_cur(2)+range,ROI_centerX_cur(1)+range);
    SMLM_bkg(:,:,i) = [SMLM_img_ROIx,fliplr(SMLM_img_ROIy)];

end
save([fileFolder,SMLM_save_Nmae],'SMLM_bkg')
end

end

end

%%

% 
% %load([fileName num2str(ID(kk)) '_3Dlipid_bkg_v2.mat']);
% SMLM_bkg2 = SMLM_bkg;
% figure(); imagesc(SMLM_bkg2(1:101,1:101,1)); axis image; title('x channel');
% 
% figure(); imagesc(SMLM_bkg2(1:101,(1:101)+101,1)); axis image; title('y channel');
% 
% %%
% 
% load([fileName num2str(ID(kk)) '_beads_bkg.mat']);
% 
% figure(); imagesc(SMLM_bkg(1:101,1:101,1)); axis image; title('x channel');
% 
% figure(); imagesc(SMLM_bkg(1:101,(1:101)+101,1)); axis image; title('y channel');

%% temp
% center1 = [1581,135]-[1024,0];
% size = 4;
% alpha = 0.2;
% indx = signal>200;
% figure();scatter(x_X(indx),y_X(indx),size,'filled','MarkerEdgeAlpha',alpha,'MarkerFaceAlpha',alpha); axis image;
% xlim([center1(1)-100,center1(1)+100]);ylim([center1(2)-100,center1(2)+100]);set(gca, 'YDir','reverse')
% xticks([]); yticks([]);
% 
% %%
% 
% center1 = [480,157];
% size = 4;
% alpha = 0.2;
% indx = signal_Y>200;
% figure();scatter(x_Y(indx),y_Y(indx),size,'filled','MarkerEdgeAlpha',alpha,'MarkerFaceAlpha',alpha); axis image;
% xlim([center1(1)-100,center1(1)+100]);ylim([center1(2)-100,center1(2)+100]);set(gca, 'YDir','reverse');set(gca, 'XDir','reverse')
% xticks([]); yticks([]);