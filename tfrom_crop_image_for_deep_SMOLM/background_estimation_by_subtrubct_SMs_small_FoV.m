ID = [20:24];

for kk = 1:length(ID)
    kk
fileFolder = ['E:\Experimental_data\20220429 A1-LCD\'];
SMLMName = ['processed data\data' num2str(ID(kk)) '_centerY_y466_x_327_FoV101_101_1th_FoV.tif'];
SMLM_offsetName = [fileFolder,'\processed data\offset.mat'];
ROI_centerY = [466,327]; 
FoV = [101,101]; 
load([fileFolder,'\processed data\offset_centerY_y466_x_327_FoV101_101_1th_FoV.mat']);

Nimg = 2000;


SM_img_offset = offset;
SMLMR = Tiff([fileFolder,SMLMName],'r');
for i=1:Nimg
    setDirectory(SMLMR,i);
    SM_img(:,:,i) = double(SMLMR.read);

end
imgSzx = size(SM_img,2)/2;
imgSzy = size(SM_img,1);


SM_img = SM_img-SM_img_offset;
%%
fileName1 = [[fileFolder,'processed data\BKG_list_data20_24_centerY_y466_x327\data'],  num2str(ID(kk)) '_xch.csv'];
data = readtable(fileName1);

%
x_X = data.x_nm_./1;
y_X = data.y_nm_./1;
signal_X = data.intensity_photon_;
frameN_X = data.frame;
%figure(); scatter(x,y); axis image
%%

fileName1 = [[fileFolder,'processed data\BKG_list_data20_24_centerY_y466_x327\data'],  num2str(ID(kk)) '_ych.csv'];
data = readtable(fileName1);

%
x_Y = data.x_nm_./1;
y_Y = data.y_nm_./1;
signal_Y = data.intensity_photon_;
frameN_Y = data.frame;
%%
SM_img_subtractX = SM_img(:,(1:imgSzx),:);
SM_img_subtractY = SM_img(:,(1:imgSzx)+imgSzx,:); 
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
center_size = 10;
range = round([-center_size/2,center_size/2]);
for i=1:50:Nimg
    count = count+1;
    indx_start = max(1,i-50);
    indx_end = min(i+50,Nimg);
    back_cur = nanmean(SM_img_subtractX(:,:,indx_start:indx_end),3);
    back_cur(back_cur<0)=0;
    back_cur(isnan(back_cur))=nanmean(back_cur(round(H/2)+range,round(W/2)+range),'all');
    SMLM_bkgX(:,:,count) = imgaussfilt(back_cur(:,:,:),5);
    

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
    SMLM_bkgY(:,:,count) = imgaussfilt(back_cur(:,:,:),5);

end
%clear SM_img_subtractY
SMLM_bkg = [SMLM_bkgX,SMLM_bkgY];
%% crop FoV; copy from the the crop_save_image.m code

SMLM_save_Nmae = ['processed data\data',num2str(ID(kk)),'_bkg_centerY_y',num2str(ROI_centerY(1)),'_x_',num2str(ROI_centerY(2)),'_','FoV',num2str(FoV(1)),'_',num2str(FoV(2)),'_1th_FoV','.mat'];
save([fileFolder,SMLM_save_Nmae],'SMLM_bkg')


end

function SM_offset = crop_offset(ROI_centerY,FoV,fileFolder,offset)
load([fileFolder,'processed data\saved_beads_loc_for_tform\tformx2y_y_center_375_178_FoV_150.mat']);

W = 1748/2;
ROI_centerX = transformPointsInverse(tformx2y,[W,0]+[-ROI_centerY(1),ROI_centerY(2)])+[W,0];
N_FoV = [1,1];
FoV_each = 101;
%------------

center_x = FoV(1)/N_FoV(1)/2*[-N_FoV(1)+1:2:N_FoV(1)-1];
center_y = FoV(2)/N_FoV(2)/2*[-N_FoV(2)+1:2:N_FoV(2)-1];
count = 0;

for ii = 1:length(center_x)
for jj = 1:length(center_y)

count = count+1;
range = round(-(FoV_each-1)/2):1:round((FoV_each-1)/2);

ROI_centerY_cur = round(ROI_centerY+[center_x(ii),center_y(jj)]);
ROI_centerX_cur = round(transformPointsInverse(tformx2y,[1024,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[1024,0]);


    offset_ROIy = offset(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range);
    offset_ROIx = offset(ROI_centerX_cur(2)+range,ROI_centerX_cur(1)+range);
    SM_offset = [offset_ROIx,fliplr(offset_ROIy)];


end
end
end