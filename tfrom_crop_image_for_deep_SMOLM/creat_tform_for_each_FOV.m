%caculate tform_x2y; means y_real_loc = tfrom(give x channel), or x_real_loc = tfrom_inverse(give y channel)

fileFolder = "E:\Experimental_data\20220530 amyloid fibril\processed data8 data9_16\saved_beads_loc_for_tform_v3\";
mainDirContents = dir(fileFolder);
mask = endsWith({mainDirContents.name},'tform');
mainDirContents(~mask)=[];

W = 1748/2;
%% combine all data into the save matrix
dataX1 = []; frameX = 0;
dataY1 = []; frameY = 0;
mainDirContents_name = mainDirContents.name;
for ii = 1:sum(mask)
this_folder = fullfile(fileFolder , mainDirContents(ii).name);
mainDirContents_cur = dir(this_folder);
mask_cur = endsWith({mainDirContents_cur.name},'ch.csv');
mainDirContents_cur(~mask_cur)=[];

for xch_idx = 1:sum(mask_cur)
fileName = fullfile(fileFolder , mainDirContents(ii).name, mainDirContents_cur(xch_idx).name);
if endsWith(fileName,'xch.csv')
    data_cur = readtable(fileName);
    data_cur = data_cur{:,:};
    data_cur(:,1) = data_cur(:,1)+frameX;
    dataX1 = [dataX1;data_cur];
    frameX = dataX1(end,1);
else if endsWith(fileName,'ych.csv')
     
    data_cur = readtable(fileName);
    data_cur = data_cur{:,:};
    data_cur(:,1) = data_cur(:,1)+frameY;
    dataY1 = [dataY1;data_cur];
    frameY = dataY1(end,1);
end
end
end
end
dataY1(:,2) = W-dataY1(:,2);

figure();
scatter(dataX1(:,2),dataX1(:,3),5,'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
scatter(W+dataY1(:,2),dataY1(:,3),5,'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
axis image; axis ij;
%% create tform
%prepare data


pixel_sz = 1;
photonThred = 300; % in photon for y channel
frame_thred = 2000;
ratio_y2x = 1;


ROI = 150;
FoV_each = ROI;

ROI_centerY = [410,200];
W = 1748/2;
FoV = [420,220];  %[x,y]
N_FoV = [8,4]; %[x,y]


center_x = FoV(1)/N_FoV(1)/2*[-N_FoV(1)+1:2:N_FoV(1)-1];
center_y = FoV(2)/N_FoV(2)/2*[-N_FoV(2)+1:2:N_FoV(2)-1];
[center_X,center_Y] = meshgrid(center_x,center_y);
center_X = center_X(:);
center_Y = center_Y(:);


for ii = 1:length(center_X)
    dataY = dataY1;
dataX = dataX1;

center = round(ROI_centerY+[center_X(ii),center_Y(ii)]);
center(1) = W-center(1);

%filter data and process data
%filter dim estimations
dataX(:,5) = dataX(:,5)*ratio_y2x;
dataX(dataX(:,5)<photonThred,:)=[];
dataY(dataY(:,5)<photonThred,:)=[];
dataX(dataX(:,1)>frame_thred,:)=[];
dataY(dataY(:,1)>frame_thred,:)=[];

%flip y

%shrink ROI
dataX(dataX(:,2)<(center(1)-round(ROI/2)) | dataX(:,2)>(center(1)+round(ROI/2)),:)=[];
dataX(dataX(:,3)<(center(2)-round(ROI/2)) | dataX(:,3)>(center(2)+round(ROI/2)),:)=[];

dataY(dataY(:,2)<(center(1)-round(ROI/2)) | dataY(:,2)>(center(1)+round(ROI/2)),:)=[];
dataY(dataY(:,3)<(center(2)-round(ROI/2)) | dataY(:,3)>(center(2)+round(ROI/2)),:)=[];

% figure();
% scatter(dataX(:,2),dataX(:,3),2,'filled'); hold on;
% scatter(W+dataY(:,2),dataY(:,3),2,'filled');
% axis image

%% filter out closely closed emitters
thred = 5; frameThred = 3;
dataX = filterDenseEmitter(dataX,thred,frameThred);
dataY = filterDenseEmitter(dataY,thred,frameThred);
% figure();
% scatter(dataX(:,2),dataX(:,3),2,'filled'); hold on;
% scatter(W+dataY(:,2),dataY(:,3),2,'filled');
% axis image


%% pair the data with optimization
% assume the only geometry difference is xy translation
initial_guess_y2x = mean(dataX(:,[2,3]),1)-mean(dataY(:,[2,3]),1); % x_channel = y_channel+initial_guess
loss = lossCaculate(initial_guess_y2x,dataX,dataY);
lossF = @(tform)lossCaculate(tform,dataX,dataY);
tform_y2x = fmincon(lossF,initial_guess_y2x);

%% extract the paired data
load('tform3.mat');
[dataX_paired,dataY_paired] = pairedData(tformx2y,tform_y2x,dataX,dataY);
loss = lossCaculate(tform_y2x,dataX,dataY);
% [dataX_paired,dataY_paired] = pairedData(tform_y2x,dataX,dataY);

% figure(); 
% scatter(dataX_paired(:,2),dataX_paired(:,3),10,'filled','r'); axis image
% hold on;
% scatter(dataY_paired(:,2)+tform_y2x(1),dataY_paired(:,3)+tform_y2x(2),10,'filled','b');axis image
% 
% figure();
% distance = sqrt((dataX_paired(:,2)-dataY_paired(:,2)-tform_y2x(1)).^2+(dataX_paired(:,3)-dataY_paired(:,3)-tform_y2x(2)).^2);
% subplot(1,2,1);
% histogram(distance); xlabel('distance');
% subplot(1,2,2);
% scatter(dataX_paired(:,2),distance,10,'filled');
% xlabel('x in X channel(pixel)'); ylabel('distance');

%plot_pairs(dataX_paired,dataY_paired,[], [],tform_y2x); axis image;

%% calculate the tform using polynominal fitting
fixedPoints = dataY_paired(:,[2,3]);
movingPoints = dataX_paired(:,[2,3]);
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
%----filter out umathed pairs
distance = sqrt((dataX_paired(:,2)-dataX_inver(:,1)).^2+(dataX_paired(:,3)-dataX_inver(:,2)).^2);
indx = distance>2;
fixedPoints(indx,:) = [];
movingPoints(indx,:) = [];
%----filter out umathed pairs v2
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
%----filter out umathed pairs
distance = sqrt((movingPoints(:,1)-dataX_inver(:,1)).^2+(movingPoints(:,2)-dataX_inver(:,2)).^2);
indx = distance>0.5;
fixedPoints(indx,:) = [];
movingPoints(indx,:) = [];
%%
%tformx2y = images.geotrans.PiecewiseLinearTransformation2D(movingPoints,fixedPoints);
%dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
distance = sqrt((movingPoints(:,1)-dataX_inver(:,1)).^2+(movingPoints(:,2)-dataX_inver(:,2)).^2);
%figure(); histogram(distance);
%----

figure(); 
scatter(movingPoints(:,1),movingPoints(:,2),10,'filled','r'); axis image
hold on;
scatter(dataX_inver(:,1),dataX_inver(:,2),10,'filled','g');axis image
for ii = 1:size(movingPoints,1)
plot([movingPoints(ii,1),dataX_inver(ii,1)],[movingPoints(ii,2),dataX_inver(ii,2)],'k-'); 
end
box on;

figure(); 
tformxy_linear = [median(movingPoints(:,1)-fixedPoints(:,1)),median(movingPoints(:,2)-fixedPoints(:,2))];
scatter(movingPoints(:,1),movingPoints(:,2),10,'filled','r'); axis image
hold on;
scatter(fixedPoints(:,1)+tformxy_linear(1),fixedPoints(:,2)+tformxy_linear(2),10,'filled','g');axis image
for ii = 1:size(movingPoints,1)
plot([movingPoints(ii,1),fixedPoints(ii,1)+tformxy_linear(1)],[movingPoints(ii,2),fixedPoints(ii,2)+tformxy_linear(2)],'k-'); 
end

xlabel('x (pixels)');
ylabel('y (pixels)'); legend('x-channel beads','registed y-channel beads');

save(strcat(fileFolder,'tformx2y_y_center_',num2str(W-center(1)),'_',num2str(center(2)),'_FoV_',num2str(ROI),'.mat'),'tformx2y');
end
%% function
function loss = lossCaculate(tform_y2x,dataX,dataY)
x_dataX = dataX(:,2);
y_dataX = dataX(:,3);
I_dataX = dataX(:,5);



x_dataY = dataY(:,2)+tform_y2x(1);
y_dataY = dataY(:,3)+tform_y2x(2);
I_dataY = dataY(:,5);

loss = 0;
count = 0;
for ii = 1:dataX(end,1)
    x_dataX_cur = x_dataX(dataX(:,1)==ii);
    y_dataX_cur = y_dataX(dataX(:,1)==ii);
    I_dataX_cur = I_dataX(dataX(:,1)==ii);
    x_dataY_cur = x_dataY(dataY(:,1)==ii);
    y_dataY_cur = y_dataY(dataY(:,1)==ii);
    I_dataY_cur = I_dataY(dataY(:,1)==ii);

    if length(x_dataX_cur)<length(x_dataY_cur)
        dim = 2;
    else 
        dim=1;
    end
    distance = sqrt((x_dataX_cur-x_dataY_cur.').^2+(y_dataX_cur-y_dataY_cur.').^2);
    I_distance = abs(I_dataX_cur-I_dataY_cur.')./repmat(I_dataX_cur,1,length(I_dataY_cur));
    %distance(I_distance>0.3)=nan;
    %distance(distance>10*min(distance,[],'all')) = nan;
    loss = loss+nansum(nanmin(distance,[],dim));
    count = count+nansum(nanmin(distance,[],dim)>0);

end

loss = loss/count;
end


function [dataX_paired,dataY_paired] = pairedData(tformx2y,tform_y2x,dataX,dataY)
x_dataX = dataX(:,2);
y_dataX = dataX(:,3);
I_dataX = dataX(:,5);

dataX_inver = transformPointsInverse(tformx2y,dataY(:,[2,3]));
x_dataY = dataX_inver(:,1);
y_dataY = dataX_inver(:,2);
I_dataY = dataY(:,5);
% x_dataY = dataY(:,2)+tform_y2x(1);
% y_dataY = dataY(:,3)+tform_y2x(2);
% I_dataY = dataY(:,5);

loss = 0;
dataX_paired = [];
dataY_paired = [];
for ii = 1:dataX(end,1)

    if ii==35
aaa = 1;
    end
    x_dataX_cur = x_dataX(dataX(:,1)==ii);
    y_dataX_cur = y_dataX(dataX(:,1)==ii);
    I_dataX_cur = I_dataX(dataX(:,1)==ii);
    dataX_cur = dataX(dataX(:,1)==ii,:);
    x_dataY_cur = x_dataY(dataY(:,1)==ii);
    y_dataY_cur = y_dataY(dataY(:,1)==ii);
    I_dataY_cur = I_dataY(dataY(:,1)==ii);
    dataY_cur = dataY(dataY(:,1)==ii,:);
    if length(x_dataX_cur)<length(x_dataY_cur)
        dim = 2;
    else 
        dim=1;
    end
    distance = sqrt((x_dataX_cur-x_dataY_cur.').^2+(y_dataX_cur-y_dataY_cur.').^2);
    I_distance = abs(I_dataX_cur-I_dataY_cur.')./repmat(I_dataX_cur,1,length(I_dataY_cur));
    %distance(I_distance>0.3)=nan;
    distance(distance>10) = nan;
    

    [~,pair_indx] = nanmin(distance,[],dim);
    pair_indx(isnan(nanmin(distance,[],dim)))=nan;
    [GC,GR] = groupcounts(pair_indx(:));
    indx = GC>1;
    %distance(isnan(nanmin(distance,[],dim)),:)=nan;

    for kk=1:length(GC) 
        if GC(kk)>1 & ~isnan(GR(kk))
            if dim ==2
                minValue = min(distance(pair_indx==GR(kk),GR(kk)));
                distance(distance(:,GR(kk))~=minValue,GR(kk))=nan;
            else 
                minValue = min(distance(GR(kk),pair_indx==GR(kk)));
                distance(GR(kk),distance(GR(kk),:)~=minValue)=nan;
            end
        end
    end
        [~,pair_indx] = nanmin(distance,[],dim);
        pair_indx(isnan(nanmin(distance,[],dim)))=[];
        if nansum(nanmin(distance,[],dim)>30)>0
        tt=1;
            end
    if length(x_dataX_cur)<length(x_dataY_cur)  
        dataX_cur_temp = dataX_cur;
        dataX_cur(isnan(nanmin(distance,[],dim)),:)=[];
        dataX_paired = [dataX_paired;dataX_cur];
        dataY_paired = [dataY_paired;dataY_cur(pair_indx,:)];
        if rem(ii,200)==10
        %plot_pairs(dataX_cur,dataY_cur(pair_indx,:),dataX_cur_temp,dataY_cur,tform_y2x);
        %plot_pairs(dataX_paired,dataY_paired,dataX_cur_temp,dataY_cur,tform_y2x);
        end
        
    else 
        dataY_cur_temp = dataY_cur;
        dataY_cur(isnan(nanmin(distance,[],dim)),:)=[];
        dataX_paired = [dataX_paired;dataX_cur(pair_indx,:)];
        dataY_paired = [dataY_paired;dataY_cur];
        if rem(ii,200)==10
            %plot_pairs(dataX_paired,dataY_paired,dataX_cur,dataY_cur_temp,tform_y2x);
            %plot_pairs(dataX_cur(pair_indx,:),dataY_cur,dataX_cur,dataY_cur_temp,tform_y2x);
        end
        
    end

end

end

function plot_pairs(dataX,dataY,dataX_orig, dataY_orig,offset)

figure();
for ii = 1:size(dataX,1)    
hold on;
plot([dataX(ii,2),dataY(ii,2)+offset(1)],[dataX(ii,3),dataY(ii,3)+offset(2)],'k-'); 

end
scatter(dataX(:,2),dataX(:,3),10,"red","filled");
scatter(dataY(:,2)+offset(1),dataY(:,3)+offset(2),10,"blue","filled");
if length(dataX_orig)>1
scatter(dataX_orig(:,2),dataX_orig(:,3),10,"red*");
scatter(dataY_orig(:,2)+offset(1),dataY_orig(:,3)+offset(2),10,"blue*");
end
% xlim([300,1000]);
% ylim([0,400]);
end


function datafiltered = filterDenseEmitter(data,thred,frameThred)

datafiltered = [];

for framei = 1:frameThred:max(data(:,1))-frameThred+1
    idx = data(:,1) >= framei & data(:,1) <= framei+frameThred-1;
    data_cur = data(idx,:);
    x_cur = data(idx,2);
    y_cur = data(idx,3);
    distance = sqrt((x_cur-x_cur.').^2+(y_cur-y_cur.').^2);
    indx_close = distance<thred;
    %indx_close(distance==distance)=0;
    data_cur(sum(indx_close,2)>1,:)=[];

    datafiltered = [datafiltered;data_cur];
    
end

end