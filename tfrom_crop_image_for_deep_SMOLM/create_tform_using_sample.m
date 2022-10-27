%% requirement:
% 1) prepare the loc sheet estimated from thunderstorm using polarized
% standard PSF
% the unit of x and y location from thunderstorm is **pixel***
%% read the file name

fileFolder = [pwd '\example_data_for_tform_using_sample\sample_loc_sheet\'];
mainDirContents = dir(fileFolder);
mask = endsWith({mainDirContents.name},'tform');
mainDirContents(~mask)=[];

W = 1748/2;
%% combine all data into the save matrix
dataX = []; frameX = 0;
dataY = []; frameY = 0;
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
    dataX = [dataX;data_cur];
    frameX = dataX(end,1);
else if endsWith(fileName,'ych.csv')
     
    data_cur = readtable(fileName);
    data_cur = data_cur{:,:};
    data_cur(:,1) = data_cur(:,1)+frameY;
    dataY = [dataY;data_cur];
    frameY = dataY(end,1);
end
end
end
end
dataY(:,2) = W-dataY(:,2);

figure();
scatter(dataX(:,2),dataX(:,3),5,'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
scatter(W+dataY(:,2),dataY(:,3),5,'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
axis image; axis ij;
%% define the tform FoV
%prepare data
pixel_sz = 1;
photonThred = 300; % in photon for y channel
frame_thred = 2000;
ratio_y2x = 1;

center = [410,200]; %in pixel; directly get from the y channel image where the sample is located; have not flip the data
center(1) = W-center(1);
ROI = 250;

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

figure();
scatter(dataX(:,2),dataX(:,3),2,'filled'); hold on;
scatter(W+dataY(:,2),dataY(:,3),2,'filled');
axis image

%% filter out closely closed emitters
thred = 5; frameThred = 3;
dataX = filterDenseEmitter(dataX,thred,frameThred);
dataY = filterDenseEmitter(dataY,thred,frameThred);
figure();
scatter(dataX(:,2),dataX(:,3),2,'filled'); hold on;
scatter(W+dataY(:,2),dataY(:,3),2,'filled');
axis image


%% pair the data with optimization (first order tform)

% assume the only geometry difference is xy translation
initial_guess_y2x = mean(dataX(:,[2,3]),1)-mean(dataY(:,[2,3]),1); % x_channel = y_channel+initial_guess
loss = lossCaculate(initial_guess_y2x,dataX,dataY);
lossF = @(tform)lossCaculate(tform,dataX,dataY);
tform_y2x = fmincon(lossF,initial_guess_y2x);

%% pair the data using the first order tfrom and get a initial poly-tform

[dataX_paired,dataY_paired] = pairedData_initial(tform_y2x,dataX,dataY);
tform_y2x = mean(dataX_paired(:,[2,3]),1)-mean(dataY_paired(:,[2,3]),1);
fixedPoints = dataY_paired(:,[2,3]);
movingPoints = dataX_paired(:,[2,3]);
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);

figure(); 
scatter(movingPoints(:,1),movingPoints(:,2),10,'filled','r'); axis image
hold on;
scatter(dataX_inver(:,1),dataX_inver(:,2),10,'filled','g');axis image
for ii = 1:size(movingPoints,1)
plot([movingPoints(ii,1),dataX_inver(ii,1)],[movingPoints(ii,2),dataX_inver(ii,2)],'k-'); 
end
box on;

%% paired data again using the above rought poly-tform

[dataX_paired,dataY_paired] = pairedData(tformx2y,tform_y2x,dataX,dataY);
tform_y2x = mean(dataX_paired(:,[2,3]),1)-mean(dataY_paired(:,[2,3]),1);
loss = lossCaculate(tform_y2x,dataX,dataY);
% [dataX_paired,dataY_paired] = pairedData(tform_y2x,dataX,dataY);

figure(); 
scatter(dataX_paired(:,2),dataX_paired(:,3),10,'filled','r'); axis image
hold on;
scatter(dataY_paired(:,2)+tform_y2x(1),dataY_paired(:,3)+tform_y2x(2),10,'filled','b');axis image

figure();
distance = sqrt((dataX_paired(:,2)-dataY_paired(:,2)-tform_y2x(1)).^2+(dataX_paired(:,3)-dataY_paired(:,3)-tform_y2x(2)).^2);
subplot(1,2,1);
histogram(distance); xlabel('distance');
title('distance when only applying first order tform');
subplot(1,2,2);
scatter(dataX_paired(:,2),distance,10,'filled');
xlabel('x in X channel(pixel)'); ylabel('distance');
title('distance when only applying first order tform');

plot_pairs(dataX_paired,dataY_paired,[], [],tform_y2x); axis image;
title('image match when only apply first order tform')

%% calculate the tform using polynominal fitting
fixedPoints = dataY_paired(:,[2,3]);
movingPoints = dataX_paired(:,[2,3]);
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
%----filter out umathed pairs
distance = sqrt((dataX_paired(:,2)-dataX_inver(:,1)).^2+(dataX_paired(:,3)-dataX_inver(:,2)).^2);
figure(); histogram(distance);  xlabel('distance'); title('distance when applying poly-tform');
%% filter out pairs with large distance and caculate the poly-tform again
indx = distance>2;
fixedPoints(indx,:) = [];
movingPoints(indx,:) = [];
tformx2y = images.geotrans.PolynomialTransformation2D(movingPoints,fixedPoints,4);
dataX_inver = transformPointsInverse(tformx2y,fixedPoints);
distance = sqrt((movingPoints(:,1)-dataX_inver(:,1)).^2+(movingPoints(:,2)-dataX_inver(:,2)).^2);
figure(); histogram(distance);  xlabel('distance'); title('distance when applying final poly-tform');
%----

%% visualize the pair results
figure(); 
scatter(movingPoints(:,1),movingPoints(:,2),10,'filled','r'); axis image
hold on;
scatter(dataX_inver(:,1),dataX_inver(:,2),10,'filled','g');axis image
for ii = 1:size(movingPoints,1)
plot([movingPoints(ii,1),dataX_inver(ii,1)],[movingPoints(ii,2),dataX_inver(ii,2)],'k-'); 
end
box on;
title('paired data using the final poly-tform');

figure(); 
tformxy_linear = [mean(movingPoints(:,1)-fixedPoints(:,1)),mean(movingPoints(:,2)-fixedPoints(:,2))];
scatter(movingPoints(:,1),movingPoints(:,2),10,'filled','r'); axis image
hold on;
scatter(fixedPoints(:,1)+tformxy_linear(1),fixedPoints(:,2)+tformxy_linear(2),10,'filled','g');axis image
for ii = 1:size(movingPoints,1)
plot([movingPoints(ii,1),fixedPoints(ii,1)+tformxy_linear(1)],[movingPoints(ii,2),fixedPoints(ii,2)+tformxy_linear(2)],'k-'); 
end

xlabel('x (pixels)');
ylabel('y (pixels)'); legend('x-channel beads','registed y-channel beads');
title('paired data if only first-order tform applied');

save(strcat(fileFolder,'tformx2y_y_center_',num2str(W-center(1)),'_',num2str(center(2)),'_FoV_',num2str(ROI),'.mat'),'tformx2y');

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
        plot_pairs(dataX_cur,dataY_cur(pair_indx,:),dataX_cur_temp,dataY_cur,tform_y2x);
        %plot_pairs(dataX_paired,dataY_paired,dataX_cur_temp,dataY_cur,tform_y2x);
        end
        
    else 
        dataY_cur_temp = dataY_cur;
        dataY_cur(isnan(nanmin(distance,[],dim)),:)=[];
        dataX_paired = [dataX_paired;dataX_cur(pair_indx,:)];
        dataY_paired = [dataY_paired;dataY_cur];
        if rem(ii,200)==10
            %plot_pairs(dataX_paired,dataY_paired,dataX_cur,dataY_cur_temp,tform_y2x);
            plot_pairs(dataX_cur(pair_indx,:),dataY_cur,dataX_cur,dataY_cur_temp,tform_y2x);
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




function [dataX_paired,dataY_paired] = pairedData_initial(tform_y2x,dataX,dataY)
x_dataX = dataX(:,2);
y_dataX = dataX(:,3);
I_dataX = dataX(:,5);

x_dataY = dataY(:,2)+tform_y2x(1);
y_dataY = dataY(:,3)+tform_y2x(2);
I_dataY = dataY(:,5);

loss = 0;
dataX_paired = [];
dataY_paired = [];
for ii = 1:dataX(end,1)

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
        if rem(ii,10)==5
               %plot_pairs(dataX_paired,dataY_paired,dataX_cur_temp,dataY_cur,tform_y2x);
                % plot_pairs(dataX_cur,dataY_cur(pair_indx,:),dataX_cur_temp,dataY_cur,tform_y2x);
     
        end
        
    else 
        dataY_cur_temp = dataY_cur;
        dataY_cur(isnan(nanmin(distance,[],dim)),:)=[];
        dataX_paired = [dataX_paired;dataX_cur(pair_indx,:)];
        dataY_paired = [dataY_paired;dataY_cur];
        if rem(ii,10)==5
            %plot_pairs(dataX_paired,dataY_paired,dataX_cur,dataY_cur_temp,tform_y2x);
           % plot_pairs(dataX_cur(pair_indx,:),dataY_cur,dataX_cur,dataY_cur_temp,tform_y2x);
        end
        
    end

end

end
