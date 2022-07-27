function [lambda,loc] = generate_lambda(signal_SMs,x_SMs_phy,y_SMs_phy,z_SMs_phy,M,imgPara)
pixel_sizexy = imgPara.pix_sizex;
pixel_sizez = imgPara.pix_sizez;

locx = round(x_SMs_phy/pixel_sizexy)*pixel_sizexy;
locy = round(y_SMs_phy/pixel_sizexy)*pixel_sizexy;
locz = round(z_SMs_phy/pixel_sizez)*pixel_sizez;

dx = (x_SMs_phy-locx)/100;
dy = (y_SMs_phy-locy)/100;
dz = (z_SMs_phy-locz)/100;

S_Ms = repmat(signal_SMs,6,1).*M;
sXX_dx = dx.*S_Ms(1,:);
sYY_dx = dx.*S_Ms(2,:);
sZZ_dx = dx.*S_Ms(3,:);
sXX_dy = dy.*S_Ms(1,:);
sYY_dy = dy.*S_Ms(2,:);
sZZ_dy = dy.*S_Ms(3,:);
sXX_dz = dz.*S_Ms(1,:);
sYY_dz = dz.*S_Ms(2,:);
sZZ_dz = dz.*S_Ms(3,:);

loc = [locx;locy;locz];
lambda = [S_Ms;sXX_dx;sYY_dx;sZZ_dx;sXX_dy;sYY_dy;sZZ_dy;sXX_dz;sYY_dz;sZZ_dz];

end