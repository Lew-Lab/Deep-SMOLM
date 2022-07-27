%simulation of a dipole at an interface--By Adam Backer, Jan 1 2013
%Adapted from Axelrod, Journal of Microscopy article, 2012. (see journal club slides for more details)
function [basisImagex,basisImagey,basisBFPx,basisBFPy,basisImagex_dx,basisImagey_dx,basisImagex_dy,basisImagey_dy] = simDipole_v5_retrieved_w_z(z,z2,zh,N,wavelength,n1,n2,nh,NA,M,pix_size,xy_ind,pmask)

% Ting modified based on Oumeng's simDipole_v4

%inputs:
% x, y, z -- molecule coordinates note: z defocus length (unit:m)
% z2 -- distance of emitter below interface
% zh -- thickness of a thin film
% pmask -- phase mask
% N -- FT size
% n1 -- imaging media r.i.
% nh -- thin film r.i. (match this to n2, and set zh to zero if you don't want a thin film)
% n2 -- sample r.i. 
% N-- size
% NA -- numerical aperture
% M -- magnification

%outputs:
% basisImagex -- x channel basis images
% basisImagey -- y channel basis images

%%simulation parameters--feel free to change%%
lambda = wavelength;%wavelength

% parameter required to be added;


%calculate both pupil and image plane sampling, 
%one will affect the other, so make sure not to introduce aliasing

dx_true = (pix_size*1e-9/M);%image plane sampling
dx = n1*dx_true;%due to Abbe sine condition, scale by imaging medium r.i. (see appendix of my journal club)
%%???????  



dv = 1/(N*dx);%pupil sampling, related to image plane by FFT
% recall comb function, 1/dx is the preriod in fourier space(pupil space)
%%%????

%define pupil coordinates
temp=linspace((-1/(2*dx)),(1/(2*dx)),N);
[eta,xi] = meshgrid(temp);
% [eta,xi] = meshgrid(((-1/(2*dx))+(1/(2*N*dx))):dv:(-(1/(2*N*dx))+(1/(2*dx))),...
%     ((-1/(2*dx))+(1/(2*N*dx))):dv:(-(1/(N*2*dx))+(1/(2*dx))));

xBFP = lambda*eta;  % why scaled by wavelength??
yBFP = lambda*xi;
[phi,rho] = cart2pol(xBFP,yBFP);
rho_max = NA/n1;%pupil region of support determined by NA and imaging medium r.i.

k1 = n1*(2*pi/lambda);
kh = nh*(2*pi/lambda);
k2 = n2*(2*pi/lambda);
%rho(rho >= rho_max) = 0;
theta1 = asin(rho);%theta in matched medium
thetah = asin((n1/nh)*sin(theta1));%theta in thin film
theta2 = asin((n1/n2)*sin(theta1));%theta in mismatched medium
theta2 = real(theta2)-1i*abs(imag(theta2));

%%%%%%%%% Start
%Fresnel coefficients
tp_2h = 2*n2*cos(theta2)./(n2*cos(thetah) + nh*cos(theta2));
ts_2h = 2*n2*cos(theta2)./(nh*cos(thetah) + n2*cos(theta2));
tp_h1 = 2*nh*cos(thetah)./(nh*cos(theta1) + n1*cos(thetah));
ts_h1 = 2*nh*cos(thetah)./(n1*cos(theta1) + nh*cos(thetah));

rp_2h = (n2*cos(theta2) - nh*cos(thetah))./(n2*cos(theta2)+ nh*cos(thetah));
rs_2h = (nh*cos(theta2) - n2*cos(thetah))./(nh*cos(theta2)+ n2*cos(thetah));
rp_h1 = (nh*cos(thetah) - n1*cos(theta1))./(nh*cos(thetah)+ n1*cos(theta1));
rs_h1 = (n1*cos(thetah) - nh*cos(theta1))./(n1*cos(thetah)+ nh*cos(theta1));

%Axelrod's equations for E-fields in back focal plane

tp = tp_2h.*tp_h1.*exp(1i*kh*cos(thetah)*zh)./(1 + rp_2h.*rp_h1.*exp(2i*kh*zh*cos(thetah))); 
ts = ts_2h.*ts_h1.*exp(1i*kh*cos(thetah)*zh)./(1 + rs_2h.*rs_h1.*exp(2i*kh*zh*cos(thetah)));

% Es = ts.*(cos(theta1)./cos(theta2)).*(n1/n2).*(muy.*cos(phi) - mux.*sin(phi));
% Ep = tp.*((n1/n2).*(mux.*cos(phi) + muy.*sin(phi)).*cos(theta1) - muz*sin(theta1).*(n1/n2)^2.*(cos(theta1)./cos(theta2)));
% Esx - Es contributed by mux

% still has problem on how to get these to equation????????

%based on the equation above, seperating the mux,muy,muz compoment from two
%polirized electric field
Esx = ts.*(cos(theta1)./cos(theta2)).*(n1/n2).*(-sin(phi));
Esy = ts.*(cos(theta1)./cos(theta2)).*(n1/n2).*cos(phi);
Epx = tp.*(n1/n2).*cos(phi).*cos(theta1);
Epy = tp.*(n1/n2).*sin(phi).*cos(theta1);
Epz = tp.*(-sin(theta1).*(n1/n2)^2.*(cos(theta1)./cos(theta2)));

% Exx - Ex contributed by mux
% the first x represents x channel and y channel on the camera, the second
% x,y,z represents the compoment of mux, muy, muz from the orientation of
% dipole
Exx = (1./sqrt(cos(theta1))).*(cos(phi).*Epx - sin(phi).*Esx).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2)); %added defocus aberration + depth aberration
Exy = (1./sqrt(cos(theta1))).*(cos(phi).*Epy - sin(phi).*Esy).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2)); %added defocus aberration + depth aberration
Exz = (1./sqrt(cos(theta1))).*(cos(phi).*Epz).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2)); %added defocus aberration + depth aberration
Eyx = (1./sqrt(cos(theta1))).*(cos(phi).*Esx + sin(phi).*Epx).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2));
Eyy = (1./sqrt(cos(theta1))).*(cos(phi).*Esy + sin(phi).*Epy).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2));
Eyz = (1./sqrt(cos(theta1))).*(sin(phi).*Epz).*exp(1i*k1*z*cos(theta1)).*exp(1i*kh*zh*cos(thetah)).*exp(1i*k2*z2*cos(theta2));



% remove the electric component that is outside the accecptant region of
% objective lens
Exx(rho >= rho_max) = 0;
Exy(rho >= rho_max) = 0;
Exz(rho >= rho_max) = 0;
Eyx(rho >= rho_max) = 0;
Eyy(rho >= rho_max) = 0;
Eyz(rho >= rho_max) = 0;

%% Oumeng's coordinate flipping
% coord flipping
if xy_ind~=1
Exx = rot90(Exx);
Exy = rot90(Exy);
Exz = rot90(Exz);
Eyx = fliplr(Eyx);
Eyy = fliplr(Eyy);
Eyz = fliplr(Eyz);
end

% [~,~,sizePmask] = size(pmask);
% if sizePmask==2
%     pmaskx=pmask(:,:,1);
%     pmasky=pmask(:,:,2);
% elseif sizePmask==1
%     pmaskx=pmask;
%     pmasky=pmask;
% end
% 
% if xy_ind==1
%     pmaskx=rot90(pmask,3);
%     pmasky=pmask;
% end
%for propagation from BFP E-field to image plane via tube-lens, paraxial
%approximation is in force.
load(pmask);
pmaskx = exp(1i*pmaskX);
pmasky = exp(1i*pmaskY);
imgExx = fftshift(fft2(Exx.*pmaskx));
imgEyx = fftshift(fft2(Eyx.*pmasky));
imgExy = fftshift(fft2(Exy.*pmaskx));
imgEyy = fftshift(fft2(Eyy.*pmasky));
imgExz = fftshift(fft2(Exz.*pmaskx));
imgEyz = fftshift(fft2(Eyz.*pmasky));

if xy_ind~=1
% coord flipping
imgExx = fliplr(flipud(imgExx'));
imgExy = fliplr(flipud(imgExy'));
imgExz = fliplr(flipud(imgExz'));
imgEyx = flipud(imgEyx);
imgEyy = flipud(imgEyy);
imgEyz = flipud(imgEyz);
end


% euqation from backer's paper Eq.22
basisImagex(:,:,1) = abs(imgExx).^2;
basisImagex(:,:,2) = abs(imgExy).^2;
basisImagex(:,:,3) = abs(imgExz).^2;
% basisImagex(:,:,4) = 2*real(imgExx.*conj(imgExy));
% basisImagex(:,:,5) = 2*real(imgExx.*conj(imgExz));
% basisImagex(:,:,6) = 2*real(imgExy.*conj(imgExz));  

basisImagex(:,:,4) = 2*real(conj(imgExx).*imgExy);
basisImagex(:,:,5) = 2*real(conj(imgExx).*imgExz);
basisImagex(:,:,6) = 2*real(conj(imgExy).*imgExz);  
%the results are same with the above equation

% temp1=real(imgEyy);
% temp2=imag(imgEyy);
% imgEyy=temp1+conj(temp2.');



basisImagey(:,:,1) = abs(imgEyx).^2;
basisImagey(:,:,2) = abs(imgEyy).^2;
basisImagey(:,:,3) = abs(imgEyz).^2;
basisImagey(:,:,4) = 2*real(imgEyx.*conj(imgEyy));
basisImagey(:,:,5) = 2*real(imgEyx.*conj(imgEyz));
basisImagey(:,:,6) = 2*real(imgEyy.*conj(imgEyz));

% basisImagey(:,:,4) = 2*real(conj(imgEyx).*imgEyy);
% basisImagey(:,:,5) = 2*real(conj(imgEyx).*imgEyz);
% basisImagey(:,:,6) = 2*real(conj(imgEyy).*imgEyz);


basisBFPx(:,:,1) = abs(Exx).^2;
basisBFPx(:,:,2) = abs(Exy).^2;
basisBFPx(:,:,3) = abs(Exz).^2;
basisBFPx(:,:,4) = 2*real(Exx.*conj(Exy));
basisBFPx(:,:,5) = 2*real(Exx.*conj(Exz));
basisBFPx(:,:,6) = 2*real(Exy.*conj(Exz));
basisBFPy(:,:,1) = abs(Eyx).^2;
basisBFPy(:,:,2) = abs(Eyy).^2;
basisBFPy(:,:,3) = abs(Eyz).^2;
basisBFPy(:,:,4) = 2*real(Eyx.*conj(Eyy));
basisBFPy(:,:,5) = 2*real(Eyx.*conj(Eyz));
basisBFPy(:,:,6) = 2*real(Eyy.*conj(Eyz));

if xy_ind==1
    [X,~] = meshgrid(1*1e-9*(1:N),1e-9*(1:N));
    maskdx = exp(1j*2*pi*X/(N*dx_true));
    maskdy = rot90(maskdx);
    %maskdy = (maskdx);
    
    imgExx_dx = fftshift(fft2(Exx.*pmaskx.*maskdx));
    imgEyx_dx = fftshift(fft2(Eyx.*pmasky.*maskdx));
    imgExy_dx = fftshift(fft2(Exy.*pmaskx.*maskdx));
    imgEyy_dx = fftshift(fft2(Eyy.*pmasky.*maskdx));
    imgExz_dx = fftshift(fft2(Exz.*pmaskx.*maskdx));
    imgEyz_dx = fftshift(fft2(Eyz.*pmasky.*maskdx));


%     % coord flipping
%     imgExx_dx = fliplr(flipud(imgExx_dx'));
%     imgExy_dx = fliplr(flipud(imgExy_dx'));
%     imgExz_dx = fliplr(flipud(imgExz_dx'));
%     imgEyx_dx = flipud(imgEyx_dx);
%     imgEyy_dx = flipud(imgEyy_dx);
%     imgEyz_dx = flipud(imgEyz_dx);
    
    imgExx_dy = fftshift(fft2(Exx.*pmaskx.*maskdy));
    imgEyx_dy = fftshift(fft2(Eyx.*pmasky.*maskdy));
    imgExy_dy = fftshift(fft2(Exy.*pmaskx.*maskdy));
    imgEyy_dy = fftshift(fft2(Eyy.*pmasky.*maskdy));
    imgExz_dy = fftshift(fft2(Exz.*pmaskx.*maskdy));
    imgEyz_dy = fftshift(fft2(Eyz.*pmasky.*maskdy));


%     % coord flipping
%     imgExx_dy = fliplr(flipud(imgExx_dy'));
%     imgExy_dy = fliplr(flipud(imgExy_dy'));
%     imgExz_dy = fliplr(flipud(imgExz_dy'));
%     imgEyx_dy = flipud(imgEyx_dy);
%     imgEyy_dy = flipud(imgEyy_dy);
%     imgEyz_dy = flipud(imgEyz_dy);
    
    
    basisImagex_dx(:,:,1) = abs(imgExx_dx).^2;
    basisImagex_dx(:,:,2) = abs(imgExy_dx).^2;
    basisImagex_dx(:,:,3) = abs(imgExz_dx).^2;
    basisImagex_dx(:,:,4) = 2*real(conj(imgExx_dx).*imgExy_dx);
    basisImagex_dx(:,:,5) = 2*real(conj(imgExx_dx).*imgExz_dx);
    basisImagex_dx(:,:,6) = 2*real(conj(imgExy_dx).*imgExz_dx);  


    basisImagey_dx(:,:,1) = abs(imgEyx_dx).^2;
    basisImagey_dx(:,:,2) = abs(imgEyy_dx).^2;
    basisImagey_dx(:,:,3) = abs(imgEyz_dx).^2;
    basisImagey_dx(:,:,4) = 2*real(imgEyx_dx.*conj(imgEyy_dx));
    basisImagey_dx(:,:,5) = 2*real(imgEyx_dx.*conj(imgEyz_dx));
    basisImagey_dx(:,:,6) = 2*real(imgEyy_dx.*conj(imgEyz_dx));
    
    basisImagex_dy(:,:,1) = abs(imgExx_dy).^2;
    basisImagex_dy(:,:,2) = abs(imgExy_dy).^2;
    basisImagex_dy(:,:,3) = abs(imgExz_dy).^2;
    basisImagex_dy(:,:,4) = 2*real(conj(imgExx_dy).*imgExy_dy);
    basisImagex_dy(:,:,5) = 2*real(conj(imgExx_dy).*imgExz_dy);
    basisImagex_dy(:,:,6) = 2*real(conj(imgExy_dy).*imgExz_dy);  


    basisImagey_dy(:,:,1) = abs(imgEyx_dy).^2;
    basisImagey_dy(:,:,2) = abs(imgEyy_dy).^2;
    basisImagey_dy(:,:,3) = abs(imgEyz_dy).^2;
    basisImagey_dy(:,:,4) = 2*real(imgEyx_dy.*conj(imgEyy_dy));
    basisImagey_dy(:,:,5) = 2*real(imgEyx_dy.*conj(imgEyz_dy));
    basisImagey_dy(:,:,6) = 2*real(imgEyy_dy.*conj(imgEyz_dy));
 
else
    basisImagex_dx=[];
    basisImagey_dx=[];
    basisImagex_dy=[];
    basisImagey_dy=[];
end







end
