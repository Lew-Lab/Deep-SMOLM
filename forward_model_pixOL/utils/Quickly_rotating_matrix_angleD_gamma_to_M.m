
function [muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(polar,azim,gamma)
% transfer the angle from degree unit to the radial unit
%polar = polar/180*pi;
%azim = azim/180*pi;

mux = cosd(azim).*sind(polar);
muy = sind(azim).*sind(polar);
muz = cosd(polar);

muxx = gamma.*mux.^2+(1-gamma)./3;
muyy = gamma.*muy.^2+(1-gamma)./3;
muzz = gamma.*muz.^2+(1-gamma)./3;
muxy = gamma.*mux.*muy;
muxz = gamma.*mux.*muz;
muyz = gamma.*muz.*muy;

end