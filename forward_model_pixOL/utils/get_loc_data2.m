function loc_data = get_loc_data2(gammaf, loc_rec)
loc_data = [];
loc_rec = reshape(loc_rec,3,[]);
gammaf = reshape(gammaf,15,[]);
z_xx = gammaf([1,7,10,13],:).';

%molecular parameters related to YY basis
z_yy = gammaf([2,8,11,14],:).';

%molecular parameters related to ZZ basis
z_zz = gammaf([3,9,12,15],:).';

%molecular parameters related toXY basis
z_xy = gammaf([4],:).';

%molecular parameters related to XZ basis
z_xz = gammaf([5],:).';

%molecular parameters related to YZ basis
z_yz = gammaf([6],:).';
for ii = 1:size(gammaf,2)

     pm = loc_rec(:,ii);
     %-----------------------------------------------------
    imx_xx = (z_xx(ii,2) / (eps + z_xx(ii,1))) * 10^2; %x position
    imy_xx = (z_xx(ii,3) / (eps + z_xx(ii,1))) * 10^2; %y position
    imz_xx = (z_xx(ii,4) / (eps + z_xx(ii,1))) * 10^2; %z position
    br_m_xx = z_xx(ii,1); %brightness scales


    % YY basis
    %-----------------------------------------------------

    imx_yy = (z_yy(ii,2) / (eps + z_yy(ii,1))) * 10^2; %x position
    imy_yy = (z_yy(ii,3) / (eps + z_yy(ii,1))) * 10^2; %y position
    imz_yy = (z_yy(ii,4) / (eps + z_yy(ii,1))) * 10^2; %z position
    br_m_yy = z_yy(ii,1); %brightness scales

    % ZZ basis
    %-----------------------------------------------------

    imx_zz = (z_zz(ii,2) / (eps + z_zz(ii,1))) * 10^2; %x position
    imy_zz = (z_zz(ii,3) / (eps + z_zz(ii,1))) * 10^2; %y position
    imz_zz = (z_zz(ii,4) / (eps + z_zz(ii,1))) * 10^2; %z position
    br_m_zz = z_zz(ii,1); %brightness scales

    % XY
    %-----------------------------------------------------
    br_m_xy = z_xy(ii,1); %brightness scales

    % XZ
    %-----------------------------------------------------
    br_m_xz = z_xz(ii,1); %brightness scales

    % YZ
    %-----------------------------------------------------
    br_m_yz = z_yz(ii,1); %brightness scales

    % combine position estimates
    %-----------------------------------------------------

    br_m = br_m_xx + br_m_yy + br_m_zz; % molecule brightness is the sum across XX, YY and ZZ


    imx = pm(1) + ((imx_xx * br_m_xx) + (imx_yy * br_m_yy) + (imx_zz * br_m_zz)) / (eps + br_m);
    imy = pm(2) + ((imy_xx * br_m_xx) + (imy_yy * br_m_yy) + (imy_zz * br_m_zz)) / (eps + br_m);
    imz = pm(3) + ((imz_xx * br_m_xx) + (imz_yy * br_m_yy) + (imz_zz * br_m_zz)) / (eps + br_m);

    % map to sencond moments
    %-----------------------------------------------------

    secondM = [br_m_xx, br_m_yy, br_m_zz, br_m_xy, br_m_xz, br_m_yz] / (br_m + eps);


    %update localizaton data
    %----------------------------------------------------
    loc_data = [loc_data;imx, imy,imz, br_m, secondM];

    end
end