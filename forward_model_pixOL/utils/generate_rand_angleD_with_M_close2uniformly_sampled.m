function [thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_with_M_uniformly_sampled(n_SMs)
    n_SMs_large = n_SMs*20;
    
    x = 0:0.01:1;
    Fx = x.^3;
    %plot(x,Fx);
    F_dist = makedist('PiecewiseLinear', 'x', x, 'Fx', Fx);
    % X1 = random(F_dist, n_SMs_large,1);
    % X2 = random(F_dist, n_SMs_large,1);
    % X3 = random(F_dist, n_SMs_large,1);
    % figure(); histogram(X1.*X2.*X3);
    
    
    X = random(F_dist, n_SMs_large,1);
    Y = random(F_dist, n_SMs_large,1);
    indx = X.^2+Y.^2>1;
    X(indx)=[];
    Y(indx)=[];
    Z = sqrt(1-X.^2-Y.^2);
    gamma = random(F_dist, length(X),1);
    %gamma = rand(length(X),1);
    %gamma = 1;
    signX = sign(rand(length(X),1)-0.5);
    signY = sign(rand(length(X),1)-0.5);
    
    mux = signX.*X;
    muy = signY.*Y;
    muz = Z;
    
    thetaD = acos(muz)/pi*180;
    phiD = atan2(muy,mux)/pi*180;
    
    thetaD_SMs = thetaD(1:n_SMs).';
    phiD_SMs = phiD(1:n_SMs).';
    gamma_SMs = gamma(1:n_SMs).';
    
end


