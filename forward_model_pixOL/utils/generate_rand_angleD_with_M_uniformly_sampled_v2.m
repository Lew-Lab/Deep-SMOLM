function [thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_with_M_uniformly_sampled_v2(n_SMs)
    n_SMs_large = n_SMs*20;
    
    XX2_avg = rand(n_SMs_large,1);
    YY2_avg = rand(n_SMs_large,1);
    indx = (XX2_avg+YY2_avg)>1;
    XX2_avg(indx)=[];
    YY2_avg(indx)=[];
    ZZ2_avg = 1-XX2_avg-YY2_avg;
    gamma = rand(length(XX2_avg),1);
    
    
   
    signX = sign(rand(length(XX2_avg),1)-0.5);
    signY = sign(rand(length(XX2_avg),1)-0.5);
    
    mux = signX.*sqrt(XX2_avg);
    muy = signY.*sqrt(YY2_avg);
    muz = sqrt(ZZ2_avg);
    
    thetaD = acos(muz)/pi*180;
    phiD = atan2(muy,mux)/pi*180;
    
    thetaD_SMs = thetaD(1:n_SMs).';
    phiD_SMs = phiD(1:n_SMs).';
    gamma_SMs = gamma(1:n_SMs).';
    
end


