function [sXX,sYY,sZZ,sXY,sXZ,sYZ,signal_SMs] = generate_non_physical_uniform_sM(signal,n_SMs)

coeff = 1.5;
sXX = rand(n_SMs,1)*signal/coeff;
sYY = rand(n_SMs,1)*signal/coeff;
sZZ = rand(n_SMs,1)*signal/coeff;
sXY = (rand(n_SMs,1)-0.5)*signal/coeff;
sXZ = (rand(n_SMs,1)-0.5)*signal/coeff;
sYZ = (rand(n_SMs,1)-0.5)*signal/coeff;
signal_SMs = (sXX+sYY+sZZ);
end