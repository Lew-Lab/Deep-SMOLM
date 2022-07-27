function s=generateSignal_distribution()

x = 0:0.01:1;
Fx = x.^1.5;
F_dist = makedist('PiecewiseLinear', 'x', x, 'Fx', Fx);
s1 = random(F_dist, 100000,1)*2000;

x = 0:0.01:1;
Fx = x.^1;
F_dist = makedist('PiecewiseLinear', 'x', x, 'Fx', Fx);
s2 = random(F_dist, 50000,1)*2000+3000;

s = [s1;s2];
end