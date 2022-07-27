
%
function [line_all,theta,phi] = generate_phantom_SMs()



R = 1000; % unit: nm
% line 1
phi_circ = linspace(0,2*pi,100000);
x_line1 = R*cos(phi_circ);
y_line1 = R*sin(phi_circ);
z_line1 = R*cos(phi_circ)*0;
line1 = [x_line1;y_line1;z_line1];

% line 2
theta_rot = 30/180*pi;
R_y = [cos(theta_rot),0,sin(theta_rot);
       0,1,0;
       -sin(theta_rot),0,cos(theta_rot)];
line2 = R_y*line1;

% line 3
theta_rot = 60/180*pi;
R_y = [cos(theta_rot),0,sin(theta_rot);
       0,1,0;
       -sin(theta_rot),0,cos(theta_rot)];
line3 = R_y*line1;

% line 4
theta_rot = 90/180*pi;
R_y = [cos(theta_rot),0,sin(theta_rot);
       0,1,0;
       -sin(theta_rot),0,cos(theta_rot)];
line4 = R_y*line1;

% line 5
theta_rot = 75/180*pi;
R_y = [cos(theta_rot),0,sin(theta_rot);
       0,1,0;
       -sin(theta_rot),0,cos(theta_rot)];
line5 = R_y*line1;

% line 6
theta_rot = 85/180*pi;
R_y = [cos(theta_rot),0,sin(theta_rot);
       0,1,0;
       -sin(theta_rot),0,cos(theta_rot)];
line6 = R_y*line1;

line_all = [line1,line2,line3,line4,line5];
theta = pi/2-atan(line_all(3,:)./sqrt(line_all(1,:).^2+line_all(2,:).^2));
phi = atan2(line_all(2,:),line_all(1,:));

% phi(theta>pi/2)=pi-phi(theta>pi/2);
theta(theta>pi/2) = pi-theta(theta>pi/2);
% phi = rem(phi,2*pi);
% phi(phi>pi)=phi(phi>pi)-2*pi;
end