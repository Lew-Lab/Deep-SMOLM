
N_colorSplit=403;
[line_all,theta,phi] = generate_phantom_SMs();

theta_color = linspace(min(theta),max(theta),N_colorSplit);
phi_color = linspace(min(phi),max(phi),N_colorSplit);

[~,theta_color_idx] = min(abs(repmat(theta.',1,length(N_colorSplit))-repmat(theta_color,length(theta),1)),[],2);
[~,phi_color_idx] = min(abs(repmat(phi.',1,length(N_colorSplit))-repmat(phi_color,length(phi),1)),[],2);

figure();
color = turbo(N_colorSplit);
scatter(line_all(1,:),line_all(2,:),[],color(theta_color_idx,:),'filled'); axis image
colormap("turbo"); caxis([min(theta_color)/pi*180,max(theta_color)/pi*180]); colorbar;
xlabel('x (nm)'); ylabel('y (nm)'); 
title('\theta (\circ)');

figure();
load("colorSpace.mat");
color = squeeze(colorSpace);
scatter(line_all(1,:),line_all(2,:),[],color(phi_color_idx,:),'filled'); axis image
colormap(color); caxis([min(phi_color)/pi*180,max(phi_color)/pi*180]); colorbar;
xlabel('x (nm)'); ylabel('y (nm)'); 
title('\phi (\circ)');
