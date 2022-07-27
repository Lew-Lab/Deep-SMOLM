function visualizeBases(basis_matrix,varargin)
            
            s_1=opt2struct(varargin);
            if (isfield(s_1,'reusefigure') && s_1.reusefigure)
                h=gcf;
            else
                h=figure;
            end
            
            [a,n]=size(basis_matrix);
            m=sqrt(a/2);
            obj=struct();
            temp=reshape(basis_matrix(:,1),m,2*m);
            obj.XXxBasis=temp(:,1:m);  obj.XXyBasis=temp(:,m+1:2*m);
            temp=reshape(basis_matrix(:,2),m,2*m);
            obj.YYxBasis=temp(:,1:m);  obj.YYyBasis=temp(:,m+1:2*m);
            temp=reshape(basis_matrix(:,3),m,2*m);
            obj.ZZxBasis=temp(:,1:m);  obj.ZZyBasis=temp(:,m+1:2*m);
            temp=reshape(basis_matrix(:,4),m,2*m);
            obj.XYxBasis=temp(:,1:m);  obj.XYyBasis=temp(:,m+1:2*m);
            temp=reshape(basis_matrix(:,5),m,2*m);
            obj.XZxBasis=temp(:,1:m);  obj.XZyBasis=temp(:,m+1:2*m);
            temp=reshape(basis_matrix(:,6),m,2*m);
            obj.YZxBasis=temp(:,1:m);  obj.YZyBasis=temp(:,m+1:2*m);
            
            %max_brightness=max(max(basis_matrix));
            max_brightness=1;
            
            util_plot(h,obj.XXxBasis,obj.XXyBasis,[.01,.70],.25,.2,'basis','XX');
            util_plot(h,obj.YYxBasis,obj.YYyBasis,[.33,.70],.25,.2,'basis','YY');
            util_plot(h,obj.ZZxBasis,obj.ZZyBasis,[.67,.70],.25,.2,'basis','ZZ');
            util_plot(h,obj.XYxBasis,obj.XYyBasis,[.01,.25],.25,.2,'negativeVal',true,'basis','XY');
            util_plot(h,obj.XZxBasis,obj.XZyBasis,[.33,.25],.25,.2,'negativeVal',true,'basis','XZ');
            util_plot(h,obj.YZxBasis,obj.YZyBasis,[.67,.25],.25,.2,'negativeVal',true,'basis','YZ');
            
            function util_plot(handleIn,imgIn1,imgIn2,pos,w,h,varargin)
                s=opt2struct(varargin);
                ax1=axes(handleIn,'Position',[pos(1),pos(2),w,h]);
                imagesc(imgIn1/max_brightness);
                axis image;axis off
                
                %colorbar
                caxis manual
                cmax=max([max(imgIn1(:)),max(imgIn2(:))])/max_brightness;
                cmin=max([min(imgIn1(:)),min(imgIn2(:))])/max_brightness;
                colormap(ax1,parula);
                
                if (isfield(s,'negativeval') && s.negativeval)
                    caxis([-cmax,cmax])
                else
                    caxis([cmin,cmax])
                end
                %c1=colorbar;
                
                %set(c1.Label,'Rotation',90);
                %c1.Position=[pos(1)+w+.005,pos(2)-h,.01,2*h];
                %set(c1, 'YAxisLocation','left')
                
                %remove ticks since both pannels use the same range value
                %c1.Ticks=[];
                
                % markup
                if (isfield(s,'basis') && (any(strcmp(s.basis,{'XX','XY'}))))
                    xLim=get(gca,'Xlim');
                    yLim=get(gca,'Ylim');
                    ht = text(0.9*xLim(1)-0.1*xLim(2),0.1*yLim(1)+0.9*yLim(2),...
                        'x-channel',...
                        'Color','k',...
                        'Rotation',90,...
                        'FontWeight','bold');
                end
                %title
                title(s.basis)
                ax2=axes(handleIn,'Position',[pos(1),pos(2)-h,w,h]);
                imagesc(imgIn2/max_brightness);
                axis image;axis off
                caxis manual
                if (isfield(s,'negativeval') && s.negativeval)
                    caxis([-cmax,cmax])
                else
                    caxis([cmin,cmax])
                end
                c2=colorbar;
                
                c2.Position=[pos(1)+w-.035,pos(2)-h,.015,2*h];
                
                % markup
                if (isfield(s,'basis') && (any(strcmp(s.basis,{'XX','XY'}))))
                    xLim=get(gca,'Xlim');
                    yLim=get(gca,'Ylim');
                    ht = text(0.9*xLim(1)-0.1*xLim(2),0.1*yLim(1)+0.9*yLim(2),...
                        'y-channel',...
                        'Color','k',...
                        'Rotation',90,...
                        'FontWeight','bold');
  
                end
            end
        end
