classdef Nanoscope


    properties (Constant)
        pixelUpsample = 1; % object-space pixel size upsampling factor
    end

    properties

        % imaging parameters
        %--------------------------------------------------------
        pixelSize = 58.5; % object pixel size (nm)
        numApt = 1.4; % numerical aperture
        emissWavelength = 610; % central emission wavelength (nm)
        ADcount = .49; % photoelectron conversion
        offset = 0; % baseline of camera (in digital units)
        refractiveIndx = 1.518; % refractive index of the medium
        refractiveIndxSam = 1.334; % refractive index of the sample
        imageSize = 95; % side length of the region of interest

        %===========================
        % phase mask parameters
        %--------------------------------------------------------


        % phase mask parameters
        %--------------------------------------------------------
        phaseMaskPara = struct('maskName', 'tri-spot', ... %parameters of the phase mask mounted on SLM
            'pupilRadius', 80, ...
            'x_shift_center_position', 0, ...
            'y_shift_center_position', 0, ...
            'maskRotation', 0);

        %===========================
        % phase mask mounted on SLM
        %---------------------------------------------------------

        phaseMask; %phase mask mounted on SLM

        %===========================
        %bases images of a dipole
        %---------------------------------------------------------

        XXxBasis %XX basis image at x_channel
        XXyBasis %XX basis image at y_channel
        YYxBasis %YY basis image at x_channel
        YYyBasis %YY basis image at y_channel
        ZZxBasis %ZZ basis image at x_channel
        ZZyBasis %ZZ basis image at y_channel
        XYxBasis %XY basis image at x_channel
        XYyBasis %XY basis image at y_channel
        XZxBasis %XZ basis image at x_channel
        XZyBasis %XZ basis image at y_channel
        YZxBasis %YZ basis image at x_channel
        YZyBasis %YZ basis image at y_channel
        brightnessScaling %Brightness scaling factor of a dipole
    end

    methods

        % constructor
        %--------------------------------------------------------
        function obj = Nanoscope(varargin)
            %Nanoscope initializes an instance of Nanoscope object
            %option-value pairs:
            %                                   pixelSize: scalar- object
            %                                   pixel size (nm)
            %                                   (default=58.5)
            %............................................................
            %                                    numApt:  scalar-  numerical
            %                                    aperture of the microscope
            %                                    (default=1.4)
            %............................................................
            %                                    emissWavelength: scalar-
            %                                    emission wavelength of
            %                                    light radiated from
            %                                    emitters (in nm) (default=637)
            %............................................................
            %                                    offset:     scalar or array
            %                                    (m*m) with m the image size
            %                                    (values of the offset are in the raw
            %                                    camera units) (default=0)
            %............................................................
            %                                    ADcount:  scalar-
            %                                    photoelectron conversion
            %                                    (default=.49)
            %............................................................
            %                                   refractiveIndx: scalar- the
            %                                   refractive index of the
            %                                   imaging medium
            %                                   (default=1.51)
            %............................................................
            %                                   imageSize: scalar- size of
            %                                   the sqaure region of interest to
            %                                   be analyzed
            %............................................................
            %                                   phaseMaskPara: structure
            %                                   with fields:
            %                                           maskname: string- name of
            %                                           the mask
            %............................................................
            %                                           pupilRadius:
            %                                           integer- radius of
            %                                           the pupil in units
            %                                           of SLM pixels
            %                                           (default=40)
            %............................................................
            %                                           x_shift_center_position:
            %                                           scalar- shift of
            %                                           the center pixel of
            %                                           the SLM w.r.t. the
            %                                           pupil in x
            %                                           direction (in units
            %                                           of SLM pixels)
            %                                           (default=0)
            %............................................................
            %                                           y_shift_center_position:
            %                                           scalar- shift of
            %                                           the center pixel of
            %                                           the SLM w.r.t. the
            %                                           pupil in y
            %                                           direction (in units
            %                                           of SLM pixels)
            %                                           (default=0)
            %............................................................
            %                                           maskRotation:
            %                                           integer- rotation of
            %                                           phase mask
            %                                           clockwise (in 90
            %                                           degree units)
            %............................................................
            %                                           zeroorder:
            %                                           array(1,2) in [0,1]
            %                                           models the pupil
            %                                           (for left and
            %                                           right)
            %                                           like
            %                                           (1-zeroorder)*pmask+zeroorder*abs(pmask)
            %                                           that is a weighted
            %                                           sum of pupil and clear apertur

            s = opt2struct(varargin);

            %=========================
            %intialize imaging parameters based on iput options
            %=========================
            if isfield(s, 'pixelsize')
                if (isnumeric(s.pixelsize) && s.pixelsize > 0)
                    obj.pixelSize = s.pixelsize;
                else
                    msg = sprintf(['Expecting a positive numeric type for pixelSize.\n', ...
                        'Setting pixelSize property to default %.2f'], obj.pixelSize);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'numapt')
                if (isnumeric(s.numapt) && s.numapt > 0)
                    obj.numApt = s.numapt;
                else
                    msg = sprintf(['Expecting a positive numeric type for numApt.\n', ...
                        'Setting numApt property to default %.2f'], obj.numApt);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'emisswavelength')
                if (isnumeric(s.emisswavelength) && s.emisswavelength > 0)
                    obj.emissWavelength = s.emisswavelength;
                else
                    msg = sprintf(['Expecting a positive numeric type for emissWavelength.\n', ...
                        'Setting emissWavelength property to default %.2f'], obj.emissWavelength);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'offset')
                if (isnumeric(s.offset) && all(s.offset(:)) > 0)
                    obj.offset = s.offset;
                else
                    msg = sprintf(['Expecting a positive numeric type for offset.\n', ...
                        'Setting offset property to default %.2f'], obj.offset);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'adcount')
                if (isnumeric(s.adcount) && s.adcount > 0)
                    obj.ADcount = s.adcount;
                else
                    msg = sprintf(['Expecting a positive numeric type for ADcount.\n', ...
                        'Setting ADcount property to default %.2f'], obj.ADcount);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'refractiveindx')
                if (isnumeric(s.refractiveindx) && s.refractiveindx > 0)
                    obj.refractiveIndx = s.refractiveindx;
                else
                    msg = sprintf(['Expecting a positive numeric type for refractiveIndx.\n', ...
                        'Setting refractiveIndx property to default %.2f'], obj.refractiveIndx);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'refractiveindxsam')
                if (isnumeric(s.refractiveindxsam) && s.refractiveindxsam > 0)
                    obj.refractiveIndxSam = s.refractiveindxsam;
                else
                    msg = sprintf(['Expecting a positive numeric type for refractiveIndxSam.\n', ...
                        'Setting refractiveIndxSam property to default %.2f'], obj.refractiveIndxSam);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            if isfield(s, 'imagesize')
                if (isnumeric(s.imagesize) && s.imagesize > 0)
                    obj.imageSize = s.imagesize;
                else
                    msg = sprintf(['Expecting a positive numeric type for imageSize.\n', ...
                        'Setting imageSize property to default %.2f'], obj.imageSize);
                    warning('Nanoscope:InconsistentInputType', msg)
                end
            end

            %display imaging parameters
            %--------------------------------------------------
            ImagingInfo = sprintf(['pixel size: %.2f\n', 'numerical aperture: %.2f\n', ...
                'emission wavelength: %.2f\n', 'offset: %.2f\n', ...
                'AD count: %.2f\n', 'medium refractive index:  %.2f\n', ...
                'sample refractive index:  %.2f\n', 'image size: %.2f\n'], ...
                obj.pixelSize, obj.numApt, obj.emissWavelength, ...
                obj.offset, obj.ADcount, obj.refractiveIndx, obj.refractiveIndxSam, obj.imageSize);

            fprintf('%s\n', repmat('=', 1, 20));
            fprintf('Imaging parameters\n');
            fprintf('%s\n', repmat('=', 1, 20));
            disp(ImagingInfo)

            %=========================
            %initialize phase mask parameters based on input options
            %=========================
            if isfield(s, 'phasemaskpara')
                %                     if isfield(s.phasemaskpara,'maskname')
                %                         if (ischar(s.phasemaskpara.maskname))
                %                             obj.phaseMaskPara.maskName=s.phasemaskpara.maskname;
                %                         else
                %                             msg=sprintf(['Expecting a character array type for maskName.\n',...
                %                                 'Setting pupilRadius property to default %s'],obj.phaseMaskPara.maskName);
                %                             warning('Nanoscope:phaseMaskPara:InconsistentInputType',msg)
                %                         end
                %                     end
                %                     if isfield(s.phasemaskpara,'pupilradius')
                %                         if (isnumeric(s.phasemaskpara.pupilradius)&&s.phasemaskpara.pupilradius>0)
                %                             obj.phaseMaskPara.pupilRadius=s.phasemaskpara.pupilradius;
                %                         else
                %                             msg=sprintf(['Expecting a positive numeric type for pupilRadius.\n',...
                %                                 'Setting pupilRadius property to default %.2f', obj.phaseMaskPara.pupilRadius]);
                %                             warning('Nanoscope:phaseMaskPara:InconsistentInputType',msg)
                %                         end
                %                     end
                %                     if isfield(s.phasemaskpara,'x_shift_center_position')
                %                         if (isnumeric(s.phasemaskpara.x_shift_center_position))
                %                             obj.phaseMaskPara.x_shift_center_position=s.phasemaskpara.x_shift_center_position;
                %                         else
                %                             msg=sprintf(['Expecting a  numeric type for x_shift_center_position.\n',...
                %                                 'Setting x_shift_center_position property to default %.2f', obj.phaseMaskPara.x_shift_center_position]);
                %                             warning('Nanoscope:phaseMaskPara:InconsistentInputType',msg)
                %                         end
                %                     end
                %                     if isfield(s.phasemaskpara,'y_shift_center_position')
                %                         if (isnumeric(s.phasemaskpara.y_shift_center_position))
                %                             obj.phaseMaskPara.y_shift_center_position=s.phasemaskpara.y_shift_center_position;
                %                         else
                %                             msg=sprintf(['Expecting a  numeric type for y_shift_center_position.\n',...
                %                                 'Setting y_shift_center_position property to default %.2f', obj.phaseMaskPara.y_shift_center_position]);
                %                             warning('Nanoscope:phaseMaskPara:InconsistentInputType',msg)
                %                         end
                %                     end
                %
                %                      if isfield(s.phasemaskpara,'maskrotation')
                %                         if (isnumeric(s.phasemaskpara.maskrotation))
                %                             obj.phaseMaskPara.maskRotation=s.phasemaskpara.maskrotation;
                %                         else
                %                             msg=sprintf(['Expecting a  numeric type for maskRotation.\n',...
                %                                 'Setting maskRotation property to default %.2f', obj.phaseMaskPara.maskRotation]);
                %                             warning('Nanoscope:phaseMaskPara:InconsistentInputType',msg)
                %                         end
                %                      end
                %             end
                obj.phaseMaskPara = s.phasemaskpara;
            end
            %display phase mask parameters
            %--------------------------------------------------
            phaseMaskInfo = sprintf(['mask name: %s\n', 'pupil radius: %.2f\n'], ...
                obj.phaseMaskPara.maskName, obj.phaseMaskPara.pupilRadius);

            fprintf('%s\n', repmat('=', 1, 20));
            fprintf('Phase mask parameters\n');
            fprintf('%s\n', repmat('=', 1, 20));
            disp(phaseMaskInfo)
            %=========================

            %phase mask initialization

            % apply zero order term
            if isfield(s, 'phasemaskpara') && isfield(s.phasemaskpara, 'zeroorder')

                obj.phaseMask = Nanoscope.mountPhaseMask(obj, 'zeroorder', ...
                    s.phasemaskpara.zeroorder);
            elseif isfield(s, 'phasemaskpara') && isfield(s.phasemaskpara, 'zeroorder') && isfield(s.phasemaskpara, 'amplitudemask')

                obj.phaseMask = Nanoscope.mountPhaseMask(obj, 'zeroorder', ...
                    s.phasemaskpara.zeroorder, 'amplitudemask', s.phasemaskpara.amplitudemask);
            elseif isfield(s, 'phasemaskpara') && isfield(s.phasemaskpara, 'amplitudemask')

                obj.phaseMask = Nanoscope.mountPhaseMask(obj, 'amplitudemask', ...
                    s.phasemaskpara.amplitudemask);
            else
                obj.phaseMask = Nanoscope.mountPhaseMask(obj);
            end
            %brightness scaling
            obj.brightnessScaling = obj.brightnessScalingCompute();

            %x_channel initialization
            obj.XXxBasis = obj.computeBasis(obj, 'XX' ...
                , true, 'x_channel', true, 'crop', true);
            obj.YYxBasis = obj.computeBasis(obj, 'YY' ...
                , true, 'x_channel', true, 'crop', true);
            obj.ZZxBasis = obj.computeBasis(obj, 'ZZ' ...
                , true, 'x_channel', true, 'crop', true);
            obj.XYxBasis = obj.computeBasis(obj, 'XY' ...
                , true, 'x_channel', true, 'crop', true);
            obj.XZxBasis = obj.computeBasis(obj, 'XZ' ...
                , true, 'x_channel', true, 'crop', true);
            obj.YZxBasis = obj.computeBasis(obj, 'YZ' ...
                , true, 'x_channel', true, 'crop', true);

            %y_channel initialization
            obj.XXyBasis = obj.computeBasis(obj, 'XX' ...
                , true, 'y_channel', true, 'crop', true);
            obj.YYyBasis = obj.computeBasis(obj, 'YY' ...
                , true, 'y_channel', true, 'crop', true);
            obj.ZZyBasis = obj.computeBasis(obj, 'ZZ' ...
                , true, 'y_channel', true, 'crop', true);
            obj.XYyBasis = obj.computeBasis(obj, 'XY' ...
                , true, 'y_channel', true, 'crop', true);
            obj.XZyBasis = obj.computeBasis(obj, 'XZ' ...
                , true, 'y_channel', true, 'crop', true);
            obj.YZyBasis = obj.computeBasis(obj, 'YZ' ...
                , true, 'y_channel', true, 'crop', true);
        end

        % properties set methods
        %---------------------------------------------------------
        function obj = set.pixelSize(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for pixel size')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for pixel size')

                end
                obj.pixelSize = val;
            end

        end

        function obj = set.numApt(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for numerical aperture')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for numerical aperture')

                end
                obj.numApt = val;
            end


        end


        function obj = set.emissWavelength(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for emission wavelength')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for emission wavelength')

                end
                obj.emissWavelength = val;

            end
        end


        function obj = set.refractiveIndx(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for refractive index')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for refractive index')

                end
                obj.refractiveIndx = val;
            end


        end

        function obj = set.refractiveIndxSam(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for refractive index of sample')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for refractive index of sample')

                end
                obj.refractiveIndxSam = val;
            end


        end


        function obj = set.ADcount(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for A/D count')
                end
                if ~(val > 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting a positive value for A/D count')

                end
                obj.ADcount = val;
            end

        end

        function obj = set.offset(obj, val)
            if nargin > 0
                if ~isnumeric(val)
                    error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for offset')
                end
                if any(val < 0)
                    error('Nanoscope:InconsistentInputValue', 'Expecting  positive values for offset')

                end
                obj.offset = val;
            end

        end


        %---------------------------------------------------------
        function obj = set.phaseMaskPara(obj, s)

            if nargin > 0
                s_t = struct();

                if ~isstruct(s)

                    error('Nanoscope:InconsistentInputType', 'Expecting a structure type as input')

                end

                if isfield(s, 'maskname')

                    if ~(ischar(s.maskname))

                        error('Nanoscope:InconsistentInputType', 'Expecting a character array for maskName')

                    end
                    s_t.maskName = s.maskname;
                else

                    s_t.maskName = obj.phaseMaskPara.maskName;
                end

                if isfield(s, 'pupilradius')

                    if ~((isnumeric(s.pupilradius) && (s.pupilradius > 0)))

                        error('Nanoscope:InconsistentInputType', 'Expecting a positive number for pupilRadius')

                    end
                    s_t.pupilRadius = s.pupilradius;

                else
                    s_t.pupilRadius = obj.phaseMaskPara.pupilRadius;
                end


                if isfield(s, 'x_shift_center_position')

                    if ~(isnumeric(s.x_shift_center_position))

                        error('Nanoscope:InconsistentInputType', 'Expecting a numeric type for x_shift_center_position')
                    end
                    s_t.x_shift_center_position = s.x_shift_center_position;
                else

                    s_t.x_shift_center_position = obj.phaseMaskPara.x_shift_center_position;
                end

                if isfield(s, 'y_shift_center_position')

                    if ~(isnumeric(s.y_shift_center_position))

                        error('Nanoscope:InconsistentInputType', 'Expecting a  numeric type for y_shift_center_position')
                    end
                    s_t.y_shift_center_position = s.y_shift_center_position;

                else

                    s_t.y_shift_center_position = obj.phaseMaskPara.y_shift_center_position;

                end

                if isfield(s, 'maskrotation')

                    if ~(floor(s.maskrotation) == s.maskrotation)

                        error('Nanoscope:InconsistentInputType', 'Expecting a positive integer for maskRotation')

                    end
                    s_t.maskRotation = s.maskrotation;

                else

                    s_t.maskRotation = obj.phaseMaskPara.maskRotation;

                end

            end

            obj.phaseMaskPara = struct(s_t);
        end

    end

    %% methods for properties initialization

    %--------------------------------------------------------

    %%

    methods (Access = protected)

        function brightnessScaling = brightnessScalingCompute(obj)
            %brightnessScalingCompute computes the brightness scaling for
            %normalizing each basis (i.e., XX, YY , etc).
            %brightnessScaling corresponds to the YYy  basis image formed on the camera for a
            %dipole with \theta=pi/2 and \phi=pi/2

            %set Emitter properties to match YYy channel
            Emitter.polar_para.phiD = pi / 2;
            Emitter.polar_para.thetaD = pi / 2;
            Emitter.position_para.x = 0;
            Emitter.position_para.y = 0;
            Emitter.position_para.z = 0;

            %             [~,brightnessScaling]=obj.simDipole_novotny(obj,Emitter);
            [brightnessScalingX, brightnessScalingY] = obj.simDipole_novotny(obj, Emitter); % 190717 TD
            brightnessScaling = brightnessScalingX + brightnessScalingY;
        end

        %%

        function B = computeBasis(obj, varargin)
            %computeBasis computes the bases images (i.e., XX, YY, etc) corresponding to a
            %dipole.

            %get options
            opt = varargin(2:end);
            s = opt2struct(opt);

            Emitter.position_para.x = 0;
            Emitter.position_para.y = 0;
            Emitter.position_para.z = 0;

            simDipole_novotny_h = @(Emitter)obj.simDipole_novotny(obj, Emitter);

            if isfield(s, 'xx') || isfield(s, 'xy') || isfield(s, 'xz')
                % XX
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = pi / 2;


                [BXXx, BXXy] = simDipole_novotny_h(Emitter);


                if isfield(s, 'x_channel') && isfield(s, 'xx')

                    B = BXXx;

                else
                    B = BXXy;

                end

            end


            % YY
            Emitter.polar_para.phiD = pi / 2;
            Emitter.polar_para.thetaD = pi / 2;

            if isfield(s, 'yy') || isfield(s, 'xy') || isfield(s, 'yz')
                [BYYx, BYYy] = simDipole_novotny_h(Emitter);
                if isfield(s, 'x_channel') && isfield(s, 'yy')

                    B = BYYx;

                else
                    B = BYYy;

                end

            end


            % ZZ
            %             Emitter.polar_para.phiD=pi/2;
            Emitter.polar_para.phiD = 0;
            Emitter.polar_para.thetaD = 0;

            if isfield(s, 'zz') || isfield(s, 'xz') || isfield(s, 'yz')
                [BZZx, BZZy] = simDipole_novotny_h(Emitter);
                if isfield(s, 'x_channel') && isfield(s, 'zz')
                    B = BZZx;

                else
                    B = BZZy;

                end

            end


            % XY
            Emitter.polar_para.phiD = pi / 4;
            Emitter.polar_para.thetaD = pi / 2;

            if isfield(s, 'xy')
                [BXYxt, BXYyt] = simDipole_novotny_h(Emitter);
                if isfield(s, 'x_channel')
                    B = 2 * BXYxt - BXXx - BYYx;

                else
                    B = 2 * BXYyt - BXXy - BYYy;

                end

            end

            % XZ
            Emitter.polar_para.phiD = 0;
            Emitter.polar_para.thetaD = pi / 4;

            if isfield(s, 'xz')
                [BXZxt, BXZyt] = simDipole_novotny_h(Emitter);
                if isfield(s, 'x_channel')
                    B = 2 * BXZxt - BXXx - BZZx;
                else
                    B = 2 * BXZyt - BXXy - BZZy;
                end

            end
            % YZ
            Emitter.polar_para.phiD = pi / 2;
            Emitter.polar_para.thetaD = pi / 4;

            if isfield(s, 'yz')
                [BYZxt, BYZyt] = simDipole_novotny_h(Emitter);
                if isfield(s, 'x_channel')

                    B = 2 * BYZxt - BYYx - BZZx;
                else
                    B = 2 * BYZyt - BYYy - BZZy;
                end
            end

            % crop the basis images to match the desired image size
            %--------------------------------------------------------

            %accounting for photon loss
            brightness_scaling = obj.brightnessScaling;

            N_pupil = size(B, 1);
            up_sample = obj.pixelUpsample;
            img_size = obj.imageSize;

            %handle for corping region of interest
            roi = @(img)img(-up_sample*(img_size - 1)/2+(N_pupil-1)/2+1:1:up_sample*(img_size - 1)/2+(N_pupil-1)/2+1, ... .
                -up_sample*(img_size - 1)/2+(N_pupil-1)/2+1:1:up_sample*(img_size - 1)/2+(N_pupil-1)/2+1, :);

            if isfield(s, 'crop') && s.crop
                sumnorm = sum(sum(roi(brightness_scaling)));

                B = roi(B) / sumnorm;

            else
                sumnorm = sum(sum(brightness_scaling));
                B = B / sumnorm;
            end

        end
    end

    %%

    methods (Static)
        %---------------------------------------------------------
        function B = computeBasis_static(obj)
            %computeBasis computes the bases images (i.e., XX, YY, etc) corresponding to a
            %dipole.

            %get options
           
            Emitter.position_para.x = 0;
            Emitter.position_para.y = 0;
            Emitter.position_para.z = 0;

            simDipole_novotny_h = @(Emitter)obj.simDipole_novotny(obj, Emitter);
            Emitter.polar_para.phiD = 0;
            Emitter.polar_para.thetaD = pi / 2;
            [BXXx, BXXy] = simDipole_novotny_h(Emitter);
            
             N_pupil = size(BXXx, 1);
            up_sample = obj.pixelUpsample;
            img_size = obj.imageSize;
             roi = @(img)img(-up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, ... .
                -up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, :);
            
            BXXx = roi(BXXx);
            BXXy = roi(BXXy);
            BXX = [BXXx,BXXy];

            
            % YY
            Emitter.polar_para.phiD = pi / 2;
            Emitter.polar_para.thetaD = pi / 2; 
            [BYYx, BYYy] = simDipole_novotny_h(Emitter);
            BYYx = roi(BYYx);
            BYYy = roi(BYYy);
            BYY = [BYYx,BYYy];

               
            %             Emitter.polar_para.phiD=pi/2;
            Emitter.polar_para.phiD = 0;
            Emitter.polar_para.thetaD = 0;
            [BZZx, BZZy] = simDipole_novotny_h(Emitter);
            BZZx = roi(BZZx);
            BZZy = roi(BZZy);       
            BZZ = [BZZx,BZZy];

            % XY
            Emitter.polar_para.phiD = pi / 4;
            Emitter.polar_para.thetaD = pi / 2;
            [BXYxt, BXYyt] = simDipole_novotny_h(Emitter);   
            BXYxt = roi(BXYxt);
            BXYyt = roi(BXYyt); 
            BXYx = 2 * BXYxt - BXXx - BYYx;            
            BXYy = 2 * BXYyt - BXXy - BYYy;
                       
            BXY = [BXYx,BXYy];

      

            % XZ
            Emitter.polar_para.phiD = 0;
            Emitter.polar_para.thetaD = pi / 4;
            [BXZxt, BXZyt] = simDipole_novotny_h(Emitter);
            BXZxt = roi(BXZxt);
            BXZyt = roi(BXZyt); 
             BXZx = 2 * BXZxt - BXXx - BZZx;
            BXZy = 2 * BXZyt - BXXy - BZZy;
            BXZ = [BXZx,BXZy];
                    
            Emitter.polar_para.phiD = pi / 2;
            Emitter.polar_para.thetaD = pi / 4;
            [BYZxt, BYZyt] = simDipole_novotny_h(Emitter);
            BYZxt = roi(BYZxt);
            BYZyt = roi(BYZyt);             
            BYZx = 2 * BYZxt - BYYx - BZZx;
            BYZy = 2 * BYZyt - BYYy - BZZy;           
            BYZ = [BYZx,BYZy];
                    
                    

            % crop the basis images to match the desired image size
            %--------------------------------------------------------

          


            %handle for corping region of interest
           

            

             
             B(1,:)=reshape(BXX,1,[]);
             B(2,:)=reshape(BYY,1,[]);
             B(3,:)=reshape(BZZ,1,[]);
             B(4,:)=reshape(BXY,1,[]);
             B(5,:)=reshape(BXZ,1,[]);
             B(6,:)=reshape(BYZ,1,[]);
             

         
        end
        
        function phaseMask_out = mountPhaseMask(obj, varargin)
            s = opt2struct(varargin);
            rho_max = obj.phaseMaskPara.pupilRadius;
            xShift = obj.phaseMaskPara.x_shift_center_position;
            yShift = obj.phaseMaskPara.y_shift_center_position;
            maskRot = obj.phaseMaskPara.maskRotation;
            maskName = obj.phaseMaskPara.maskName;

            maskSize = round((obj.pixelSize)^-1*(obj.emissWavelength * rho_max)/obj.numApt);
            if mod(maskSize, 2) ~= 0
                maskSize = maskSize + 1;
            end

            pmask = imread([maskName, '.bmp']);
            pmask = im2double(pmask,'indexed');
            bias = double(pmask(1, 1));
            %pmask = double(pmask) - bias;
            pmaskSize = size(pmask);
            
            pmask = pmask/255*2*pi-pi;
            %pmask = pmask / bias * pi;


            %Pick up the aperture size of the mask
            mask_nonZeroStartV = find(pmask(:, pmaskSize(2) / 2) ~= 0, 1, 'first');
            mask_nonZeroEndV = find(flipud(pmask(:, pmaskSize(2) / 2)) ~= 0, 1, 'first');
            mask_nonZeroStartH = find(pmask(pmaskSize(1) / 2, :) ~= 0, 1, 'first');
            mask_nonZeroEndH = find(flipud(pmask(pmaskSize(2) / 2, :)) ~= 0, 1, 'first');
            mask_aper = pmaskSize(1) - min((mask_nonZeroStartV-1)+(mask_nonZeroEndV - 1), ...
                (mask_nonZeroStartH - 1)+(mask_nonZeroEndH - 1));

            scaleFactor = 1;%(rho_max + 1) * 2 / mask_aper; % 190814 TD for solving mismatch of BFP size with vectorial forward model
            %             scaleFactor = rho_max*2 / mask_aper;
            newMaskSize = ceil(pmaskSize(1)*scaleFactor);

            if mod(newMaskSize, 2) ~= 0
                newMaskSize = newMaskSize + 1;
                % set the new mask size as even number
            end

            %adjust the phase mask size using nearest neighbor interpolation
            MaskResized = imresize(pmask, [newMaskSize, NaN], 'nearest');

            %amplitude modulation
            if isfield(s, 'amplitudemask')
                nameAmpMask = s.amplitudemask;
                amplitude_mask_struct = load(fullfile('phasemask', nameAmpMask));
                amplitude_mask = imresize(amplitude_mask_struct.xAmpMask_padded, [newMaskSize, NaN], 'nearest');

            else
                amplitude_mask = ones(size(MaskResized));
            end

            if maskRot ~= 0
                MaskResized = rot90(MaskResized, maskRot);
                amplitude_mask = rot90(amplitude_mask, maskRot);
            end

            if xShift > 0
                MaskResized = [zeros(size(MaskResized, 1), 2 * xShift), MaskResized];
                amplitude_mask = [zeros(size(amplitude_mask, 1), 2 * xShift), amplitude_mask];

            elseif xShift < 0
                MaskResized = [MaskResized, zeros(size(MaskResized, 1), 2 * -xShift)];
                amplitude_mask = [amplitude_mask, zeros(size(amplitude_mask, 1), 2 * -xShift)];
            end
            if yShift > 0
                MaskResized = [zeros(2 * yShift, size(MaskResized, 2)); MaskResized];
                amplitude_mask = [zeros(2 * yShift, size(amplitude_mask, 2)); amplitude_mask];
            elseif yShift < 0
                MaskResized = [MaskResized; zeros(2 * -yShift, size(MaskResized, 2))];
                amplitude_mask = [amplitude_mask; zeros(2 * -yShift, size(amplitude_mask, 2))];
            end
            if size(MaskResized, 1) < maskSize
                MaskResized = [zeros(floor((maskSize-size(MaskResized, 1)) / 2), size(MaskResized, 2)); ...
                    MaskResized; zeros(floor((maskSize-size(MaskResized, 1)) / 2), size(MaskResized, 2))];

                amplitude_mask = [zeros(floor((maskSize-size(amplitude_mask, 1)) / 2), size(amplitude_mask, 2)); ...
                    amplitude_mask; zeros(floor((maskSize-size(amplitude_mask, 1)) / 2), size(amplitude_mask, 2))];
            end
            if size(MaskResized, 2) < maskSize
                MaskResized = [zeros(size(MaskResized, 1), floor((maskSize-size(MaskResized, 2)) / 2)), ...
                    MaskResized, zeros(size(MaskResized, 1), floor((maskSize-size(MaskResized, 2)) / 2))];

                amplitude_mask = [zeros(size(amplitude_mask, 1), floor((maskSize-size(amplitude_mask, 2)) / 2)), ...
                    amplitude_mask, zeros(size(amplitude_mask, 1), floor((maskSize-size(amplitude_mask, 2)) / 2))];
            end

            maskNew = MaskResized(size(MaskResized, 1)/2-floor(maskSize / 2)+1:size(MaskResized, 1)/2+floor(maskSize / 2), ...
                size(MaskResized, 2)/2-floor(maskSize / 2)+1:size(MaskResized, 2)/2+floor(maskSize / 2));

            amplitude_maskNew = amplitude_mask(size(amplitude_mask, 1)/2-floor(maskSize / 2)+1:size(amplitude_mask, 1)/2+floor(maskSize / 2), ...
                size(amplitude_mask, 2)/2-floor(maskSize / 2)+1:size(amplitude_mask, 2)/2+floor(maskSize / 2));

            %the initil amplitude was inverse of the true amplitude!
            phaseMask = (amplitude_maskNew) .* exp(1i*maskNew);

            if isfield(s, 'zeroorder')
                phaseMask_x = (1 - s.zeroorder(1)) * phaseMask + s.zeroorder(1) * abs(phaseMask);
                phaseMask_y = (1 - s.zeroorder(2)) * phaseMask + s.zeroorder(2) * abs(phaseMask);
            else
                phaseMask_x = phaseMask;
                phaseMask_y = phaseMask;
            end


            phaseMask_out(:, :, 1) = phaseMask_x;
            phaseMask_out(:, :, 2) = phaseMask_y;

        end
    end

    %% methods for image formation

    %----------------------------------------------------
    methods (Static)

        %%

        function [imgx, imgy, Ex, Ey] = simDipole_novotny(Nanoscope, Emitter, varargin)
            %simDipole_novotny computes

            s = opt2struct(varargin);

            % get position parameters
            %--------------------------------------------------------
            fileds = {'z', 'x', 'y'};
            position_para = Emitter.position_para; % in (nm)
            checkFileds(position_para, fileds); %validate fields
            z = position_para.z;
            deltax = position_para.x;
            deltay = position_para.y;
            if isfield(position_para,'z2')
            z2 = position_para.z2; % distance from the emitter to the interface
            else
                z2 = 0;
            end

            % get molecular orientation parameters
            %--------------------------------------------------------
            fileds = {'phiD', 'thetaD'};
            polar_para = Emitter.polar_para;
            checkFileds(polar_para, fileds); %validate fields
            phiD = polar_para.phiD;
            thetaD = polar_para.thetaD;

            % get emitter and imaging parameters
            %--------------------------------------------------------

            n1 = Nanoscope.refractiveIndx;
            zh = 0; % thickness of film
            %z2 = 0; % distance from the emitter to the interface
            n2 = 1.33; % sample refractive index
            nh = n1; % thin film refractive index
            lambda = Nanoscope.emissWavelength; %wavelength (nm)
            NA = Nanoscope.numApt; %numerical aperture
            pmask = Nanoscope.phaseMask;
            N = size(pmask, 1);
            if mod(N,2)==0
                N = N+1;
                pmask_temp = zeros(size(pmask)+[1,1,0]);
                pmask_temp(1:end-1,1:end-1,:)=pmask;
                pmask = pmask_temp;
                
            end
            pixel_size = Nanoscope.pixelSize; %object pixel size (nm)

            if isfield(s, 'upsampling')
                upsampling = s.upsampling;
            else
                upsampling = 1; %upsampling factor of image space
            end
            molecule_num = numel(deltax);
            %calculate both pupil and image plane sampling,
            %one will affect the other, so make sure not to introduce aliasing

            dx = n1 * (pixel_size / (upsampling)); %  in (nm) due to Abbe sine...
            % condition, scale by imaging medium r.i.
            dv = 1 / (N * dx); %pupil sampling, related to image plane by FFT

            % define pupil coordinates
            %--------------------------------------------------------
            [eta, xi] = meshgrid(((-1 / (2 * dx)) + (1 / (2 * N * dx))):dv:(-(1 / (2 * N * dx)) ...
                +(1 / (2 * dx))), ((-1 / (2 * dx)) + (1 / (2 * N * dx))):dv:(-(1 / (N * 2 * dx)) + (1 / (2 * dx))));
            x = lambda * (eta);
            y = lambda * (xi);
            [phi, rho] = cart2pol(x, y);
            rho_max = NA / n1; %pupil region of support determined by NA and imaging medium r.i.
            k1 = n1 * (2 * pi / lambda);
            kh = nh * (2 * pi / lambda);
            k2 = n2 * (2 * pi / lambda);
            theta1 = asin(rho); %theta in matched medium
            thetah = asin((n1/nh)*sin(theta1)); %theta in thin film
            theta2 = asin((n1/n2)*sin(theta1)); %theta in mismatched medium
            theta2 = real(theta2)-1i*abs(imag(theta2));
            
            % cache fixed terms
            costheta2 = cos(theta2);
            costhetah = cos(thetah);
            costheta1 = cos(theta1);
            sintheta1 = sin(theta1);
            cosphi = cos(phi);
            sinphi = sin(phi);
            % Fresnel coefficients
            %--------------------------------------------------------
            tp_2h = 2 * n2 * costheta2 ./ (n2 * costhetah + nh * costheta2);
            ts_2h = 2 * n2 * costheta2 ./ (nh * costhetah + n2 * costheta2);
            tp_h1 = 2 * nh * costhetah ./ (nh * costheta1 + n1 * costhetah);
            ts_h1 = 2 * nh * costhetah ./ (n1 * costheta1 + nh * costhetah);

            rp_2h = (n2 * costheta2 - nh * costhetah) ./ (n2 * costheta2 + nh * costhetah);
            rs_2h = (nh * costheta2 - n2 * costhetah) ./ (nh * costheta2 + n2 * costhetah);
            rp_h1 = (nh * costhetah - n1 * costheta1) ./ (nh * costhetah + n1 * costheta1);
            rs_h1 = (n1 * costhetah - nh * costheta1) ./ (n1 * costhetah + nh * costheta1);

            % Axelrod's equations for E-fields at pupil plane
            %--------------------------------------------------------
            mux = reshape(sin(thetaD).*cos(phiD), 1, 1, molecule_num);
            muy = reshape(sin(thetaD).*sin(phiD), 1, 1, molecule_num);
            muz = reshape(cos(thetaD), 1, 1, molecule_num);
            tp = tp_2h .* tp_h1 .* exp(1i*kh*costhetah*zh) ./ (1 + rp_2h .* rp_h1 .* exp(2i * kh * zh * costhetah));
            ts = ts_2h .* ts_h1 .* exp(1i*kh*costhetah*zh) ./ (1 + rs_2h .* rs_h1 .* exp(2i * kh * zh * costhetah));

            Es = bsxfun(@times, ts.*(costheta1 ./ costheta2).*(n1 / n2), ...
                (bsxfun(@times, muy, cosphi) - bsxfun(@times, mux, sinphi)));

            Ep = bsxfun(@times, tp, bsxfun(@times, (n1 / n2) .* costheta1, (bsxfun(@times, mux, cosphi) + ...
                bsxfun(@times, muy, sinphi)))- ...
                bsxfun(@times, bsxfun(@times, muz, sintheta1), (n1 / n2)^2 .* (costheta1 ./ costheta2)));

            PupilFilt = (rho < rho_max) .* 1; %casting to numeric


            %computing E-fields
            %--------------------------------------------------------

            E_common = PupilFilt .* (1 ./ sqrt(costheta1)) .* exp(1i*kh*zh*costhetah) .* ...
                exp(1i*k2*z2*costheta2);
            Ey_common = bsxfun(@times, E_common, ...
                (bsxfun(@times, cosphi, Es) + bsxfun(@times, sinphi, Ep)));
            Ex_common = bsxfun(@times, E_common, ...
                (bsxfun(@times, cosphi, Ep) + bsxfun(@times, -sinphi, Es)));


            Ey_1 = exp(1i*k1*(bsxfun(@times, reshape(z, 1, 1, molecule_num), costheta1))); %defocus term at pupil plane
            Ex_1 = exp(1i*k1*(bsxfun(@times, reshape(z, 1, 1, molecule_num), costheta1))); %defocus term at pupil plane

            % aplly channel mismatch


            Ex_2 = exp(1i*k1*(bsxfun(@times, reshape(deltax, 1, 1, molecule_num), sintheta1 .* cosphi))); %phase shift
            Ex_3 = exp(1i*k1*(bsxfun(@times, reshape(-deltay, 1, 1, molecule_num), sintheta1 .* sinphi))); %phase shift

            if isfield(s, 'channel_mismatch')
                deltax = deltax + s.channel_mismatch(1);
                deltay = deltay + s.channel_mismatch(2);

            end

            Ey_2 = exp(1i*k1*(bsxfun(@times, reshape(deltax, 1, 1, molecule_num), sintheta1 .* cosphi))); %phase shift
            Ey_3 = exp(1i*k1*(bsxfun(@times, reshape(-deltay, 1, 1, molecule_num), sintheta1 .* sinphi))); %phase shift


            Ey_t = Ey_1 .* Ey_2 .* Ey_3;
            Ex_t = Ex_1 .* Ex_2 .* Ex_3;
            Ex = bsxfun(@times, Ex_t, Ex_common);
            Ey = bsxfun(@times, Ey_t, Ey_common);

            % for propagation from pupil plane E-field to image plane via tube-lens, paraxial
            % approximation is in force.
            %--------------------------------------------------------


            %
            %             pmask_x=pmask(:,:,1);
            %             pmask_y=pmask(:,:,2);
            %             pmaskRot_y=rot90(pmask_y,2);
            %             pmaskRot_x=rot90(pmask_x,1);
            %
            %             imgy = fftshift(fft2(ifftshift(Ey.*repmat((pmaskRot_y),1,1,molecule_num))));
            %             imgx= fliplr((fftshift(fft2(ifftshift(Ex.*repmat((pmaskRot_x),1,1,molecule_num))))));


            % 1- rotate electric field in x channel
            Ex_rotated = rot90(Ex);
            % 2- flip the  electric field in y channel
            Ey_flipped = fliplr(Ey);

            pmask_x = pmask(:, :, 1);
            pmask_y = pmask(:, :, 2);

            imgy = flipud(fftshift(fft2(ifftshift(Ey_flipped .* repmat(pmask_y, 1, 1, molecule_num)))));
            imgx = rot90(fftshift(fft2(ifftshift(Ex_rotated .* repmat(pmask_x, 1, 1, molecule_num))))', 2);

            % flip the y image left to right to mirror the x image

            imgy = fliplr(imgy);

            % image on the camera is the amplitude squared of the electric field
            %--------------------------------------------------------
            imgy = abs(imgy).^2;
            imgx = abs(imgx).^2;
        end

        %%

        function [imgx, imgy, Ex, Ey] = simDipole_novotny_costum(Nanoscope, Emitter, pmask, varargin)
            %simDipole_novotny computes

            s = opt2struct(varargin);

            % get position parameters
            %--------------------------------------------------------
            fileds = {'z', 'x', 'y'};
            position_para = Emitter.position_para; % in (nm)
            checkFileds(position_para, fileds); %validate fields
            z = position_para.z;
            deltax = position_para.x;
            deltay = position_para.y;

            % get molecular orientation parameters
            %--------------------------------------------------------
            fileds = {'phiD', 'thetaD'};
            polar_para = Emitter.polar_para;
            checkFileds(polar_para, fileds); %validate fields
            phiD = polar_para.phiD;
            thetaD = polar_para.thetaD;

            % get emitter and imaging parameters
            %--------------------------------------------------------

            n1 = Nanoscope.refractiveIndx;
            zh = 0; % thickness of film
            z2 = 0; % distance from the emitter to the interface
            n2 = Nanoscope.refractiveIndxSam; % sample refractive index
            nh = n1; % thin film refractive index
            lambda = Nanoscope.emissWavelength; %wavelength (nm)
            NA = Nanoscope.numApt; %numerical aperture
            N = size(pmask, 1);
            pixel_size = Nanoscope.pixelSize; %object pixel size (nm)
            
            if mod(N,2)==0
                N = N+1;
                pmask_temp = zeros(size(pmask)+[1,1,0]);
                pmask_temp(1:end-1,1:end-1,:)=pmask;
                pmask = pmask_temp;
                
            end

            if isfield(s, 'upsampling')
                upsampling = s.upsampling;
            else
                upsampling = 1; %upsampling factor of image space
            end
            molecule_num = numel(deltax);
            %calculate both pupil and image plane sampling,
            %one will affect the other, so make sure not to introduce aliasing

            dx = n1 * (pixel_size / (upsampling)); %  in (nm) due to Abbe sine...
            % condition, scale by imaging medium r.i.
            dv = 1 / (N * dx); %pupil sampling, related to image plane by FFT

            % define pupil coordinates
            %--------------------------------------------------------
            [eta, xi] = meshgrid(((-1 / (2 * dx)) + (1 / (2 * N * dx))):dv:(-(1 / (2 * N * dx)) ...
                +(1 / (2 * dx))), ((-1 / (2 * dx)) + (1 / (2 * N * dx))):dv:(-(1 / (N * 2 * dx)) + (1 / (2 * dx))));
            x = lambda * (eta);
            y = lambda * (xi);
            [phi, rho] = cart2pol(x, y);
            rho_max = NA / n1; %pupil region of support determined by NA and imaging medium r.i.
            k1 = n1 * (2 * pi / lambda);
            kh = nh * (2 * pi / lambda);
            k2 = n2 * (2 * pi / lambda);
            theta1 = asin(rho); %theta in matched medium
            thetah = asin((n1/nh)*sin(theta1)); %theta in thin film
            theta2 = asin((n1/n2)*sin(theta1)); %theta in mismatched medium

            % cache fixed terms
            costheta2 = cos(theta2);
            costhetah = cos(thetah);
            costheta1 = cos(theta1);
            sintheta1 = sin(theta1);
            cosphi = cos(phi);
            sinphi = sin(phi);
            % Fresnel coefficients
            %--------------------------------------------------------
            tp_2h = 2 * n2 * costheta2 ./ (n2 * costhetah + nh * costheta2);
            ts_2h = 2 * n2 * costheta2 ./ (nh * costhetah + n2 * costheta2);
            tp_h1 = 2 * nh * costhetah ./ (nh * costheta1 + n1 * costhetah);
            ts_h1 = 2 * nh * costhetah ./ (n1 * costheta1 + nh * costhetah);

            rp_2h = (n2 * costheta2 - nh * costhetah) ./ (n2 * costheta2 + nh * costhetah);
            rs_2h = (nh * costheta2 - n2 * costhetah) ./ (nh * costheta2 + n2 * costhetah);
            rp_h1 = (nh * costhetah - n1 * costheta1) ./ (nh * costhetah + n1 * costheta1);
            rs_h1 = (n1 * costhetah - nh * costheta1) ./ (n1 * costhetah + nh * costheta1);

            % Axelrod's equations for E-fields at pupil plane
            %--------------------------------------------------------
            mux = reshape(sin(thetaD).*cos(phiD), 1, 1, molecule_num);
            muy = reshape(sin(thetaD).*sin(phiD), 1, 1, molecule_num);
            muz = reshape(cos(thetaD), 1, 1, molecule_num);
            tp = tp_2h .* tp_h1 .* exp(1i*kh*costhetah*zh) ./ (1 + rp_2h .* rp_h1 .* exp(2i * kh * zh * costhetah));
            ts = ts_2h .* ts_h1 .* exp(1i*kh*costhetah*zh) ./ (1 + rs_2h .* rs_h1 .* exp(2i * kh * zh * costhetah));

            Es = bsxfun(@times, ts.*(costheta1 ./ costheta2).*(n1 / n2), ...
                (bsxfun(@times, muy, cosphi) - bsxfun(@times, mux, sinphi)));

            Ep = bsxfun(@times, tp, bsxfun(@times, (n1 / n2) .* costheta1, (bsxfun(@times, mux, cosphi) + ...
                bsxfun(@times, muy, sinphi)))- ...
                bsxfun(@times, bsxfun(@times, muz, sintheta1), (n1 / n2)^2 .* (costheta1 ./ costheta2)));

            PupilFilt = (rho < rho_max) .* 1; %casting to numeric


            %computing E-fields
            %--------------------------------------------------------

            E_common = PupilFilt .* (1 ./ sqrt(costheta1)) .* exp(1i*kh*zh*costhetah) .* ...
                exp(1i*k2*z2*costheta2);
            Ey_common = bsxfun(@times, E_common, ...
                (bsxfun(@times, cosphi, Es) + bsxfun(@times, sinphi, Ep)));
            Ex_common = bsxfun(@times, E_common, ...
                (bsxfun(@times, cosphi, Ep) + bsxfun(@times, -sinphi, Es)));


            Ey_1 = exp(1i*k1*(bsxfun(@times, reshape(z, 1, 1, molecule_num), costheta1))); %defocus term at pupil plane
            Ex_1 = exp(1i*k1*(bsxfun(@times, reshape(z, 1, 1, molecule_num), costheta1))); %defocus term at pupil plane

            % aplly channel mismatch


            Ex_2 = exp(1i*k1*(bsxfun(@times, reshape(-deltax, 1, 1, molecule_num), sintheta1 .* cosphi))); %phase shift
            Ex_3 = exp(1i*k1*(bsxfun(@times, reshape(deltay, 1, 1, molecule_num), sintheta1 .* sinphi))); %phase shift

            if isfield(s, 'channel_mismatch')
                deltax = deltax + s.channel_mismatch(1);
                deltay = deltay + s.channel_mismatch(2);

            end

            Ey_2 = exp(1i*k1*(bsxfun(@times, reshape(deltax, 1, 1, molecule_num), sintheta1 .* cosphi))); %phase shift
            Ey_3 = exp(1i*k1*(bsxfun(@times, reshape(deltay, 1, 1, molecule_num), sintheta1 .* sinphi))); %phase shift


            Ey_t = Ey_1 .* Ey_2 .* Ey_3;
            Ex_t = Ex_1 .* Ex_2 .* Ex_3;
            Ex = bsxfun(@times, Ex_t, Ex_common);
            Ey = bsxfun(@times, Ey_t, Ey_common);

            % for propagation from pupil plane E-field to image plane via tube-lens, paraxial
            % approximation is in force.
            %--------------------------------------------------------

            %apply a zero order effect
            %             amp=abs(pmask);
            %             phase_t=angle(pmask);
            %             pmask=.15*amp+.85*amp.*exp(1i*phase_t);
            pmaskRot = rot90(pmask, -1);
            imgy = (fftshift(fft2(ifftshift(Ey .* repmat((pmask), 1, 1, molecule_num)))));
            imgx = (fftshift(fft2(ifftshift(Ex .* repmat((pmaskRot), 1, 1, molecule_num)))));
            % image on the camera is the amplitude squared of the electric field
            %--------------------------------------------------------
            imgy = abs(imgy).^2;
            imgx = abs(imgx).^2;
        end

        %%

        function [BXXx, BYYx, BZZx, BXYx, BXZx, BYZx, ...
                BXXy, BYYy, BZZy, BXYy, BXZy, BYZy] = computeBases(Nanoscope, Emitter, varargin)
            %computeBases computes

            num_molecules = numel(Emitter.position_para.x);
            %number of molecules

            simDipole_novotny_h = @(Emitter)Nanoscope.simDipole_novotny(Nanoscope, Emitter, varargin{:});
            % XX

            Emitter.polar_para.phiD = zeros(num_molecules, 1);
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BXXx, BXXy] = simDipole_novotny_h(Emitter);

            % YY
            Emitter.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BYYx, BYYy] = simDipole_novotny_h(Emitter);

            % ZZ
            % Emitter.polar_para.phiD=ones(num_molecules,1)*pi/2;
            Emitter.polar_para.phiD = ones(num_molecules, 1) * 0;
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * 0;

            [BZZx, BZZy] = simDipole_novotny_h(Emitter);

            % XY
            Emitter.polar_para.phiD = ones(num_molecules, 1) * pi / 4;
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BXYxt, BXYyt] = simDipole_novotny_h(Emitter);

            % XZ
            Emitter.polar_para.phiD = ones(num_molecules, 1) * 0;
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * pi / 4;

            [BXZxt, BXZyt] = simDipole_novotny_h(Emitter);

            % YZ
            Emitter.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter.polar_para.thetaD = ones(num_molecules, 1) * pi / 4;


            [BYZxt, BYZyt] = simDipole_novotny_h(Emitter);

            BXYx = 2 * BXYxt - BXXx - BYYx;
            BXZx = 2 * BXZxt - BXXx - BZZx;
            BYZx = 2 * BYZxt - BYYx - BZZx;

            BXYy = 2 * BXYyt - BXXy - BYYy;
            BXZy = 2 * BXZyt - BXXy - BZZy;
            BYZy = 2 * BYZyt - BYYy - BZZy;

        end

        %%

        function [FPSFx, FPSFy, lateral_grid_p] = PSF_Fourier_tf(obj)

            % compute Fourier transforms
            % ---------------------------------------------------
            up_sample = obj.pixelUpsample;
            FPSFy = struct();
            FPSFx = struct();

            %y_channel
            FPSFy.FXXy = single(fft2(ifftshift(up_sample^2 * obj.XXyBasis)));
            FPSFy.FYYy = single(fft2(ifftshift(up_sample^2 * obj.YYyBasis)));
            FPSFy.FZZy = single(fft2(ifftshift(up_sample^2 * obj.ZZyBasis)));
            FPSFy.FXYy = single(fft2(ifftshift(up_sample^2 * obj.XYyBasis)));
            FPSFy.FXZy = single(fft2(ifftshift(up_sample^2 * obj.XZyBasis)));
            FPSFy.FYZy = single(fft2(ifftshift(up_sample^2 * obj.YZyBasis)));

            %x_channel
            FPSFx.FXXx = single(fft2(ifftshift(up_sample^2 * (obj.XXxBasis))));
            FPSFx.FYYx = single(fft2(ifftshift(up_sample^2 * (obj.YYxBasis))));
            FPSFx.FZZx = single(fft2(ifftshift(up_sample^2 * (obj.ZZxBasis))));
            FPSFx.FXYx = single(fft2(ifftshift(up_sample^2 * (obj.XYxBasis))));
            FPSFx.FXZx = single(fft2(ifftshift(up_sample^2 * (obj.XZxBasis))));
            FPSFx.FYZx = single(fft2(ifftshift(up_sample^2 * (obj.YZxBasis))));

            % compute  lateral grid points
            %----------------------------------------------------
            pixel_size = obj.pixelSize;
            img_size = obj.imageSize;
            lateral_grid_p = (-(img_size - 1) / 2:1 / up_sample:(img_size - 1) / 2) * pixel_size;

        end


        function [FPSFx, FPSFy] = createPSFstruct(obj, varargin)

            %extract input options and set parameters
            %---------------------------------------------------------
            s = opt2struct(varargin);

            %determine upsampling
            if isfield(s, 'upsampling')
                up_sample = s.upsampling;
            else
                up_sample = obj.pixelUpsample;
            end

            %determine image size
            if isfield(s, 'imagesize')

                imageSize = s.imagesize;
            else
                imageSize = obj.imageSize;

            end

            %determine channel transmission ratio

            if isfield(s, 'ytoxchanneltransratio')

                yToxChanTransRatio = s.ytoxchanneltransratio;
            else
                yToxChanTransRatio = 1;

            end
            
            if isfield(s, 'normal_focal_plane')

                normal_focal_plane = s.normal_focal_plane;
            else
                normal_focal_plane = 0;

            end
            
            if isfield(s, 'molecule_plane')

                molecule_plane = s.molecule_plane;
            else
                molecule_plane = 0;

            end

            %deltax for computing gradients of PSFs
            deltax = 10^-2; %in nm

            % output structures for PSFs
            FPSFy = struct();
            FPSFx = struct();

            %define a handle
            simDipole_novotny_h = @(Emitter) obj.simDipole_novotny(obj, ...
                Emitter, 'upsampling', up_sample);

            for ii = 1:length(molecule_plane)
                %XX
                %---------------------------------------------------------
                %set molecular orientation
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = pi / 2;


                %set position parameters
                Emitter.position_para.x = 0;
                Emitter.position_para.y = 0;
                Emitter.position_para.z = normal_focal_plane;
                Emitter.position_para.z2 = molecule_plane(ii);


                [imgx, imgy] = simDipole_novotny_h(Emitter);

                %gradient along x
                Emitter.position_para.x = -0 / 2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);

                Emitter.position_para.x = deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %gradient along y
                %                 Emitter.position_para.y=-deltax/2;
                [imgxdy1, imgydy1] = simDipole_novotny_h(Emitter);

                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;


                imgXXx = imgx;
                imgXXy = imgy;
                imgXXxdx = (imgxdx2 - imgxdx1) / deltax;
                imgXXydx = (imgydx2 - imgydx1) / deltax;
                imgXXxdy = (imgxdy2 - imgxdy1) / deltax;
                imgXXydy = (imgydy2 - imgydy1) / deltax;
                %YY
                %---------------------------------------------------------
                %set molecular orientation
                Emitter.polar_para.phiD = pi / 2;
                Emitter.polar_para.thetaD = pi / 2;

                %set position parameters
                Emitter.position_para.x = 0;
                Emitter.position_para.y = 0;

                [imgx, imgy] = simDipole_novotny_h(Emitter);
                %                 Emitter.position_para.x=-deltax/2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = +deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %                 Emitter.position_para.y=-deltax/2;

                [imgxdy1, imgydy1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;

                imgYYx = imgx;
                imgYYy = imgy;
                imgYYxdx = (imgxdx2 - imgxdx1) / deltax;
                imgYYydx = (imgydx2 - imgydx1) / deltax;
                imgYYxdy = (imgxdy2 - imgxdy1) / deltax;
                imgYYydy = (imgydy2 - imgydy1) / deltax;

                % ZZ
                %---------------------------------------------------------
                %set molecular orientation
                %             Emitter.polar_para.phiD=pi/2;
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = 0;


                [imgx, imgy] = simDipole_novotny_h(Emitter);
                %                 Emitter.position_para.x=-deltax/2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %                 Emitter.position_para.y=-deltax/2;

                [imgxdy1, imgydy1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;

                imgZZx = imgx;
                imgZZy = imgy;
                imgZZxdx = (imgxdx2 - imgxdx1) / deltax;
                imgZZydx = (imgydx2 - imgydx1) / deltax;
                imgZZxdy = (imgxdy2 - imgxdy1) / deltax;
                imgZZydy = (imgydy2 - imgydy1) / deltax;

                % XY
                Emitter.polar_para.phiD = pi / 4;
                Emitter.polar_para.thetaD = pi / 2;

                [imgXYxt, imgXYyt] = simDipole_novotny_h(Emitter);

                imgXYx = 2 * imgXYxt - imgXXx - imgYYx;
                imgXYy = 2 * imgXYyt - imgXXy - imgYYy;

                % XZ
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = pi / 4;


                [imgXZxt, imgXZyt] = simDipole_novotny_h(Emitter);

                imgXZx = 2 * imgXZxt - imgXXx - imgZZx;
                imgXZy = 2 * imgXZyt - imgXXy - imgZZy;


                % YZ
                Emitter.polar_para.phiD = pi / 2;
                Emitter.polar_para.thetaD = pi / 4;

                [imgYZxt, imgYZyt] = simDipole_novotny_h(Emitter);

                imgYZx = 2 * imgYZxt - imgYYx - imgZZx;
                imgYZy = 2 * imgYZyt - imgYYy - imgZZy;

                %crop and normalize
                %---------------------------------------------------------

                %handle for croping region of interest

                N_pupil = size(imgYZx, 1);

                roi = @(img)img(-up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1:1:up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1, ... .
                    -up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1:1:up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1, :);

                sumNormalize = sum(sum(roi(obj.brightnessScaling)));
                if ii==1
                    %x_channel
                    %---------------------------------------------------------
                    imgXXx_crop = bsxfun(@times, roi(imgXXx), 1./sumNormalize);
                    %gradient along x
                    imgXXxdx_crop = bsxfun(@times, roi(imgXXxdx), 1./sumNormalize);
                    %gradient along y
                    imgXXxdy_crop = bsxfun(@times, roi(imgXXxdy), 1./sumNormalize);

                    imgYYx_crop = bsxfun(@times, roi(imgYYx), 1./sumNormalize);
                    %gradient along x
                    imgYYxdx_crop = bsxfun(@times, roi(imgYYxdx), 1./sumNormalize);
                    %gradient along y
                    imgYYxdy_crop = bsxfun(@times, roi(imgYYxdy), 1./sumNormalize);

                    imgZZx_crop = bsxfun(@times, roi(imgZZx), 1./sumNormalize);
                    %gradient along x
                    imgZZxdx_crop = bsxfun(@times, roi(imgZZxdx), 1./sumNormalize);
                    %gradient along y
                    imgZZxdy_crop = bsxfun(@times, roi(imgZZxdy), 1./sumNormalize);

                    imgXYx_crop = bsxfun(@times, roi(imgXYx), 1./sumNormalize);
                    imgXZx_crop = bsxfun(@times, roi(imgXZx), 1./sumNormalize);
                    imgYZx_crop = bsxfun(@times, roi(imgYZx), 1./sumNormalize);

                    %y_channel

                    sumNormalize = (1 / yToxChanTransRatio) * sumNormalize;
                    %---------------------------------------------------------
                    imgXXy_crop = bsxfun(@times, roi(imgXXy), 1./sumNormalize);
                    %gradient along x
                    imgXXydx_crop = bsxfun(@times, roi(imgXXydx), 1./sumNormalize);
                    %gradient along y
                    imgXXydy_crop = bsxfun(@times, roi(imgXXydy), 1./sumNormalize);

                    imgYYy_crop = bsxfun(@times, roi(imgYYy), 1./sumNormalize);
                    %gradient along x
                    imgYYydx_crop = bsxfun(@times, roi(imgYYydx), 1./sumNormalize);
                    %gradient along y
                    imgYYydy_crop = bsxfun(@times, roi(imgYYydy), 1./sumNormalize);

                    imgZZy_crop = bsxfun(@times, roi(imgZZy), 1./sumNormalize);
                    %gradient along x
                    imgZZydx_crop = bsxfun(@times, roi(imgZZydx), 1./sumNormalize);
                    %gradient along y
                    imgZZydy_crop = bsxfun(@times, roi(imgZZydy), 1./sumNormalize);

                    imgXYy_crop = bsxfun(@times, roi(imgXYy), 1./sumNormalize);
                    imgXZy_crop = bsxfun(@times, roi(imgXZy), 1./sumNormalize);
                    imgYZy_crop = bsxfun(@times, roi(imgYZy), 1./sumNormalize);
                else
                                        %x_channel
                    %---------------------------------------------------------
                    imgXXx_crop = cat(3,imgXXx_crop,bsxfun(@times, roi(imgXXx), 1./sumNormalize));
                    %gradient along x
                    imgXXxdx_crop = cat(3,imgXXxdx_crop,bsxfun(@times, roi(imgXXxdx), 1./sumNormalize));
                    %gradient along y
                    imgXXxdy_crop = cat(3,imgXXxdy_crop,bsxfun(@times, roi(imgXXxdy), 1./sumNormalize));

                    imgYYx_crop = cat(3,imgYYx_crop,bsxfun(@times, roi(imgYYx), 1./sumNormalize));
                    %gradient along x
                    imgYYxdx_crop = cat(3,imgYYxdx_crop,bsxfun(@times, roi(imgYYxdx), 1./sumNormalize));
                    %gradient along y
                    imgYYxdy_crop = cat(3,imgYYxdy_crop,bsxfun(@times, roi(imgYYxdy), 1./sumNormalize));

                    imgZZx_crop = cat(3,imgZZx_crop,bsxfun(@times, roi(imgZZx), 1./sumNormalize));
                    %gradient along x
                    imgZZxdx_crop = cat(3,imgZZxdx_crop,bsxfun(@times, roi(imgZZxdx), 1./sumNormalize));
                    %gradient along y
                    imgZZxdy_crop = cat(3,imgZZxdy_crop,bsxfun(@times, roi(imgZZxdy), 1./sumNormalize));

                    imgXYx_crop = cat(3,imgXYx_crop,bsxfun(@times, roi(imgXYx), 1./sumNormalize));
                    imgXZx_crop = cat(3,imgXZx_crop,bsxfun(@times, roi(imgXZx), 1./sumNormalize));
                    imgYZx_crop = cat(3,imgYZx_crop,bsxfun(@times, roi(imgYZx), 1./sumNormalize));

                    %y_channel

                    sumNormalize = (1 / yToxChanTransRatio) * sumNormalize;
                    %---------------------------------------------------------
                    imgXXy_crop = cat(3,imgXXy_crop,bsxfun(@times, roi(imgXXy), 1./sumNormalize));
                    %gradient along x
                    imgXXydx_crop = cat(3,imgXXydx_crop,bsxfun(@times, roi(imgXXydx), 1./sumNormalize));
                    %gradient along y
                    imgXXydy_crop = cat(3,imgXXydy_crop,bsxfun(@times, roi(imgXXydy), 1./sumNormalize));

                    imgYYy_crop = cat(3,imgYYy_crop,bsxfun(@times, roi(imgYYy), 1./sumNormalize));
                    %gradient along x
                    imgYYydx_crop = cat(3,imgYYydx_crop,bsxfun(@times, roi(imgYYydx), 1./sumNormalize));
                    %gradient along y
                    imgYYydy_crop = cat(3,imgYYydy_crop,bsxfun(@times, roi(imgYYydy), 1./sumNormalize));

                    imgZZy_crop = cat(3,imgZZy_crop,bsxfun(@times, roi(imgZZy), 1./sumNormalize));
                    %gradient along x
                    imgZZydx_crop = cat(3,imgZZydx_crop,bsxfun(@times, roi(imgZZydx), 1./sumNormalize));
                    %gradient along y
                    imgZZydy_crop = cat(3,imgZZydy_crop,bsxfun(@times, roi(imgZZydy), 1./sumNormalize));

                    imgXYy_crop = cat(3,imgXYy_crop,bsxfun(@times, roi(imgXYy), 1./sumNormalize));
                    imgXZy_crop = cat(3,imgXZy_crop,bsxfun(@times, roi(imgXZy), 1./sumNormalize));
                    imgYZy_crop = cat(3,imgYZy_crop,bsxfun(@times, roi(imgYZy), 1./sumNormalize));
                end

            end
            %combine basis image
             %---------------------------------------------------------
            if length(molecule_plane)>1
            imgXXx = mean(imgXXx_crop,3);
            %gradient along x
            imgXXxdx = mean(imgXXxdx_crop,3);
            %gradient along y
            imgXXxdy = mean(imgXXxdy_crop,3);

            imgYYx = mean(imgYYx_crop,3);
            %gradient along x
            imgYYxdx = mean(imgYYxdx_crop,3);
            %gradient along y
            imgYYxdy = mean(imgYYxdy_crop,3);

            imgZZx = mean(imgZZx_crop,3);
            %gradient along x
            imgZZxdx = mean(imgZZxdx_crop,3);
            %gradient along y
            imgZZxdy = mean(imgZZxdy_crop,3);

            imgXYx = mean(imgXYx_crop,3);
            imgXZx = mean(imgXZx_crop,3);
            imgYZx = mean(imgYZx_crop,3);

            %y_channel
            %---------------------------------------------------------
            imgXXy = mean(imgXXy_crop,3);
            %gradient along x
            imgXXydx = mean(imgXXydx_crop,3);
            %gradient along y
            imgXXydy = mean(imgXXydy_crop,3);

            imgYYy = mean(imgYYy_crop,3);
            %gradient along x
            imgYYydx = mean(imgYYydx_crop,3);
            %gradient along y
            imgYYydy = mean(imgYYydy_crop,3);

            imgZZy = mean(imgZZy_crop,3);
            %gradient along x
            imgZZydx = mean(imgZZydx_crop,3);
            %gradient along y
            imgZZydy = mean(imgZZydy_crop,3);

            imgXYy = mean(imgXYy_crop,3);
            imgXZy = mean(imgXZy_crop,3);
            imgYZy = mean(imgYZy_crop,3);
            else
            imgXXx = imgXXx_crop;
            %gradient along x
            imgXXxdx = imgXXxdx_crop;
            %gradient along y
            imgXXxdy = imgXXxdy_crop;

            imgYYx = imgYYx_crop;
            %gradient along x
            imgYYxdx = imgYYxdx_crop;
            %gradient along y
            imgYYxdy = imgYYxdy_crop;

            imgZZx = imgZZx_crop;
            %gradient along x
            imgZZxdx = imgZZxdx_crop;
            %gradient along y
            imgZZxdy = imgZZxdy_crop;

            imgXYx = imgXYx_crop;
            imgXZx = imgXZx_crop;
            imgYZx = imgYZx_crop;

            %y_channel
            %---------------------------------------------------------
            imgXXy = imgXXy_crop;
            %gradient along x
            imgXXydx = imgXXydx_crop;
            %gradient along y
            imgXXydy = imgXXydy_crop;

            imgYYy = imgYYy_crop;
            %gradient along x
            imgYYydx = imgYYydx_crop;
            %gradient along y
            imgYYydy = imgYYydy_crop;

            imgZZy = imgZZy_crop;
            %gradient along x
            imgZZydx = imgZZydx_crop;
            %gradient along y
            imgZZydy = imgZZydy_crop;

            imgXYy = imgXYy_crop;
            imgXZy = imgXZy_crop;
            imgYZy = imgYZy_crop;
            end
            %compute FFTs (single precision)
            %---------------------------------------------------------

            %x_channel
            FPSFx.FXXx = single(fft2((fftshift(up_sample^2 * imgXXx))));
            FPSFx.FYYx = single(fft2((fftshift(up_sample^2 * imgYYx))));
            FPSFx.FZZx = single(fft2((fftshift(up_sample^2 * imgZZx))));

            %gradients
            FPSFx.FXXxdx = single(fft2((fftshift(10^2 * up_sample^2 * imgXXxdx))));
            FPSFx.FXXxdy = single(fft2((fftshift(10^2 * up_sample^2 * imgXXxdy))));
            FPSFx.FYYxdx = single(fft2((fftshift(10^2 * up_sample^2 * imgYYxdx))));
            FPSFx.FYYxdy = single(fft2((fftshift(10^2 * up_sample^2 * imgYYxdy))));
            FPSFx.FZZxdx = single(fft2((fftshift(10^2 * up_sample^2 * imgZZxdx))));
            FPSFx.FZZxdy = single(fft2((fftshift(10^2 * up_sample^2 * imgZZxdy))));

            FPSFx.FXYx = single(fft2((fftshift(up_sample^2 * imgXYx))));
            FPSFx.FXZx = single(fft2((fftshift(up_sample^2 * imgXZx))));
            FPSFx.FYZx = single(fft2((fftshift(up_sample^2 * imgYZx))));

            %y_channel
            FPSFy.FXXy = single(fft2((fftshift(up_sample^2 * imgXXy))));
            FPSFy.FYYy = single(fft2((fftshift(up_sample^2 * imgYYy))));
            FPSFy.FZZy = single(fft2((fftshift(up_sample^2 * imgZZy))));

            %gradients
            FPSFy.FXXydx = single(fft2((fftshift(10^2 * up_sample^2 * imgXXydx))));
            FPSFy.FXXydy = single(fft2((fftshift(10^2 * up_sample^2 * imgXXydy))));
            FPSFy.FYYydx = single(fft2((fftshift(10^2 * up_sample^2 * imgYYydx))));
            FPSFy.FYYydy = single(fft2((fftshift(10^2 * up_sample^2 * imgYYydy))));
            FPSFy.FZZydx = single(fft2((fftshift(10^2 * up_sample^2 * imgZZydx))));
            FPSFy.FZZydy = single(fft2((fftshift(10^2 * up_sample^2 * imgZZydy))));

            FPSFy.FXYy = single(fft2((fftshift(up_sample^2 * imgXYy))));
            FPSFy.FXZy = single(fft2((fftshift(up_sample^2 * imgXZy))));
            FPSFy.FYZy = single(fft2((fftshift(up_sample^2 * imgYZy))));

        end
        
   
        function [PSFx, PSFy] = createPSFstruct3D(obj, varargin)

            %extract input options and set parameters
            %---------------------------------------------------------
            s = opt2struct(varargin);

            %determine upsampling
            if isfield(s, 'upsampling')
                up_sample = s.upsampling;
            else
                up_sample = obj.pixelUpsample;
            end

            %determine image size
            if isfield(s, 'imagesize')

                imageSize = s.imagesize;
            else
                imageSize = obj.imageSize;

            end

            %determine channel transmission ratio

            if isfield(s, 'ytoxchanneltransratio')

                yToxChanTransRatio = s.ytoxchanneltransratio;
            else
                yToxChanTransRatio = 1;

            end
            
            if isfield(s, 'number_axial_pixels')

                number_axial_pixels = s.number_axial_pixels;

            else
                number_axial_pixels = 1;

            end
            
            
            if isfield(s, 'axial_pixel_size')

                axial_pixel_size = s.axial_pixel_size; %in unit of m
            else
                axial_pixel_size = 0;

            end
            
            if isfield(s, 'normal_focal_plane')

                normal_focal_plane = s.normal_focal_plane; %in unit of m
            else
                normal_focal_plane = 0;

            end
            
            if isfield(s, 'axial_grid_points')

                axial_grid_points = s.axial_grid_points; %in unit of m
            else
                axial_grid_points = 0;

            end
            
            z_position = axial_grid_points;

            %deltax for computing gradients of PSFs
            deltax = 10^-2; %in nm

            % output structures for PSFs
            PSFy = struct();
            PSFx = struct();
            for ii = 1:length(z_position)
                z_cur = z_position(ii);
                %define a handle
                simDipole_novotny_h = @(Emitter) obj.simDipole_novotny(obj, ...
                    Emitter, 'upsampling', up_sample);

                %XX
                %---------------------------------------------------------
                %set molecular orientation
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = pi / 2;


                %set position parameters
                Emitter.position_para.x = 0;
                Emitter.position_para.y = 0;
                Emitter.position_para.z = normal_focal_plane;
                Emitter.position_para.z2 = z_cur;


                [imgx, imgy] = simDipole_novotny_h(Emitter);

                %gradient along x
                Emitter.position_para.x = -0 / 2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);

                Emitter.position_para.x = deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %gradient along y
                imgxdy1 = imgxdx1;
                imgydy1 = imgydx1;

                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;
                
                %gradient along z
                imgxdz1 = imgxdx1;
                imgydz1 = imgydx1;

                Emitter.position_para.z2 = z_cur+deltax;
                [imgxdz2, imgydz2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.z2 = z_cur;


                imgXXx = imgx;
                imgXXy = imgy;
                imgXXxdx = (imgxdx2 - imgxdx1) / deltax;
                imgXXydx = (imgydx2 - imgydx1) / deltax;
                imgXXxdy = (imgxdy2 - imgxdy1) / deltax;
                imgXXydy = (imgydy2 - imgydy1) / deltax;
                imgXXxdz = (imgxdz2 - imgxdz1) / deltax;
                imgXXydz = (imgydz2 - imgydz1) / deltax;
                %YY
                %---------------------------------------------------------
                %set molecular orientation
                Emitter.polar_para.phiD = pi / 2;
                Emitter.polar_para.thetaD = pi / 2;

                %set position parameters
                Emitter.position_para.x = 0;
                Emitter.position_para.y = 0;
                Emitter.position_para.z2 = z_cur;

                [imgx, imgy] = simDipole_novotny_h(Emitter);
                %                 Emitter.position_para.x=-deltax/2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = +deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %                 Emitter.position_para.y=-deltax/2;

                imgxdy1 = imgxdx1;
                imgydy1 = imgydx1;
                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;
                
                %gradient along z
                imgxdz1 = imgxdx1;
                imgydz1 = imgydx1;

                Emitter.position_para.z2 = z_cur+deltax;
                [imgxdz2, imgydz2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.z2 = z_cur;

                %
                imgYYx = imgx;
                imgYYy = imgy;
                imgYYxdx = (imgxdx2 - imgxdx1) / deltax;
                imgYYydx = (imgydx2 - imgydx1) / deltax;
                imgYYxdy = (imgxdy2 - imgxdy1) / deltax;
                imgYYydy = (imgydy2 - imgydy1) / deltax;
                imgYYxdz = (imgxdz2 - imgxdz1) / deltax;
                imgYYydz = (imgydz2 - imgydz1) / deltax;

                % ZZ
                %---------------------------------------------------------
                %set molecular orientation
                %             Emitter.polar_para.phiD=pi/2;
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = 0;


                [imgx, imgy] = simDipole_novotny_h(Emitter);
                %                 Emitter.position_para.x=-deltax/2;
                [imgxdx1, imgydx1] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = deltax;
                [imgxdx2, imgydx2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.x = 0;

                %                 Emitter.position_para.y=-deltax/2;

                imgxdy1 = imgxdx1;
                imgydy1 = imgydx1;
                Emitter.position_para.y = deltax;
                [imgxdy2, imgydy2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.y = 0;

                
                %gradient along z
                imgxdz1 = imgxdx1;
                imgydz1 = imgydx1;

                Emitter.position_para.z2 = z_cur+deltax;
                [imgxdz2, imgydz2] = simDipole_novotny_h(Emitter);
                Emitter.position_para.z2 = z_cur;

                
                imgZZx = imgx;
                imgZZy = imgy;
                imgZZxdx = (imgxdx2 - imgxdx1) / deltax;
                imgZZydx = (imgydx2 - imgydx1) / deltax;
                imgZZxdy = (imgxdy2 - imgxdy1) / deltax;
                imgZZydy = (imgydy2 - imgydy1) / deltax;
                imgZZxdz = (imgxdz2 - imgxdz1) / deltax;
                imgZZydz = (imgydz2 - imgydz1) / deltax;

                % XY
                Emitter.polar_para.phiD = pi / 4;
                Emitter.polar_para.thetaD = pi / 2;

                [imgXYxt, imgXYyt] = simDipole_novotny_h(Emitter);

                imgXYx = 2 * imgXYxt - imgXXx - imgYYx;
                imgXYy = 2 * imgXYyt - imgXXy - imgYYy;

                % XZ
                Emitter.polar_para.phiD = 0;
                Emitter.polar_para.thetaD = pi / 4;


                [imgXZxt, imgXZyt] = simDipole_novotny_h(Emitter);

                imgXZx = 2 * imgXZxt - imgXXx - imgZZx;
                imgXZy = 2 * imgXZyt - imgXXy - imgZZy;


                % YZ
                Emitter.polar_para.phiD = pi / 2;
                Emitter.polar_para.thetaD = pi / 4;

                [imgYZxt, imgYZyt] = simDipole_novotny_h(Emitter);

                imgYZx = 2 * imgYZxt - imgYYx - imgZZx;
                imgYZy = 2 * imgYZyt - imgYYy - imgZZy;

                %crop and normalize
                %---------------------------------------------------------

                %handle for croping region of interest

                N_pupil = size(imgYZy, 1);

                roi = @(img)img(-up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1:1:up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1, ... .
                    -up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1:1:up_sample*(imageSize - 1)/2+(N_pupil-1)/2+1, :);

                sumNormalize = sum(sum(roi(obj.brightnessScaling)));
                %x_channel
                %---------------------------------------------------------
                imgXXx = bsxfun(@times, roi(imgXXx), 1./sumNormalize);
                %gradient along x
                imgXXxdx = bsxfun(@times, roi(imgXXxdx), 1./sumNormalize);
                %gradient along y
                imgXXxdy = bsxfun(@times, roi(imgXXxdy), 1./sumNormalize);
                %gradient along z
                imgXXxdz = bsxfun(@times, roi(imgXXxdz), 1./sumNormalize);

                imgYYx = bsxfun(@times, roi(imgYYx), 1./sumNormalize);
                %gradient along x
                imgYYxdx = bsxfun(@times, roi(imgYYxdx), 1./sumNormalize);
                %gradient along y
                imgYYxdy = bsxfun(@times, roi(imgYYxdy), 1./sumNormalize);
                %gradient along z
                imgYYxdz = bsxfun(@times, roi(imgYYxdz), 1./sumNormalize);

                imgZZx = bsxfun(@times, roi(imgZZx), 1./sumNormalize);
                %gradient along x
                imgZZxdx = bsxfun(@times, roi(imgZZxdx), 1./sumNormalize);
                %gradient along y
                imgZZxdy = bsxfun(@times, roi(imgZZxdy), 1./sumNormalize);
                %gradient along z
                imgZZxdz = bsxfun(@times, roi(imgZZxdz), 1./sumNormalize);

                imgXYx = bsxfun(@times, roi(imgXYx), 1./sumNormalize);
                imgXZx = bsxfun(@times, roi(imgXZx), 1./sumNormalize);
                imgYZx = bsxfun(@times, roi(imgYZx), 1./sumNormalize);

                %y_channel

                sumNormalize = (1 / yToxChanTransRatio) * sumNormalize;
                %---------------------------------------------------------
                imgXXy = bsxfun(@times, roi(imgXXy), 1./sumNormalize);
                %gradient along x
                imgXXydx = bsxfun(@times, roi(imgXXydx), 1./sumNormalize);
                %gradient along y
                imgXXydy = bsxfun(@times, roi(imgXXydy), 1./sumNormalize);
                %gradient along z
                imgXXydz = bsxfun(@times, roi(imgXXydz), 1./sumNormalize);                

                imgYYy = bsxfun(@times, roi(imgYYy), 1./sumNormalize);
                %gradient along x
                imgYYydx = bsxfun(@times, roi(imgYYydx), 1./sumNormalize);
                %gradient along y
                imgYYydy = bsxfun(@times, roi(imgYYydy), 1./sumNormalize);
                %gradient along z
                imgYYydz = bsxfun(@times, roi(imgYYydz), 1./sumNormalize);                

                imgZZy = bsxfun(@times, roi(imgZZy), 1./sumNormalize);
                %gradient along x
                imgZZydx = bsxfun(@times, roi(imgZZydx), 1./sumNormalize);
                %gradient along y
                imgZZydy = bsxfun(@times, roi(imgZZydy), 1./sumNormalize);
                %gradient along z
                imgZZydz = bsxfun(@times, roi(imgZZydz), 1./sumNormalize);                

                imgXYy = bsxfun(@times, roi(imgXYy), 1./sumNormalize);
                imgXZy = bsxfun(@times, roi(imgXZy), 1./sumNormalize);
                imgYZy = bsxfun(@times, roi(imgYZy), 1./sumNormalize);

                %compute FFTs (single precision)
                %---------------------------------------------------------
                if ii==1
                    %x_channel
                    PSFx.XXx = single(imgXXx);
                    PSFx.YYx = single(imgYYx);
                    PSFx.ZZx = single(imgZZx);

                    %gradients
                    PSFx.XXxdx = single(10^2 * imgXXxdx);
                    PSFx.XXxdy = single(10^2 * imgXXxdy);
                    PSFx.XXxdz = single(10^2 * imgXXxdz);
                    PSFx.YYxdx = single(10^2 * imgYYxdx);
                    PSFx.YYxdy = single(10^2 * imgYYxdy);
                    PSFx.YYxdz = single(10^2 * imgYYxdz);                
                    PSFx.ZZxdx = single(10^2 * imgZZxdx);
                    PSFx.ZZxdy = single(10^2 * imgZZxdy);
                    PSFx.ZZxdz = single(10^2 * imgZZxdz);


                    PSFx.XYx = single(imgXYx);
                    PSFx.XZx = single(imgXZx);
                    PSFx.YZx = single(imgYZx);

                    %y_channel
                    PSFy.XXy = single(imgXXy);
                    PSFy.YYy = single(imgYYy);
                    PSFy.ZZy = single(imgZZy);

                    %gradients
                    PSFy.XXydx = single(10^2 * imgXXydx);
                    PSFy.XXydy = single(10^2 * imgXXydy);
                    PSFy.XXydz = single(10^2 * imgXXydz);
                    PSFy.YYydx = single(10^2 * imgYYydx);
                    PSFy.YYydy = single(10^2 * imgYYydy);
                    PSFy.YYydz = single(10^2 * imgYYydz);
                    PSFy.ZZydx = single(10^2 * imgZZydx);
                    PSFy.ZZydy = single(10^2 * imgZZydy);
                    PSFy.ZZydz = single(10^2 * imgZZydz);

                    PSFy.XYy = single(imgXYy);
                    PSFy.XZy = single(imgXZy);
                    PSFy.YZy = single(imgYZy);
                else
                    %x_channel
                    PSFx.XXx = cat(3,PSFx.XXx,single(imgXXx));
                    PSFx.YYx = cat(3,PSFx.YYx,single(imgYYx));
                    PSFx.ZZx = cat(3,PSFx.ZZx,single(imgZZx));

                    %gradients
                    PSFx.XXxdx = cat(3,PSFx.XXxdx,single(10^2 * imgXXxdx));
                    PSFx.XXxdy = cat(3,PSFx.XXxdy,single(10^2 * imgXXxdy));
                    PSFx.XXxdz = cat(3,PSFx.XXxdz,single(10^2 * imgXXxdz));
                    PSFx.YYxdx = cat(3,PSFx.YYxdx,single(10^2 * imgYYxdx));
                    PSFx.YYxdy = cat(3,PSFx.YYxdy,single(10^2 * imgYYxdy));
                    PSFx.YYxdz = cat(3,PSFx.YYxdz,single(10^2 * imgYYxdz));                
                    PSFx.ZZxdx = cat(3,PSFx.ZZxdx,single(10^2 * imgZZxdx));
                    PSFx.ZZxdy = cat(3,PSFx.ZZxdy,single(10^2 * imgZZxdy));
                    PSFx.ZZxdz = cat(3,PSFx.ZZxdz,single(10^2 * imgZZxdz));


                    PSFx.XYx = cat(3,PSFx.XYx,single(imgXYx));
                    PSFx.XZx = cat(3,PSFx.XZx,single(imgXZx));
                    PSFx.YZx = cat(3,PSFx.YZx,single(imgYZx));

                    %y_channel
                    PSFy.XXy = cat(3,PSFy.XXy,single(imgXXy));
                    PSFy.YYy = cat(3,PSFy.YYy,single(imgYYy));
                    PSFy.ZZy = cat(3,PSFy.ZZy,single(imgZZy));

                    %gradients
                    PSFy.XXydx = cat(3,PSFy.XXydx,single(10^2 * imgXXydx));
                    PSFy.XXydy = cat(3,PSFy.XXydy,single(10^2 * imgXXydy));
                    PSFy.XXydz = cat(3,PSFy.XXydz,single(10^2 * imgXXydz));
                    PSFy.YYydx = cat(3,PSFy.YYydx,single(10^2 * imgYYydx));
                    PSFy.YYydy = cat(3,PSFy.YYydy,single(10^2 * imgYYydy));
                    PSFy.YYydz = cat(3,PSFy.YYydz,single(10^2 * imgYYydz));
                    PSFy.ZZydx = cat(3,PSFy.ZZydx,single(10^2 * imgZZydx));
                    PSFy.ZZydy = cat(3,PSFy.ZZydy,single(10^2 * imgZZydy));
                    PSFy.ZZydz = cat(3,PSFy.ZZydz,single(10^2 * imgZZydz));

                    PSFy.XYy = cat(3,PSFy.XYy,single(imgXYy));
                    PSFy.XZy = cat(3,PSFy.XZy,single(imgXZy));
                    PSFy.YZy = cat(3,PSFy.YZy,single(imgYZy));
                end
            end

        end

    end

    methods

        %%

        function [imgx, imgy] = formImage(obj, Emitter, varargin)


            try
                molecule_num = numel(Emitter.position_para.x);
            catch
                error('Emitter does not have position_para.x field.')
            end

            if molecule_num > 20
                error('number of molecules exceedes maximum allowable number (20)')
            end

            s = opt2struct(varargin);
            %object parameters
            pixel_size = obj.pixelSize;
            up_sample = obj.pixelUpsample;
            N_pupil = size(obj.phaseMask, 1);

            %map orientation and rotational mobility to second moments


            rotMobil = Emitter.rotMobility;
            mux = sin(Emitter.theta) .* cos(Emitter.phi);
            muy = sin(Emitter.theta) .* sin(Emitter.phi);
            muz = cos(Emitter.theta);

            %consider only [-pi/2 pi/2] for mux muy

            Emitter.secondMoments.muxx = ...
                rotMobil .* mux.^2 + (1 - rotMobil) / 3;
            Emitter.secondMoments.muyy = ...
                rotMobil .* muy.^2 + (1 - rotMobil) / 3;
            Emitter.secondMoments.muzz = ...
                rotMobil .* muz.^2 + (1 - rotMobil) / 3;

            Emitter.secondMoments.muxy = ...
                rotMobil .* mux .* muy;
            Emitter.secondMoments.muxz = ...
                rotMobil .* mux .* muz;
            Emitter.secondMoments.muyz = ...
                rotMobil .* muy .* muz;

            %molecules' positions
            xpos = Emitter.position_para.x; %in nm
            ypos = Emitter.position_para.y; % in nm

            %sub_pixel positions
            %             x_pixel = floor(xpos./(pixel_size));
            %             y_pixel = floor(ypos./(pixel_size));
            %             x_subpixel = xpos - x_pixel.*pixel_size;
            %             y_subpixel = ypos - y_pixel.*pixel_size;
            x_subpixel = xpos;
            y_subpixel = ypos;
            if isfield(s, 'displayinfo') && s.displayinfo
                %display info
                %----------------------------------------------------
                display(strcat('muxx: ', num2str(Emitter.secondMoments.muxx)))
                display(strcat('muyy: ', num2str(Emitter.secondMoments.muyy)))
                display(strcat('muzz: ', num2str(Emitter.secondMoments.muzz)))
                %----------------------------------------------------
                %display info
                %----------------------------------------------------
                display(strcat('xpos: ', num2str(xpos)))
                display(strcat('ypos: ', num2str(ypos)))
                %----------------------------------------------------
                %display info
                %----------------------------------------------------
                display(strcat('x_pixel: ', num2str(x_pixel)))
                display(strcat('y_pixel: ', num2str(y_pixel)))
                display(strcat('x_subpixel: ', num2str(x_subpixel)))
                display(strcat('y_subpixel: ', num2str(y_subpixel)))
            end

            %define molecules with sub_pixel positions
            Emitter_t.position_para.x = x_subpixel;
            Emitter_t.position_para.y = y_subpixel;
            Emitter_t.position_para.z = 0 * y_subpixel;

            %allocate space

            img_size = obj.imageSize;
            imgy_shifted = zeros([[img_size, img_size] + N_pupil, molecule_num]);
            imgx_shifted = zeros([[img_size, img_size] + N_pupil, molecule_num]);
            imgy_corped = zeros([[img_size, img_size], molecule_num]);
            imgx_corped = zeros([[img_size, img_size], molecule_num]);

            [bx.XX, bx.YY, bx.ZZ, bx.XY, bx.XZ, bx.YZ, ...
                by.XX, by.YY, by.ZZ, by.XY, by.XZ, by.YZ] = ...
                Nanoscope.computeBases(obj, Emitter_t, varargin{:});

            %function handle for computing images formed on the camera
            img = @(bases, moments) bsxfun(@times, bases.XX, reshape(moments.muxx, 1, 1, molecule_num)) ...
                +bsxfun(@times, bases.YY, reshape(moments.muyy, 1, 1, molecule_num)) + ...
                +bsxfun(@times, bases.ZZ, reshape(moments.muzz, 1, 1, molecule_num)) + ...
                bsxfun(@times, bases.XY, reshape(moments.muxy, 1, 1, molecule_num)) + ...
                bsxfun(@times, bases.XZ, reshape(moments.muxz, 1, 1, molecule_num)) + ...
                bsxfun(@times, bases.YZ, reshape(moments.muyz, 1, 1, molecule_num));

            imgx = img(bx, Emitter.secondMoments);
            imgy = img(by, Emitter.secondMoments);


            % shift images to the corresponding positions

            %             for nn=1:molecule_num
            %
            %                 start_indx_y=y_pixel(nn)+(img_size-1)/2;
            %                 end_indx_y=y_pixel(nn)+(img_size-1)/2+N_pupil-1;
            %                 start_indx_x=x_pixel(nn)+(img_size-1)/2;
            %                 end_indx_x=x_pixel(nn)+(img_size-1)/2+N_pupil-1;
            %
            %                 imgy_shifted(start_indx_y:end_indx_y,start_indx_x:end_indx_x,nn) = imgy(:,:,nn);
            %
            %                 imgx_shifted(start_indx_y:end_indx_y,start_indx_x:end_indx_x,nn) = imgx(:,:,nn);
            %             end


            % cropping the image to match the region of interest
            %             imgy_corped = imgy_shifted(N_pupil/2+1:img_size+...
            %                 N_pupil/2,N_pupil/2+1:img_size+N_pupil/2,:);
            %             imgx_corped = imgx_shifted(N_pupil/2+1:img_size+...
            %                 N_pupil/2,N_pupil/2+1:img_size+N_pupil/2,:);
            %accounting for photon loss and normalization

            Emitter_t.polar_para.phiD = pi / 2;
            Emitter_t.polar_para.thetaD = pi / 2;
            Emitter_t.position_para.x = 0;
            Emitter_t.position_para.y = 0;
            Emitter_t.position_para.z = 0;

            %             [~,brightness_scaling]=obj.simDipole_novotny(obj,Emitter_t);
            [brightness_scalingX, brightness_scalingY] = obj.simDipole_novotny(obj, Emitter_t); % 190717
            brightness_scaling = brightness_scalingX + brightness_scalingY;

            %handle for croping region of interest
            roi = @(img)img(-up_sample*(img_size - 1)/2+N_pupil/2+2:1:up_sample*(img_size - 1)/2+N_pupil/2+2, ... .
                -up_sample*(img_size - 1)/2+N_pupil/2+2:1:up_sample*(img_size - 1)/2+N_pupil/2+2, :);


            imgy_corped = roi(imgy);
            imgx_corped = roi(imgx);
            sumnorm = sum(sum(roi(brightness_scaling)));

            imgx = (imgx_corped) / sumnorm;
            imgy = (imgy_corped) / sumnorm;

        end

        function [CRLB_vector] = CRB_orinet(obj, Emitter, varargin)


            s = opt2struct(varargin);

            if isfield(s, 'brightness')
                br = s.brightness;
            else
                br = 2000;
            end

            if isfield(s, 'background')
                bg = s.background;
            else
                bg = 5;
            end

            [imgx, imgy] = formImage(obj, Emitter);

            % dx
            Emitter_p = Emitter;
            Emitter_p.position_para.x = Emitter.position_para.x + .001;
            [imgxdx, imgydx] = formImage(obj, Emitter_p);

            gradx = br * ([imgxdx, imgydx] - [imgx, imgy]) / .001;

            % dy
            Emitter_p.position_para.x = Emitter.position_para.x;
            Emitter_p.position_para.y = Emitter.position_para.y + .001;
            [imgxdy, imgydy] = formImage(obj, Emitter_p);

            grady = br * ([imgxdy, imgydy] - [imgx, imgy]) / .001;
            % dtheta
            Emitter_p.position_para.y = Emitter.position_para.y;
            Emitter_p.theta = Emitter.theta + .001;
            [imgxdtheta, imgydtheta] = formImage(obj, Emitter_p);

            gradtheta = br * ([imgxdtheta, imgydtheta] - [imgx, imgy]) / .001;

            % dphi
            Emitter_p.theta = Emitter.theta;
            Emitter_p.phi = Emitter.phi + .001;
            [imgxdphi, imgydphi] = formImage(obj, Emitter_p);

            gradphi = br * ([imgxdphi, imgydphi] - [imgx, imgy]) / .001;

            % drotmob
            Emitter_p.phi = Emitter.phi;
            Emitter_p.rotMobility = Emitter.rotMobility + .001;
            [imgxdrotmob, imgydrotmob] = formImage(obj, Emitter_p);

            gradrotMob = br * ([imgxdrotmob, imgydrotmob] - [imgx, imgy]) / .001;

            FI = zeros(5, 5);
            FI(1, 2) = sum(sum((gradx.*grady) ./ ([imgx, imgy] * br + bg)));
            FI(1, 3) = sum(sum((gradx.*gradtheta) ./ ([imgx, imgy] * br + bg)));
            FI(1, 4) = sum(sum((gradx.*gradphi) ./ ([imgx, imgy] * br + bg)));
            FI(1, 5) = sum(sum((gradx.*gradrotMob) ./ ([imgx, imgy] * br + bg)));

            FI(2, 3) = sum(sum((grady.*gradtheta) ./ ([imgx, imgy] * br + bg)));
            FI(2, 4) = sum(sum((grady.*gradphi) ./ ([imgx, imgy] * br + bg)));
            FI(2, 5) = sum(sum((grady.*gradrotMob) ./ ([imgx, imgy] * br + bg)));

            FI(3, 4) = sum(sum((gradtheta.*gradphi) ./ ([imgx, imgy] * br + bg)));
            FI(3, 5) = sum(sum((gradtheta.*gradrotMob) ./ ([imgx, imgy] * br + bg)));

            FI(4, 5) = sum(sum((gradphi.*gradrotMob) ./ ([imgx, imgy] * br + bg)));

            FI = FI' + FI;

            FI(1, 1) = sum(sum(gradx.^2 ./ ([imgx, imgy] * br + bg)));
            FI(2, 2) = sum(sum(grady.^2 ./ ([imgx, imgy] * br + bg)));
            FI(3, 3) = sum(sum(gradtheta.^2 ./ ([imgx, imgy] * br + bg)));

            FI(4, 4) = sum(sum(gradphi.^2 ./ ([imgx, imgy] * br + bg)));
            FI(5, 5) = sum(sum(gradrotMob.^2 ./ ([imgx, imgy] * br + bg)));

            if Emitter.theta == 0
                FI(4, :) = [];
                FI(:, 4) = [];
                FI = FI + diag(ones(1, 4)*eps);
                CRLB_vector = sqrt(diag(inv(FI)));
                CRLB_vector(5) = CRLB_vector(4);
                CRLB_vector(4) = intmax;
            else
                FI = FI + diag(ones(1, 5)*eps);
                CRLB_vector = sqrt(diag(inv(FI)));
            end


        end

        %%

        function [imgx, imgy] = formImgIsotropic(obj, Emitter, varargin)

            s = opt2struct(varargin);

            %imaging parameters
            up_sample = obj.pixelUpsample;
            N_pupil = size(obj.phaseMask, 1);
            img_size = obj.imageSize;
            num_molecules = numel(Emitter.position_para.x);


            Emitter_t.position_para.x = Emitter.position_para.x; %in nm
            Emitter_t.position_para.y = Emitter.position_para.y; % in nm
            Emitter_t.position_para.z = Emitter.position_para.z; % in nm;

            simDipole_novotny_h = @(Emitter)Nanoscope.simDipole_novotny(obj, Emitter, varargin{:});
            % XX

            Emitter_t.polar_para.phiD = zeros(num_molecules, 1);
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BXXx, BXXy] = simDipole_novotny_h(Emitter_t);

            % YY
            Emitter_t.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BYYx, BYYy] = simDipole_novotny_h(Emitter_t);

            % ZZ
            Emitter_t.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * 0;


            [BZZx, BZZy] = simDipole_novotny_h(Emitter_t);


            %sum images of XX ,YY, and  ZZ bases

            %handle for croping region of interest
            if isfield(s, 'imgsize')
                img_size = s.imgsize;
            end
            roi = @(img)img(-up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, ... .
                -up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, :);
            imgx = BXXx + BYYx + BZZx;
            sumNormalize = sum(sum(roi(imgx)));
            imgx = bsxfun(@times, roi(imgx), 1./sumNormalize);

            imgy = BXXy + BYYy + BZZy;
            sumNormalize = sum(sum(roi(imgy)));
            imgy = bsxfun(@times, roi(imgy), 1./sumNormalize);
        end

        %%

        function [imgx, imgy] = formImgIsotropic_costum(obj, Emitter, pmask, varargin)

            s = opt2struct(varargin);

            %imaging parameters
            up_sample = obj.pixelUpsample;
            N_pupil = size(obj.phaseMask, 1);
            img_size = obj.imageSize;
            num_molecules = numel(Emitter.position_para.x);


            Emitter_t.position_para.x = Emitter.position_para.x; %in nm
            Emitter_t.position_para.y = Emitter.position_para.y; % in nm
            Emitter_t.position_para.z = Emitter.position_para.z; % in nm;

            simDipole_novotny_h = @(Emitter)Nanoscope.simDipole_novotny_costum(obj, ...
                Emitter, pmask, varargin{:});
            % XX

            Emitter_t.polar_para.phiD = zeros(num_molecules, 1);
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BXXx, BXXy] = simDipole_novotny_h(Emitter_t);

            % YY
            Emitter_t.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * pi / 2;


            [BYYx, BYYy] = simDipole_novotny_h(Emitter_t);

            % ZZ
            Emitter_t.polar_para.phiD = ones(num_molecules, 1) * pi / 2;
            Emitter_t.polar_para.thetaD = ones(num_molecules, 1) * 0;


            [BZZx, BZZy] = simDipole_novotny_h(Emitter_t);


            %sum images of XX ,YY, and  ZZ bases

            %handle for croping region of interest
            if isfield(s, 'imgsize')
                img_size = s.imgsize;
            end
            roi = @(img)img(-up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, ... .
                -up_sample*(img_size - 1)/2+N_pupil/2+1:1:up_sample*(img_size - 1)/2+N_pupil/2+1, :);
            imgx = BXXx + BYYx + BZZx;
            sumNormalize = sum(sum(roi(imgx)));
            imgx = bsxfun(@times, roi(imgx), 1./sumNormalize);

            imgy = BXXy + BYYy + BZZy;
            sumNormalize = sum(sum(roi(imgy)));
            imgy = bsxfun(@times, roi(imgy), 1./sumNormalize);
        end

        %%

        function Emitter = SimLipidBilayerExp(obj, varargin)


            s = opt2struct(varargin);

            if isfield(s, 'numberofmolecules')

                numMol = s.numberofmolecules;

            else
                numMol = 1;
            end


            if isfield(s, 'position')

                Emitter.position_para.x = s.position.x;
                Emitter.position_para.y = s.position.y;
                Emitter.position_para.z = s.position.z;

            else

                positionRangeVal = (obj.imageSize - 1) / 3 * obj.pixelSize;
                Emitter.position_para.x = 2 * rand(1, numMol) * positionRangeVal - positionRangeVal;
                Emitter.position_para.y = 2 * rand(1, numMol) * positionRangeVal - positionRangeVal;
                Emitter.position_para.z = (2 * rand(1, numMol) * positionRangeVal - positionRangeVal) * 0;

            end

            if isfield(s, 'theta')

                Emitter.theta = s.theta;
            else

                if isfield(s, 'thetarange')

                    try
                        rangeVal = (s.thetarange(2) - s.thetarange(1));

                    catch ME

                        error('expecting an array with 2 elements for range of theta')

                    end

                    if rangeVal < 0

                        error('expecting an array with 2 elements ([min_range,max_range]) for range of theta')
                    end
                else

                    s.thetarange = [0, pi / 2];
                end

                Emitter.theta = s.thetarange(1) + rand(1, numMol) * rangeVal;
            end

            if isfield(s, 'phi')

                Emitter.theta = s.phi;
            else

                Emitter.phi = rand(1, numMol) * 2 * pi;
            end

            if isfield(s, 'rotationalmobility')


                Emitter.rotMobility = s.rotationalmobility;
            else
                Emitter.rotMobility = rand(1, numMol);
            end


            %display emitter info

            if isfield(s, 'displayemitterpara') && s.displayemitterpara
                %orientation
                %----------------------------------------------
                h = histogram2(Emitter.theta*(90 / (pi / 2)), Emitter.rotMobility, ...
                    'DisplayStyle', 'tile', 'ShowEmptyBins', 'on', 'EdgeColor', 'none');
                h.XBinLimits = [0, 90];
                h.YBinLimits = [0, 1];
                h.BinWidth = [90 * .05, .05];
                xlabel('\theta (degree)')
                ylabel('\gamma')
            end

        end
    end

    %% plotting methods

    %--------------------------------------------------------
    methods

        %%

        function visualizePhaseMask(obj, varargin)
            s_1 = opt2struct(varargin);
            if (isfield(s_1, 'reusefigure') && s_1.reusefigure)

                h = gcf;
            else
                h = figure;
            end

            [phaseMask_t, rho_max, sizePhaseMask] = circ(obj);


            ax1 = subplot(1, 2, 1);
            imagesc(angle(phaseMask_t(sizePhaseMask / 2 - rho_max:sizePhaseMask / 2 + rho_max, ...
                sizePhaseMask / 2 - rho_max:sizePhaseMask / 2 + rho_max)))
            axis image;
            axis off;
            title('angle ', 'FontSize', 10)

            colorbar;
            ax2 = subplot(1, 2, 2);
            imagesc(abs(phaseMask_t(sizePhaseMask / 2 - rho_max:sizePhaseMask / 2 + rho_max, ...
                sizePhaseMask / 2 - rho_max:sizePhaseMask / 2 + rho_max)))
            axis image;
            axis off;
            title('amplitude', 'FontSize', 10)
            colorbar;

            function [phaseMask_t, radius, sizePhaseMask] = circ(obj)


                phaseMask_t = obj.phaseMask(:, :, 1);
                sizePhaseMask = size(phaseMask_t, 1);
                radius = obj.phaseMaskPara.pupilRadius;

                eta = -sizePhaseMask / 2:1:sizePhaseMask / 2 - 1;
                [eta, zeta] = meshgrid(eta);

                indx = (eta.^2 + zeta.^2) > radius^2;

                phaseMask_t(indx > 0) = 0;

            end

        end

        %%

        function visualizeBases(obj, varargin)

            s_1 = opt2struct(varargin);
            if (isfield(s_1, 'reusefigure') && s_1.reusefigure)
                h = gcf;
            else
                h = figure;
            end
            max_brightness = max(obj.YYyBasis(:));

            util_plot(h, obj.XXxBasis, obj.XXyBasis, [.01, .70], .25, .2, 'basis', 'XX');
            util_plot(h, obj.YYxBasis, obj.YYyBasis, [.33, .70], .25, .2, 'basis', 'YY');
            util_plot(h, obj.ZZxBasis, obj.ZZyBasis, [.67, .70], .25, .2, 'basis', 'ZZ');
            util_plot(h, obj.XYxBasis, obj.XYyBasis, [.01, .25], .25, .2, 'negativeVal', true, 'basis', 'XY');
            util_plot(h, obj.XZxBasis, obj.XZyBasis, [.33, .25], .25, .2, 'negativeVal', true, 'basis', 'XZ');
            util_plot(h, obj.YZxBasis, obj.YZyBasis, [.67, .25], .25, .2, 'negativeVal', true, 'basis', 'YZ');

            function util_plot(handleIn, imgIn1, imgIn2, pos, w, h, varargin)
                s = opt2struct(varargin);
                ax1 = axes(handleIn, 'Position', [pos(1), pos(2), w, h]);
                imagesc(imgIn1/max_brightness);
                axis image; axis off

                %colorbar
                caxis manual
                cmax = max([max(imgIn1(:)), max(imgIn2(:))]) / max_brightness;
                cmin = max([min(imgIn1(:)), min(imgIn2(:))]) / max_brightness;
                colormap(ax1, parula);

                if (isfield(s, 'negativeval') && s.negativeval)
                    caxis([-cmax, cmax])
                else
                    caxis([cmin, cmax])
                end
                c1 = colorbar;

                set(c1.Label, 'Rotation', 90);
                c1.Position = [pos(1) + w + .005, pos(2) - h, .01, 2 * h];
                set(c1, 'YAxisLocation', 'left')

                %remove ticks since both pannels use the same range value
                c1.Ticks = [];

                % markup
                if (isfield(s, 'basis') && (any(strcmp(s.basis, {'XX', 'XY'}))))
                    xLim = get(gca, 'Xlim');
                    yLim = get(gca, 'Ylim');
                    ht = text(0.9*xLim(1)-0.1*xLim(2), 0.1*yLim(1)+0.9*yLim(2), ...
                        'x-channel', ...
                        'Color', 'k', ...
                        'Rotation', 90, ...
                        'FontWeight', 'bold');
                end
                %title
                title(s.basis)
                ax2 = axes(handleIn, 'Position', [pos(1), pos(2) - h, w, h]);
                imagesc(imgIn2/max_brightness);
                axis image; axis off
                caxis manual
                if (isfield(s, 'negativeval') && s.negativeval)
                    caxis([-cmax, cmax])
                else
                    caxis([cmin, cmax])
                end
                c2 = colorbar;

                c2.Position = [pos(1) + w + .015, pos(2) - h, .01, 2 * h];

                % markup
                if (isfield(s, 'basis') && (any(strcmp(s.basis, {'XX', 'XY'}))))
                    xLim = get(gca, 'Xlim');
                    yLim = get(gca, 'Ylim');
                    ht = text(0.9*xLim(1)-0.1*xLim(2), 0.1*yLim(1)+0.9*yLim(2), ...
                        'y-channel', ...
                        'Color', 'k', ...
                        'Rotation', 90, ...
                        'FontWeight', 'bold');
                end
            end
        end

        %% methods for loading TiFF data

        function [img, h, raw_img] = loadImg(obj, filename, varargin)


            if ~ischar(filename)

                error('Nanoscope:loadImg:InconsistentInputType', ...
                    'Expecting a character array for filename.')
            end

            s = opt2struct(varargin);


            if isfield(s, 'fullpath') && s.fullpath

                FileTif = filename;

            else

                if 7 == exist('dataset', 'dir')

                    FileTif = fullfile('dataset', filename);
                else
                    error('Nanoscope:loadImg:DirNotFound', ...
                        strcat('Expecting ', filename, ' in a folder named as dataset'));
                end

            end


            % get image info

            info_image = imfinfo(FileTif);
            %remove unknown tags
            info_image(1).UnknownTags = [];
            number_images = length(info_image);
            img_sizex = info_image(1).Width;
            img_sizey = info_image(1).Height;
            raw_img = zeros(img_sizey, img_sizex, number_images);

            %check input and set appropriate parameter

            if isfield(s, 'sizeroi') && isnumeric(s.sizeroi)

                %make sure image size is an odd number
                if mod(s.sizeroi, 2) == 0
                    sizeROI = s.sizeroi - 1;
                else
                    sizeROI = s.sizeroi;
                end
            else
                sizeROI = obj.imageSize;
            end

            if isfield(s, 'centerroi') && isnumeric(s.centerroi)
                x_c = s.centerroi(1);
                y_c = s.centerroi(2);
            else
                x_c = round(img_sizex/2);
                y_c = round(img_sizey/2);
            end

            %creat a Tiff object and read the data from TIFF file

            t = Tiff(FileTif, 'r');

            for j = 1:number_images
                setDirectory(t, j);
                raw_img(:, :, j) = t.read();
            end

            %close the Tiff object
            t.close();

            % crop images
            %---------------------------------------------------------
            if (isfield(s, 'squareroi') && ~(s.squareroi))

                try

                    if ((x_c + (sizeROI(1) - 1) / 2 > img_sizex) || (x_c - (sizeROI(1) - 1) / 2) <= 0 ...
                            || (y_c + (sizeROI(2) - 1) / 2 > img_sizey) || (y_c - (sizeROI(2) - 1) / 2) <= 0)
                        error('Nanoscope:loadImg:BadInput', ...
                            'size of the ROI  exceeds the  size  of  input image.')
                    end
                catch ME
                    if (strcmp(ME.identifier, 'MATLAB:badsubscript'))
                        error('Expecting sizeROI with two elements')
                    else
                        rethrow(ME)

                    end
                end
                widht = sizeROI(1);
                height = sizeROI(2);
                % specify the width and height of the croped image
                %---------------------------------------------------------
                width_ROI = x_c - round((sizeROI(1)-1)/2):x_c + round((sizeROI(1)-1)/2);
                height_ROI = y_c - round((sizeROI(2)-1)/2):y_c + round((sizeROI(2)-1)/2);
                corped_img = raw_img(height_ROI, width_ROI, :);
            else

                %check the size of ROI
                if ((x_c + (sizeROI - 1) / 2 > img_sizex) || (x_c - (sizeROI - 1) / 2) <= 0 ...
                        || (y_c + (sizeROI - 1) / 2 > img_sizey) || (y_c - (sizeROI - 1) / 2) <= 0)

                    error('Nanoscope:loadImg:BadInput', ...
                        'size of the ROI  exceeds the  size  of  input image.')
                end
                widht = sizeROI(1);
                height = sizeROI(1);
                % specify the width and height of the croped image
                %---------------------------------------------------------
                width_ROI = x_c - (sizeROI - 1) / 2:x_c + (sizeROI - 1) / 2;
                height_ROI = y_c - (sizeROI - 1) / 2:y_c + (sizeROI - 1) / 2;
                corped_img = raw_img(height_ROI, width_ROI, :);
            end

            %subtract offset
            if isfield(s, 'offset')
                offset_t = s.offset;
                dimOffset = size(s.offset);

                %check offset dimension to match croped image

                if ~all(dimOffset == 1) % if not a scalar

                    if (dimOffset(1) ~= widht || dimOffset(2) ~= height)

                        error('Nanoscope:loadImg:BadInput', ...
                            'Expecting an offset image with the same size as the region of interest (sizeROI).')
                    end

                    try

                        % average offset over the stack of images
                        offset_t = sum(s.offset, 3) / dimOffset(3);

                    catch ME
                        if ~(strcmp(ME.identifier, 'MATLAB:badsubscript'))
                            rethrow(ME)
                        end
                    end
                    %
                    %                     if dimOffset(3)~=size(corped_img,3)
                    %
                    %                         % average offset over the stack of images
                    %                         offset_t=sum(s.offset,3)/dimOffset(3);
                    %                     end

                end
                corped_img = bsxfun(@plus, corped_img, -offset_t);
            else
                corped_img = bsxfun(@plus, corped_img, -obj.offset);
            end

            %make sure all pixels take on positive values
            min_pixel_val = .001;
            non_pos_indx = corped_img <= 0;
            corped_img(non_pos_indx) = min_pixel_val;

            %convert to photon counts
            if isfield(s, 'adcount')
                pixel_val_to_photon = s.adcount;
            else
                pixel_val_to_photon = obj.ADcount;
            end

            img = single(corped_img.*pixel_val_to_photon);

            h = []; % empty handle
            if isfield(s, 'visualize') && s.visualize

                h = figure;
                imagesc(raw_img(:, :, ceil(number_images / 2)))
                axis image
                hold on

                if isfield(s, 'roi') && s.roi

                    % show ROI with a rectangle
                    rectangle('Position', [x_c - (widht - 1) / 2, y_c - (height - 1) / 2, widht, height], 'EdgeColor', 'y')

                    % display center of ROI with a marker
                    plot(x_c, y_c, '*y', 'MarkerSize', 3)
                end

                % zoom in
                imagesc(img(:, :, ceil(number_images / 2)))
                axis image
                drawnow
            end
        end

        %%

        function [SMLM_img, folder_path, posRect, centerROI] = loadImgInteractive(obj, varargin)
            %loadImgInteractive load raw images form a specificed path; a
            %ROI is selected by user;
            %it subtracts off-set and accounts for A/D count; it also applies a geometric
            %transform to find the corresponding ROI in the other channel
            %(optional)


            s = opt2struct(varargin);

            % specify the data folder
            if isfield(s, 'datapath') && ischar(s.datapath)
                folder_path = s.datapath;
            else
                folder_path = uigetdir('', 'Folder of the desired set of camera images'); % specify the path of the desired set of camera images
            end
            if isfield(s, 'offsetpath') && ischar(s.offsetpath)
                offset_folder_path = s.offsetpath;
            else
                offset_folder_path = uigetdir('', 'Folder of  the off-set image'); % specify the path of the off-set images
            end


            addpath(folder_path);
            addpath(offset_folder_path);
            FilesInfo = dir(fullfile(folder_path, '*.tif')); % extracting  stack of images

            offset_files_info = dir(fullfile(offset_folder_path, '*.tif'));

            %catch possible errors
            if isempty(FilesInfo)
                error('Nanoscope:BadFileName', ...
                    'The specified folder contains no file with .tif extension')
            end
            if isempty(offset_files_info)
                error('Nanoscope:BadFileName', ...
                    'The specified folder for offset contains no file with .tif extension')
            end

            % load SMLM images
            %---------------------------------------------------------
            %get dimension info
            FileTif = fullfile(folder_path, FilesInfo(1).name);
            info_image = imfinfo(FileTif);
            number_axial_scans = length(FilesInfo);
            chip_img_width = info_image(1).Width; % width of the image captured on camera
            %launch an interactive ROI selection via a rectangle

            filename = fullfile(folder_path, FilesInfo(ceil(number_axial_scans / 2)).name);

            % the image at focus is displayed
            [~, h_t, img_t] = obj.loadImg(filename, 'fullpath', true, ...
                'visualize', true);
            title('Select a region of interet', 'FontSize', 11)

            if ~isfield(s, 'nointeractive') || ~(s.nointeractive)

                %setup  an inteactive rectangle
                hRect = imrect(gca, [info_image(1).Width / 3, info_image(1).Height / 3, ...
                    info_image(1).Height / 3, info_image(1).Height / 3]);

                %add an event listener to display chosen ROI at the top
                %left corner
                addID = addNewPositionCallback(hRect, ...
                    @(positionRect) zoominfcn(positionRect, img_t));

                %get the position of the rectangle object
                posRect = round(wait(hRect));

                %get the rid of the listener
                removeNewPositionCallback(hRect, addID);

                %make sure the chosen ROI is valid
                if (posRect(3) < 81 || posRect(4) < 81)

                    error('Nanoscope:InconsistentInputValue', ...
                        'Soryy! the ROI selected is small for 3D PSFs.')
                end


                %get ROI info
                if (isfield(s, 'squareroi') && ~(s.squareroi))
                    %make sure ROI is large enough for sliding window algorithm
                    posRect(3:4) = posRect(3:4) + round(obj.imageSize/2);
                    sizeROI = [posRect(3), posRect(4)];
                    if mod(sizeROI(1), 2) == 0
                        sizeROI(1) = sizeROI(1) + 1;
                    end
                    if mod(sizeROI(2), 2) == 0
                        sizeROI(2) = sizeROI(2) + 1;
                    end
                    width = sizeROI(1);
                    height = sizeROI(2);
                else
                    s.squareroi = true;
                    sizeROI = min(posRect(3), posRect(4));
                    if mod(sizeROI, 2) == 0
                        sizeROI = sizeROI + 1;
                    end
                    width = sizeROI(1);
                    height = sizeROI(1);
                end

                centerROI = [floor(posRect(1) + width / 2), floor(posRect(2) + height / 2)];

            else
                s.squareroi = true;
                posRect = [];
                try
                    sizeROI = s.sizeroi;
                catch ME
                    error('Expecting sizeROI option as input!')
                end

                if mod(sizeROI, 2) == 0
                    sizeROI = sizeROI + 1;
                end
                width = sizeROI(1);
                height = sizeROI(1);

                try
                    centerROI = s.centerroi;
                catch ME
                    error('Expecting centerROI option as input!')
                end
            end


            close(h_t)

            number_frames = size(img_t, 3);
            SMLM_raw_img = zeros(height, width, ...
                number_frames, number_axial_scans);

            %set the flag to display image acquired at focus
            visulaizeFlag = zeros(1, number_axial_scans);
            visulaizeFlag(ceil((number_axial_scans+1) / 2)) = 1;

            % read images
            for i = 1:number_axial_scans
                filename = fullfile(folder_path, FilesInfo(i).name);

                SMLM_raw_img(:, :, :, i) = obj.loadImg(filename, 'fullpath', true, ...
                    'offset', 0, ...
                    'sizeROI', sizeROI, ...
                    'centerROI', centerROI, ...
                    'ADcount', 1, ...
                    'visualize', visulaizeFlag(i), ...
                    'ROI', true, ...
                    'squareroi', s.squareroi);
            end

            % load  offset image
            %---------------------------------------------------------
            offset_file_name = fullfile(offset_folder_path, offset_files_info(1).name);

            if length(offset_files_info) > 1

                error('Nanoscope:InconsistentInputValue', ...
                    'offset (averaged) must contain a single stack of tif file.')
            end

            offfset_raw_img = obj.loadImg(offset_file_name, 'fullpath', true, ...
                'offset', 0, ...
                'sizeROI', sizeROI, ...
                'centerROI', centerROI, ...
                'ADcount', 1, ...
                'visualize', false, ...
                'squareroi', s.squareroi);

            %average offset stack
            if isfield(s, 'nooffset') && s.nooffset
                offfset_avg_img = 0;
            else
                offfset_avg_img = sum(double(offfset_raw_img), 3) / size(offfset_raw_img, 3);
            end
            %subtract offset
            SMLM_img = bsxfun(@plus, double(SMLM_raw_img), -offfset_avg_img);

            %measurments shoud be positive
            SMLM_img((SMLM_img <= 0) > 0) = .001;

            %apply photoelecton conversion
            if ~isfield(s, 'noadconversion') || ~s.noadconversion
                SMLM_img = SMLM_img .* obj.ADcount;
            end

            % select the appropriate region in other channel
            %--------------------------------------------------
            if isfield(s, 'tform')
                if ~isobject(s.tform)
                    error('Nanoscope:InconsistentInputType', ...
                        'expecting an  images.geotrans.PolynomialTransformation2D object for input transform ');
                end
                %step 1- identify the current channel
                if centerROI(1) < chip_img_width / 2
                    cur_channel = 'L';
                else
                    cur_channel = 'R';
                end

                %step 2- map the centerROI to an appropriate
                %coordinate
                ref_x_coordinate = chip_img_width / 2; % the 0 coordinate for x-axis
                %used in obtaining the transformation
                if strcmp(cur_channel, 'L')

                    NewCenterROI(1) = ref_x_coordinate - centerROI(1);
                    NewCenterROI(2) = centerROI(2);
                else
                    NewCenterROI(1) = -ref_x_coordinate + centerROI(1);
                    NewCenterROI(2) = centerROI(2);

                end

                %step 3- apply the transform on the center coordinate

                if isfield(s, 'transformscale')

                    scale = s.transformscale;

                else
                    scale = 1;
                end
                [transformed_centerROI(1), transformed_centerROI(2)] = ...
                    transformPointsInverse(s.tform, NewCenterROI(1)*scale, NewCenterROI(2)*scale);

                %step 4- map the transformed coordinate to original
                %coordinate(pixel)
                transformed_centerROI = ceil(transformed_centerROI/scale);

                if isfield(s, 'registershiftx')
                    deltax = s.registershiftx;
                else
                    deltax = 0;

                end

                if isfield(s, 'registershifty')

                    deltay = s.registershifty;
                else
                    deltay = 0;
                end


                transformed_centerROI(1) = transformed_centerROI(1) - deltax;
                transformed_centerROI(2) = transformed_centerROI(2) - deltay;

                transformed_centerROI = ceil(transformed_centerROI);
                if strcmp(cur_channel, 'L')
                    transformed_centerROI(1) = (transformed_centerROI(1)) + ref_x_coordinate;
                else
                    transformed_centerROI(1) = -(transformed_centerROI(1)) + ref_x_coordinate;
                end

                %step 5- extract the region on the other channel

                % read images
                for i = 1:number_axial_scans
                    filename = fullfile(folder_path, FilesInfo(i).name);

                    SMLM_raw_img_2(:, :, :, i) = obj.loadImg(filename, 'fullpath', true, ...
                        'offset', 0, ...
                        'sizeROI', sizeROI, ...
                        'centerROI', transformed_centerROI, ...
                        'ADcount', 1, ...
                        'visualize', visulaizeFlag(i), ...
                        'ROI', true, ...
                        'squareroi', s.squareroi);
                end

                offfset_raw_img_2 = obj.loadImg(offset_file_name, 'fullpath', true, ...
                    'offset', 0, ...
                    'sizeROI', sizeROI, ...
                    'centerROI', transformed_centerROI, ...
                    'ADcount', 1, ...
                    'visualize', false, ...
                    'squareroi', s.squareroi);
                %average offset stack
                if isfield(s, 'nooffset') && s.nooffset
                    offfset_avg_img = 0;
                else
                    offfset_avg_img = sum(double(offfset_raw_img_2), 3) / size(offfset_raw_img_2, 3);
                end
                %subtract offset
                SMLM_img_2 = bsxfun(@plus, double(SMLM_raw_img_2), -offfset_avg_img);

                %measurments shoud be positive
                SMLM_img_2((SMLM_img_2 <= 0) > 0) = .001;

                %apply photoelecton conversion
                if ~isfield(s, 'noadconversion') || ~s.noadconversion
                    SMLM_img_2 = SMLM_img_2 .* obj.ADcount;
                end

                if strcmp(cur_channel, 'L')
                    %              SMLM_img=flipud(SMLM_img);
                    %              SMLM_img_2=fliplr(SMLM_img_2);
                    %            SMLM_img_2=flipud(SMLM_img_2);

                    SMLM_img = [SMLM_img_2, fliplr(SMLM_img)];
                else
                    SMLM_img_2 = flipud(SMLM_img_2);
                    %              SMLM_img=fliplr(SMLM_img);
                    SMLM_img = flipud(SMLM_img);

                    SMLM_img = [SMLM_img, SMLM_img_2];
                end


            end
            %local functions
            %--------------------------------------------------
            function zoominfcn(p, img)
                imagesc(img(round(p(2)):round(p(2)) + round(p(4)), (round(p(1)):round(p(1)) + round(p(3)))));
                axis image
            end

        end


    end

    %% recovery methods

    methods (Static)
        function[gammahat1, recovStruct] = TriSpotDetection(obj, SMLM_img, b, varargin)
            %TriSpotDetection returns a list of molecular parameter estimates
            %->---
            %input
            %->---
            %SMLM_img :            array(2*m,2*m,n_f)    -stack of single-molecule...
            %localization microscopy images(x-y channel concanotated)
            %
            %                       *  *   ...             *
            %                       *  *   ...             *
            %                       :  x-channel     :    y-channel
            %SMLM_img=*  *   ...             *
            %
            %
            %images are realizations of the following statistical measurement model:
            %SMLM_img ~ Poisson(img+background),
            %b:                    array(2,n_f) -estimates of background for all
            %frames (assuming uniform background)
            %---->-
            %output
            %---->-
            %loc_data:                estimated molecular parameters per localization
            %each row contains frame number, brightness, position(x,y) , second
            %moments


            %check for proper input images
            %------------------------------------------------------------
            if any(SMLM_img < 0)
                error('input image must not be background subtracted')
            end

            img_size = size(SMLM_img, 1); % image side-length size

            if 2 * img_size ~= size(SMLM_img, 2)

                error('input image must be a square region')

            end

            %make sure image size is an odd number
            if mode(img_size, 2) == 0
                error('side length (in number of camera pixels) of input image must be an odd value')
            end


            %Fourier transforms of the  point-spread function
            %------------------------------------------------------------

            [FPSFx, FPSFy, lateral_grid_p] = obj.PSF_Fourier_tf(obj);

            %x_channel
            FXXx = FPSFx.FXXx;
            FYYx = FPSFx.FYYx;
            FZZx = FPSFx.FZZx;
            FXYx = FPSFx.FXYx;
            FXZx = FPSFx.FXZx;
            FYZx = FPSFx.FYZx;
            %y_channel
            FXXy = FPSFy.FXXy;
            FYYy = FPSFy.FYYy;
            FZZy = FPSFy.FZZy;
            FXYy = FPSFy.FXYy;
            FXZy = FPSFy.FXZy;
            FYZy = FPSFy.FYZy;

            FXX(:, :, 1) = FXXx;
            FXX(:, :, 2) = FXXy;
            FYY(:, :, 1) = FYYx;
            FYY(:, :, 2) = FYYy;
            FZZ(:, :, 1) = FZZx;
            FZZ(:, :, 2) = FZZy;

            FXY(:, :, 1) = FXYx;
            FXY(:, :, 2) = FXYy;
            FXZ(:, :, 1) = FXZx;
            FXZ(:, :, 2) = FXZy;
            FYZ(:, :, 1) = FYZx;
            FYZ(:, :, 2) = FYZy;

            %break all the frames into a set of sub_frames of length subframe_l
            %------------------------------------------------------------
            s = opt2struct(varargin);

            if isfield(s, 'subframelength')
                subframe_l = s.subframelength;
            else
                subframe_l = 1; % number of sub-frames for sub_frame analysis
            end

            num_frames = size(SMLM_img, 3); % total number of frames

            SMLM_img = reshape(SMLM_img, 2*img_size^2, num_frames);


            if mod(num_frames, subframe_l) == 0
                n_sub_frame = floor(num_frames/subframe_l) - 1;
            else
                n_sub_frame = floor(num_frames/subframe_l);
            end

            SMLM_img(:, end+1:(n_sub_frame + 1)*subframe_l) = 0;

            b(:, end+1:(n_sub_frame + 1)*subframe_l) = 0;

            %re-arranging the input frames for sub_frame  analysis
            SMLM_img_n = reshape(SMLM_img, 2*img_size^2, subframe_l, n_sub_frame+1);

            % fixed or learned parameters for recovery
            %------------------------------------------------------------
            n_boundry_p = 5; %number of camera pixels for removing boundry artifacts

            upsample_factor = obj.pixelUpsample; %object space pixel upsampling

            n_grid_p = (upsample_factor * img_size - (upsample_factor - 1))^2; % number of grid points in the object space

            % joint sparsity regularizer
            reg_val = .16;
            MaxIt1 = 400; %maximum number of iterations (first stage)

            w = ones(1, subframe_l, (n_sub_frame + 1)); %weights

            %build recovery structure
            %------------------------------------------------------------
            recovStruct = struct();
            recovStruct.img_size = img_size;
            recovStruct.lateral_grid_p = lateral_grid_p;
            recovStruct.num_frames = num_frames;
            recovStruct.subframe_l = subframe_l;
            recovStruct.upsample_factor = upsample_factor;
            recovStruct.n_grid_p = n_grid_p;
            recovStruct.reg_val = reg_val;
            %----------------------------------------------------
            gamma_init = zeros(6*n_grid_p, num_frames);

            SMLM_img_r_x = zeros(sqrt(n_grid_p), sqrt(n_grid_p), num_frames);
            SMLM_img_r_y = zeros(sqrt(n_grid_p), sqrt(n_grid_p), num_frames);

            SMLM_img_r_x(1:upsample_factor:end, 1:upsample_factor:end, :) = ...
                reshape(SMLM_img(1:img_size^2, 1:num_frames), img_size, img_size, num_frames); % up_sample

            SMLM_img_r_y(1:upsample_factor:end, 1:upsample_factor:end, :) = ...
                reshape(SMLM_img(1 + img_size^2:2 * img_size^2, 1:num_frames), img_size, img_size, num_frames); % up_sample
            %gamma_init_t=A'*SMLM_img; computation in Fourier space.

            gamma_init_t_xx = real(ifft2(bsxfun(@times, conj(FXXx), fft2(SMLM_img_r_x)))) + real(ifft2(bsxfun(@times, conj(FXXy), fft2(SMLM_img_r_y))));
            gamma_init_t_yy = real(ifft2(bsxfun(@times, conj(FYYx), fft2(SMLM_img_r_x)))) + real(ifft2(bsxfun(@times, conj(FYYy), fft2(SMLM_img_r_y))));
            gamma_init_t_zz = real(ifft2(bsxfun(@times, conj(FZZx), fft2(SMLM_img_r_x)))) + real(ifft2(bsxfun(@times, conj(FZZy), fft2(SMLM_img_r_y))));


            gamma_init_t_xx = reshape(padarray(gamma_init_t_xx(n_boundry_p + 1:end - n_boundry_p, n_boundry_p + 1:end - n_boundry_p, :) ...
                , [n_boundry_p, n_boundry_p]), n_grid_p, num_frames);

            gamma_init_t_yy = reshape(padarray(gamma_init_t_yy(n_boundry_p + 1:end - n_boundry_p, n_boundry_p + 1:end - n_boundry_p, :) ...
                , [n_boundry_p, n_boundry_p]), n_grid_p, num_frames);
            gamma_init_t_zz = reshape(padarray(gamma_init_t_zz(n_boundry_p + 1:end - n_boundry_p, n_boundry_p + 1:end - n_boundry_p, :) ...
                , [n_boundry_p, n_boundry_p]), n_grid_p, num_frames);

            gamma_init(1:n_grid_p, :) = repmat(sum(SMLM_img(:, 1:num_frames)) ...
                ./sum(gamma_init_t_xx), n_grid_p, 1) .* (gamma_init_t_xx);

            gamma_init(n_grid_p+1:2*n_grid_p, :) = repmat(sum(SMLM_img(:, 1:num_frames)) ...
                ./sum(gamma_init_t_yy), n_grid_p, 1) .* (gamma_init_t_yy);

            gamma_init(n_grid_p*2+1:3*n_grid_p, :) = repmat(sum(SMLM_img(:, 1:num_frames)) ...
                ./sum(gamma_init_t_zz), n_grid_p, 1) .* (gamma_init_t_zz);


            % upper bound on Lipschitz constant of the Poisson negative log-likelihood
            %------------------------------------------------------------
            l_max = upper_bound_Lipschitz_cnt(SMLM_img, b, FPSFx, FPSFy, recovStruct);

            l = l_max / 10;

            %re-arranging parameters for sub_frame analysis
            l(end+1:(n_sub_frame + 1)*subframe_l) = 1;
            l = reshape(l, subframe_l, (n_sub_frame + 1))';

            %re-arranging background estimates
            b = reshape(b, 2, subframe_l, (n_sub_frame + 1));
            % routiens
            %------------------------------------------------------------

            down_samp = @(x)x(1:upsample_factor:end, 1:upsample_factor:end, :);

            xxgrid = @(x)(reshape(x(1:n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            yygrid = @(x)(reshape(x(n_grid_p + 1:2 * n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            zzgrid = @(x)(reshape(x(2 * n_grid_p + 1:3 * n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            xygrid = @(x)(reshape(x(3 * n_grid_p + 1:4 * n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            xzgrid = @(x)(reshape(x(4 * n_grid_p + 1:5 * n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            yzgrid = @(x)(reshape(x(5 * n_grid_p + 1:6 * n_grid_p, :), sqrt(n_grid_p), sqrt(n_grid_p), subframe_l));

            %routines for computing Az via FFT

            %             fA=@(x) abs(reshape([down_samp( real(ifft2(bsxfun(@times,FXXx,fft2(xxgrid(x)))+...
            %                 bsxfun(@times,FYYx,fft2(yygrid(x)))+bsxfun(@times,FZZx,fft2(zzgrid(x)))+...
            %                 bsxfun(@times,FXYx,fft2(xygrid(x)))+bsxfun(@times,FXZx,fft2(xzgrid(x)))+...
            %                 bsxfun(@times,FYZx,fft2(yzgrid(x)))))),...
            %                 down_samp( real(ifft2(bsxfun(@times,FXXy,fft2(xxgrid(x)))+...
            %                 bsxfun(@times,FYYy,fft2(yygrid(x)))+bsxfun(@times,FZZy,fft2(zzgrid(x)))+...
            %                 bsxfun(@times,FXYy,fft2(xygrid(x)))+bsxfun(@times,FXZy,fft2(xzgrid(x)))+...
            %                 bsxfun(@times,FYZy,fft2(yzgrid(x))))))], 2*img_size^2,subframe_l));

            fA = @(x) abs(reshape(down_samp(real(ifft2(bsxfun(@times, FXX, fft2(xxgrid(x))) + ...
                bsxfun(@times, FYY, fft2(yygrid(x))) + bsxfun(@times, FZZ, fft2(zzgrid(x))) + ...
                bsxfun(@times, FXY, fft2(xygrid(x))) + bsxfun(@times, FXZ, fft2(xzgrid(x))) + ...
                bsxfun(@times, FYZ, fft2(yzgrid(x)))))), 2 * img_size^2, subframe_l));
            % loop over sub_frames
            %------------------------------------------------------------

            %allocate space for first stage estimates

            gammahat1 = zeros(6*n_grid_p, subframe_l, n_sub_frame+1);
            gamma_init = reshape(gamma_init, 6*n_grid_p, subframe_l, n_sub_frame+1);
            num_char = 0;

            for nn = 1:n_sub_frame + 1


                %prepare sub_frame structures
                b_it(1:img_size^2, :) = repmat(b(1, :, nn), img_size^2, 1);
                b_it(img_size^2+1:2*img_size^2, :) = repmat(b(2, :, nn), img_size^2, 1);
                SMLM_it = SMLM_img_n(:, :, nn);
                z = gamma_init(:, :, nn);
                gammaold = gamma_init(:, :, nn);
                w_it = w(:, :, nn);
                t = 1;
                i = 1;
                l_it = l(nn, :);
                recovStruct.w_it = w_it;
                %sub_frame routines
                %--------------------------------------------------------

                %gradient of negative Poisson log-likelihood

                gz = @(z)gradf(z, SMLM_it, b_it, FPSFx, FPSFy, recovStruct);

                zq = @(z, gz_var, l_var)proxmu(z-repmat((1./l_var), 6 * n_grid_p, 1).*gz_var, recovStruct);

                %Poisson negative log-likelihood

                f1 = @(var)sum(var) - sum(SMLM_it.*log(bsxfun(@plus, var, b_it)));

                q1 = @(var, z, gz_var, zq_var, l_var)sum(var) - sum(SMLM_it.*log(bsxfun(@plus, var, b_it))) + sum((gz_var).*(zq_var - z), 1) + ...
                    (l_var / 2) .* sum((z-zq_var).^2, 1);

                %handle for updating gamma via proximal operator

                gammanew_update = @(z, gz_var, l_var) proxmu(z-repmat((1./l_var), 6 * n_grid_p, 1).* ...
                    gz_var, recovStruct);


                while (i < MaxIt1)

                    if (mod(i, 20) == 0 && i < 250)
                        l_it = l_it / 5;
                    end

                    %backtracking line search
                    k = 1;
                    eta = 1.1 * ones(1, subframe_l); % line-search parameter
                    gz_it = gz(z);
                    zq_it = zq(z, gz_it, l_it);
                    Azq = fA((zq_it));
                    Az = fA((z));
                    comp1 = f1(Azq) > (q1(Az, z, gz_it, zq_it, l_it) + .05); % descent condition

                    while (any(comp1))

                        l_it(comp1) = (eta(comp1).^k) .* l_it(comp1);

                        zq_it = zq(z, gz_it, l_it);

                        Azq = fA((zq_it));

                        comp1 = f1(Azq) > (q1(Az, z, gz_it, zq_it, l_it) + .05);
                        %disp(f1(Azq)-(q1(Az,z,gz_it,zq_it,l_it)))
                        %disp(l_it)
                        k = k + 1;
                    end

                    %update estimates
                    %-----------------------------------------------------

                    ll = l_it; %Lipschitz constant of the smoothed objectvie function

                    gammanew = gammanew_update(z, gz_it, ll);

                    tnew = 1 / 2 + (sqrt(1 + 4 * t^2)) / 2;

                    %momentum term of FISTA

                    z = gammanew + ((t - 1) / tnew) * (gammanew - gammaold);

                    gammaold = gammanew; % record the most recent estimate

                    t = tnew;
                    i = i + 1;

                end

                %record final estimates

                gammahat1(:, :, nn) = gammanew;
                %num_char=progress_bar(round(nn/(n_sub_frame+1),2),num_char,20);

            end
        end

        function localMaxIndx = JointProcessBases(obj, gamma, recovStruct)


            n_grid_p = recovStruct.n_grid_p;
            localMaxWindowSize = 2; % in pixels
            bd_pixels = 4; %boundary pixels
            num_frames = recovStruct.num_frames;


            %step 1: find local maximum in each "basis"
            %--------------------------------------------------

            for i = 1:num_frames
                indx = cell(1, 3);
                gamma_t = gamma(:, i);


                %get local maxima in each basis
                for ii = 1:3
                    indx{ii} = findLocalMax(gamma_t, ii);
                end

                %step 2: jointly process local maxima to find molecules

                localMaxIndx{i} = mergLocalMaxBases(indx, localMaxWindowSize);
            end


            % routines
            %--------------------------------------------------

            %%

            function locMax_t = findLocalMax(gamma, basisIndx)

                %set boundry pixels to zero
                basisGrid = @(x, basisIndx)(abs(reshape(x((basisIndx-1) * n_grid_p + 1:(basisIndx) * n_grid_p, :), ...
                    sqrt(n_grid_p), sqrt(n_grid_p))));

                gamma_reshaped = basisGrid(gamma, basisIndx);

                gamma_padded = padarray(gamma_reshaped(bd_pixels + 1:size(gamma_reshaped, 1) - bd_pixels, ...
                    bd_pixels + 1:size(gamma_reshaped, 2) - bd_pixels), [bd_pixels, bd_pixels], 0);


                locMax_t = [];

                if any(gamma_padded(:) >= 1)

                    % filtering pixels with low vlaue
                    [I, J] = find(gamma_padded >= max(gamma_padded(:))*.1);

                    %loop over pixels

                    for k = 1:length(I)

                        % creat a local image
                        gamma_padded_local = gamma_padded(I(k)-2:I(k)+2, J(k)-2:J(k)+2);

                        % check if current index is a local max

                        if (gamma_padded(I(k), J(k)) == max(gamma_padded_local(:)))

                            % add to current indices

                            locMax_t = [locMax_t; J(k), I(k)];
                        end

                    end

                end

            end

            %%

            function localMax = mergLocalMaxBases(localMaxIndx, localMaxWindowSize)

                localMax = [];

                %loop over bases
                for k = 1:3

                    %indices of local maxima in the current basis
                    indx = localMaxIndx{k};

                    %loop over local maxima
                    for kk = 1:size(indx, 1)

                        curindx = [indx(kk, 1), indx(kk, 2)];

                        %loop over local maxima in the rmaining bases
                        for kkk = k + 1:3
                            if ~isempty(localMaxIndx{kkk})

                                indx_t = localMaxIndx{kkk};

                                %check if the set of local maxima can be reduced by
                                %merging close local maxima across all bases
                                rm_indx = sum((indx_t-repmat(curindx, size(indx_t, 1), 1)).^2, 2) <= localMaxWindowSize;
                                if any(rm_indx)

                                    %remove close indices
                                    indx_t(rm_indx, :) = [];
                                end
                            end

                            if ~isempty(localMaxIndx{kkk})
                                localMaxIndx{kkk} = indx_t;

                            end
                        end
                    end

                    localMax{k} = indx;
                end

            end

        end
    end
end

%%
