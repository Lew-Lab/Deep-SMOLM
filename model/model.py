import torch
import torch.nn as nn
from torch.nn.functional import interpolate


# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


# Localization architecture
#class LocalizationCNN(nn.Module):
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=2, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(2, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 2, 64, 3, 1, 1, 0.2)
        #if setup_params['dilation_flag']:
        # self.layer3 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (2, 2), (2, 2), 0.2)
        # self.layer4 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (4, 4), (4, 4), 0.2)
        # self.layer5 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (8, 8), (8, 8), 0.2)
        # self.layer6 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (16, 16), (16, 16), 0.2)
        # else:
        self.layer3 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64 + 2, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64 + 2, 64, 3, 1, 1, 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64 + 2, 64, 3, 1, 1, 0.2)
        #d_max = setup_params['D']
        
        self.deconv1 = Conv2DLeakyReLUBN(64 + 2, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        d_max = 6
        # self.layer7 = Conv2DLeakyReLUBN(64, d_max, 3, 1, 1, 0.2)
        # self.layer8 = Conv2DLeakyReLUBN(d_max, d_max, 3, 1, 1, 0.2)
        # self.layer9 = Conv2DLeakyReLUBN(d_max, d_max, 3, 1, 1, 0.2)
        # self.layer10 = nn.Conv2d(d_max, d_max, kernel_size=1, dilation=1)
        self.layer7 = Conv2DLeakyReLUBN(64, 32, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(32, 16, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(16, 8, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(8, 6, kernel_size=1, dilation=1)



        

        #self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):

        # extract multi-scale features
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out


        # upsample by 6 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=3)
        out = self.deconv1(out)
        out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        intermediate_out = out

        channel_outputs = []

        out = self.layer7(intermediate_out)
        out = self.layer8(out) 
        out = self.layer9(out)
        out = self.layer10(out)
        # channel_outputs.append(out)

        # out = self.layer7_2(intermediate_out)
        # out = self.layer8_2(out) + out
        # out = self.layer9_2(out) + out
        # out = self.layer10_2(out)
        # channel_outputs.append(out)

        # out = self.layer7_3(intermediate_out)
        # out = self.layer8_3(out) + out
        # out = self.layer9_3(out) + out
        # out = self.layer10_3(out)
        # channel_outputs.append(out)

        # out = self.layer7_4(intermediate_out)
        # out = self.layer8_4(out) + out
        # out = self.layer9_4(out) + out
        # out = self.layer10_4(out)
        # channel_outputs.append(out)

        # out = self.layer7_5(intermediate_out)
        # out = self.layer8_5(out) + out
        # out = self.layer9_5(out) + out
        # out = self.layer10_5(out)
        # channel_outputs.append(out)

        # out = self.layer7_6(intermediate_out)
        # out = self.layer8_6(out) + out
        # out = self.layer9_6(out) + out
        # out = self.layer10_6(out)
        # channel_outputs.append(out)



        # out = torch.cat(channel_outputs, 1)

        return out