import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ General Convolution Blocks -------------------#
class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x

# -----------------Attention Blocks--------------------------#
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.relu = nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        gate =self.relu(g1 + x1)
        gate = self.gate(gate)
        return x * gate

def init_weights(m, init_type):
    clsname = m.__class__.__name__
    if hasattr(m, 'weight') and (clsname.find('Conv') != -1 or clsname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0)
        elif init_type =='xavier':
            nn.init.xavier_normal_(m.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data)
        else:
            raise NotImplementedError(f'initiation method {init_type} is not implemented')

        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    
    elif clsname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

###################################################################
##                              UNet                             ##
###################################################################
class UNet(nn.Module):
    def __init__(self, num_classes=4, img_channel=1):
        super(UNet, self).__init__()

        self.maxpool = nn.MaxPool2d((2,2))

        self.conv1 = conv_block(ch_in=img_channel, ch_out=64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)

        self.up4 = up_conv(512, 256)
        self.up_conv4 = conv_block(512, 256)

        self.up3 = up_conv(256, 128)
        self.up_conv3 = conv_block(256, 128)

        self.up2 = up_conv(128, 64)
        self.up_conv2 = conv_block(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        # decoding + concat path
        d4 = self.up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        # after concat, the channel doubled
        # apply another conv_block to reduce it half
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.output(d2)
        return d1

###################################################################
##                         AttenUnet                             ##
###################################################################
class AttenUnet(nn.Module):
    def __init__(self, num_classes=4, img_channel=1):
        super(AttenUnet, self).__init__()

        self.maxpool = nn.MaxPool2d((2,2))

        self.conv1 = conv_block(ch_in=img_channel, ch_out=64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)

        self.up4 = up_conv(512, 256)
        self.att4 = attention_block(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = conv_block(512, 256)

        self.up3 = up_conv(256, 128)
        self.att3 = attention_block(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = conv_block(256, 128)

        self.up2 = up_conv(128, 64)
        self.att2 = attention_block(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = conv_block(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        # decoding + concat path
        d4 = self.up4(x4)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.output(d2)
        return d1

###################################################################
##                 AttenUnetV2 with Deeper CNN                   ##
###################################################################
class AttenUnetV2(nn.Module):
    def __init__(self, num_classes=4, img_channel=1):
        super(AttenUnetV2, self).__init__()

        self.maxpool = nn.MaxPool2d((2,2))

        self.conv1 = conv_block(ch_in=img_channel, ch_out=64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        
        self.up5 = up_conv(1024, 512)
        self.att5 = attention_block(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = conv_block(1024, 512)

        self.up4 = up_conv(512, 256)
        self.att4 = attention_block(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = conv_block(512, 256)
        
        self.up3 = up_conv(256, 128)
        self.att3 = attention_block(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = conv_block(256, 128)

        self.up2 = up_conv(128, 64)
        self.att2 = attention_block(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = conv_block(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.output(d2)
        return d1