import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

import numpy as np

class ConvAtten(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 img_size,
                 proj_drop=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.layer_q = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 'same', groups=channels),
            nn.ReLU()
        )
        self.layernorm_q = nn.LayerNorm([channels, img_size, img_size], eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 'same', groups=channels),
            nn.ReLU()
        )
        self.layernorm_k = nn.LayerNorm([channels, img_size, img_size], eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 'same', groups=channels),
            nn.ReLU()
        )
        self.layernorm_v = nn.LayerNorm([channels, img_size, img_size], eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               num_heads=self.num_heads,
                                               batch_first=True,
                                               dropout=self.proj_drop,
                                               )
    
    def _build_projection(self, x, mode):
        if mode == 0:
            x1 = self.layer_q(x)
            proj = self.layernorm_q(x1)
        elif mode == 1:
            x1 = self.layer_k(x)
            proj = self.layernorm_k(x1)
        elif mode == 2:
            x1 = self.layer_v(x)
            proj = self.layernorm_v(x1)
        return proj
    
    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v
    
    def forward(self, x):
        q, k, v = self.get_qkv(x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2]*q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2]*k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2]*v.shape[3])
        # flattened w*h as sequence length
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        # restore image dimension
        x1 = x1[0].permute(0, -1, 1)
        wh = np.sqrt(x1.shape[2]).astype(int)
        x1 = x1.view(x1.shape[0], x1.shape[1],wh, wh)
        
        return x1

class Wide_Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dialation1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same', dilation=1),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.dialation2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same', dilation=2),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.dialation3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same', dilation=3),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.GELU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x1 = self.dialation1(x)
        x2 = self.dialation2(x)
        x3 = self.dialation3(x)
        added = x1 + x2 + x3
        out = self.layer4(added)
        return out

class Transformer(nn.Module):
    def __init__(self,
                 out_channels,
                 num_heads,
                 dpr,
                 img_size,
                 proj_drop=0.5,):
        super().__init__()

        self.attention = ConvAtten(channels=out_channels,
                                   num_heads=num_heads,
                                   img_size=img_size,
                                   proj_drop=proj_drop)
        self.stochastic_depth = StochasticDepth(dpr, mode='batch')
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 'same')
        self.layernorm = nn.LayerNorm([out_channels, img_size, img_size])
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention(x)
        x1 = self.stochastic_depth(x1)
        x2 = self.conv1(x1) + x

        x3 = self.layernorm(x2)
        x3 = self.wide_focus(x3)
        x3 = self.stochastic_depth(x3)

        out = x2 + x3
        return out

class Blocker_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super().__init__()
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, 3, 1, 'same'),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.transformer = Transformer(out_channels, att_heads, dpr, img_size*2)
    
    def forward(self, x, skip):
        x1 = self.layernorm(x)
        x1 = self.upsample(x1)
        x1 = self.layer1(x1)
        x1 = torch.cat((skip, x1), dim=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.transformer(x1)
        return x1
    
class DeepSupervised(nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super().__init__()
        
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size], eps=1e-5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same')
        )
        
    def forward(self, x):
        x1 = self.layernorm(x)
        out = self.conv(x1)
        return out

class Block_encoder_without_skip(nn.Module):
    def __init__(self, in_ch, out_ch, att_heads, dpr, img_size, last=False):
        super().__init__()

        self.layernorm = nn.LayerNorm([in_ch, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        # img_size -> img_size // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.maxpool = nn.MaxPool2d((2,2))
        self.last = last
        if self.last:
            self.img_size = img_size//2
        else:
            self.img_size = img_size
        self.transformer = Transformer(out_ch, att_heads, dpr, self.img_size)
    
    def forward(self, x):
        x = self.layernorm(x)
        x1 = self.layer1(x)
        x1 = self.layer2(x1)
        # for last encoder
        if self.last:
            x1 = self.maxpool(x1)
        out = self.transformer(x1)
        return out

class Block_encoder_with_skip(nn.Module):
    def __init__(self, in_ch, out_ch, att_heads, dpr, img_size):
        super().__init__()

        self.layernorm = nn.LayerNorm([in_ch, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, in_ch, 3, 1, 'same'),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3, 1, 'same'),
            nn.ReLU()
        )
        # img_size -> img_size / 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 'same'),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2,2))
        )
        self.transformer = Transformer(out_ch, att_heads, dpr, img_size//2)
    
    def forward(self, x, scale_img):
        x1 = self.layernorm(x)
        x1 = torch.cat((self.layer1(scale_img), x1), dim=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.transformer(x1)
        return x1

class FCT(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.img_size = img_size

        att_heads = [2, 4, 8, 16, 32, 16, 8, 4, 2]
        filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]

        num_blocks = len(filters)
        stochastic_depth_rate = 0.5
        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2,2)

        self.encoder1 = Block_encoder_without_skip(1, filters[0], att_heads[0], dpr[0], self.img_size)
        self.encoder2 = Block_encoder_with_skip(filters[0], filters[1], att_heads[1], dpr[1], self.img_size//2)
        self.encoder3 = Block_encoder_with_skip(filters[1], filters[2], att_heads[2], dpr[2], self.img_size//4)
        self.encoder4 = Block_encoder_without_skip(filters[2], filters[3], att_heads[3], dpr[3], self.img_size//8)
        self.encoder5 = Block_encoder_without_skip(filters[3], filters[4], att_heads[4], dpr[4], self.img_size//16,
                                                   last=True)

        self.decoder4 = Blocker_decoder(filters[4], filters[5], att_heads[5], dpr[5], self.img_size//16)
        self.decoder3 = Blocker_decoder(filters[5], filters[6], att_heads[6], dpr[6], self.img_size//8)
        self.decoder2 = Blocker_decoder(filters[6], filters[7], att_heads[7], dpr[7], self.img_size//4)
        self.decoder1 = Blocker_decoder(filters[7], filters[8], att_heads[8], dpr[8], self.img_size//2)

        self.ds3 = DeepSupervised(filters[6], 4, self.img_size//4)
        self.ds2 = DeepSupervised(filters[7], 4, self.img_size//3)
        self.ds1 = DeepSupervised(filters[8], 4, self.img_size)
    
    def forward(self, x):
        # Multi-scale input
        scale_img2 = self.scale_img(x)
        scale_img3 = self.scale_img(scale_img2)
        
        # encoding path
        x1 = self.encoder1(x)
        print('x1 shape', x1.shape)
        x2 = self.encoder2(x1, scale_img2)
        x3 = self.encoder3(x2, scale_img3)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # decoding path
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d1, x1)

        out3 = self.ds3(d3)
        out2 = self.ds2(d2)
        out1 = self.ds1(d1)

        return out3, out2, out1

if __name__ == '__main__':
    img_size = (0, 1, 256, 256)
    x = torch.rand(*img_size)
    model = FCT(256)
    model.eval()
    y = model(x)

    print(y.shape)


        


        

        