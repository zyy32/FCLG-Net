import torch
import torch.nn as nn
from .dsc import DSC,IDSC 
from .FMAI import High_frequency_enhance
import torch.nn.functional as F
from .transformer import Block


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):

        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
    
class cfa(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(cfa, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat, kernel_size, padding=1), act, nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)]
        self.body = nn.Sequential(*modules_body)
        ## Pixel Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = CALayer(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res



class LFWF(nn.Module):
    def __init__(self, n_feat, n_res):
        super(LFWF, self).__init__()

        self.conv1 = DSC(n_feat, n_feat)
        self.conv3 = DSC(n_res, n_feat)
        self.CAF = cfa(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)
        self.conv1x11 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x, res):
        x1 = self.conv1(x)
        temp = self.CAF(x1)
        x1 = torch.sigmoid(temp)
        x2 = self.conv3(res)
        x2 = torch.cat([x2, temp], dim=1)
        x2 = self.conv1x1(x2)
        x1 = x1*x2
        r = torch.cat([x1, x2], dim=1)
        r = self.conv1x11(r) 
        return r


class Encoder5(nn.Module):
    def __init__(self):
        super(Encoder5, self).__init__()

        self.layer1 = DSC(3,16)
        self.layer2 = nn.Sequential(
            Block(16, window_size=2, alpha=0.5)

        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Decoder5(nn.Module):
    def __init__(self):
        super(Decoder5, self).__init__()
        self.block32 = Block(16, window_size=2, alpha=0.5)
        self.up2 = IDSC(16,3)
        self.sam5 = ASISF(n_feat=16,n_res=3)

    def forward(self, x):

        x = self.block32(x)
        res = x
        x = self.up2(x)
        res4_sam = self.sam5(res, x)
        return res4_sam

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()

        self.encoder_2 = Encoder5()
        self.encoder_3 = Encoder5()
        self.encoder_4 = Encoder5()
        self.encoder_5 = Encoder5()

        self.decoder_2 = Decoder5()
        self.decoder_3 = Decoder5()
        self.decoder_4 = Decoder5()
        self.decoder_5 = Decoder5()


    def forward(self, x):
 
        x_2x = F.upsample(x, scale_factor=0.5)
        x_4x = F.upsample(x_2x, scale_factor=0.5)
        x_8x = F.upsample(x_4x, scale_factor=0.5)
        x_16x = F.upsample(x_8x, scale_factor=0.5)

        stage5 = self.encoder_5(x_16x)
        res5_sam = self.decoder_5(stage5)

        stage4 = self.encoder_4(x_8x)
        res4_sam = self.decoder_4(stage4)

        stage3 = self.encoder_3(x_4x)
        res3_sam = self.decoder_3(stage3)

        stage2 = self.encoder_2(x_2x)
        res2_sam = self.decoder_2(stage2)

        return res2_sam,res3_sam,res4_sam,res5_sam

class FMLF(nn.Module):
    def __init__(self):
        super(FMLF, self).__init__()

        self.encoder_stage = LR()


        self.conv1_1 = DSC(3, 32)
        self.conv1_2 = DSC(32, 32)
        self.conv1_3 = DSC(32, 32)
        self.norm1 = nn.BatchNorm2d(32)
        self.act = nn.GELU()

        self.res1 = nn.Conv2d(64, 32, kernel_size=1)

        self.pool1 = DSC(32, 32, 2, 2, 0)


        self.conv2_1 = DSC(32, 64)
        self.conv2_2 = DSC(64, 64)
        self.conv2_3 = DSC(64, 64)
        self.norm2 = nn.BatchNorm2d(64)

        self.res2 = nn.Conv2d(128, 64, kernel_size=1)

        self.pool2 = DSC(64, 64, 2, 2, 0)


        self.conv3_1 = DSC(64, 128)
        self.conv3_2 = DSC(128, 128)
        self.conv3_3 = DSC(128, 128)
        self.conv3_4 = DSC(128, 128)
        self.conv3_5 = DSC(128, 128)
        self.norm3 = nn.BatchNorm2d(128)

        self.res3 = nn.Conv2d(256, 128, 1)

        self.pool3 = DSC(128, 128, 2, 2, 0)


        self.conv4_1 = DSC(128, 256)
        self.conv4_2 = DSC(256, 256)
        self.conv4_3 = DSC(256, 256)
        self.norm4 = nn.BatchNorm2d(256)

        self.res4 = nn.Conv2d(512, 256, 1)

        self.pool4 = DSC(256, 256, 2, 2, 0)

        self.pool5 = DSC(256, 512, 2, 2, 0)

        self.e0 = Hfe(32)
        self.e1 = Hfe(64)
        self.e2 = Hfe(128)
        self.e3 = Hfe(256)


        self.sam1 = LFWF(n_feat=32,n_res=16)
        self.sam2 = LFWF(n_feat=64,n_res=16)
        self.sam3 = LFWF(n_feat=128,n_res=16)
        self.sam4 = LFWF(n_feat=256,n_res=16)

        self.t3 = IDSC(128, 256)
        self.t2 = IDSC(64, 128)
        self.t1 = IDSC(32, 64)
        self.t0 = IDSC(16, 32)        

        self.up = nn.PixelShuffle(2)

        self.final = nn.Sequential(nn.PixelShuffle(4),
                                   IDSC(4, 1))

        self.desam1 = LFWF(n_feat=16,n_res=16)
        self.desam2 = LFWF(n_feat=32,n_res=16)
        self.desam3 = LFWF(n_feat=64,n_res=16)
        self.desam4 = LFWF(n_feat=128,n_res=16)

        self.d0 = Hfe(32)
        self.d1 = Hfe(64)
        self.d2 = Hfe(128)
        self.d3 = Hfe(256)

        self.conv1 = DSC(64, 32)
        self.conv2 = DSC(128, 64)
        self.conv3 = DSC(256, 128)
        self.conv4 = DSC(512, 256)

        self.CONV = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        y2, y3, y4, y5  = self.encoder_stage(x)   


        x1_1 = self.c1_1(x)
        x1_p = self.pool1(x1_1)
        x1_2 = self.c1_2(x1_p)
        x1_3 = self.c1_3(x1_2)
        x1_3 = self.e0(x1_3)
        temp1 = torch.cat([x1_p, x1_3], dim=1)
        x1_out = self.res1(temp1)

        x1_out = self.sam1(x1_out,y2) + x1_out


        x2_1 = self.c2_1(x1_out)
        x2_p = self.pool2(x2_1)
        x2_2 = self.c2_2(x2_p)
        x2_3 = self.c2_3(x2_2)
        x2_3 = self.e1(x2_3)
        temp2 = torch.cat([x2_p, x2_3], dim=1)
        x2_out = self.res2(temp2)
        x2_out = self.sam2(x2_out,y3) + x2_out


        x3_1 = self.c3_1(x2_out)
        x3_2 = self.c3_2(x3_1)
        x3_p = self.pool3(x3_2)
        x3_3 = self.c3_3(x3_p)
        x3_4 = self.c3_4(x3_3)
        x3_5 = self.c3_5(x3_4)
        x3_5 = self.e2(x3_5)
        temp3 = torch.cat([x3_p, x3_5], dim=1)
        x3_out = self.res3(temp3)
        x3_out = self.sam3(x3_out,y4) + x3_out


        x4_1 = self.c4_1(x3_out)
        x4_p = self.pool4(x4_1)
        x4_2 = self.c4_2(x4_p)
        x4_3 = self.c4_3(x4_2)
        x4_3 = self.e3(x4_3)
        temp4 = torch.cat([x4_p, x4_3], dim=1)
        x4_out = self.res4(temp4)
        x4_out = self.sam4(x4_out,y5) + x4_out

        out = self.pool5(x4_out)



        temp = self.up(out)
        temp = self.t3(temp)
        temp = torch.cat([temp, x4_out], dim=1)
        temp = self.conv4(temp)
        temp = self.d3(temp)


      
        temp = self.up(temp)
        temp = self.desam3(temp,y4) + temp
        temp = self.t2(temp)
        temp = torch.cat([temp, x3_out], dim=1)
        temp = self.conv3(temp)
        temp = self.d2(temp)


        temp = self.up(temp)
        temp = self.desam2(temp,y3) + temp      
        temp = self.t1(temp)
        temp = torch.cat([temp, x2_out], dim=1)
        temp = self.conv2(temp)
        temp = self.d1(temp)



        temp = self.up(temp)
        temp = self.desam1(temp,y2) + temp
        temp = self.t0(temp)
        temp = torch.cat([temp, x1_out], dim=1)
        temp = self.conv1(temp)
        temp = self.d0(temp)
        temp = self.CONV(temp)

        out = self.final(temp)
        y5 = self.littlefinal3(y5)
        y4 = self.littlefinal2(y4)
        y3 = self.littlefinal1(y3)
        y2 = self.littlefinal0(y2)

        return out,y5,y4,y3,y2




    





