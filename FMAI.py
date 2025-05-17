import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .dsc import DSC

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return (x_LL, x_HL, x_LH, x_HH)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll, hl, lh, hh):
        x = torch.cat((ll, hl, lh, hh), 1)
        return iwt_init(x)



class Hfe(nn.Module):
    def __init__(self, dim):
        super(Hfe, self).__init__()

        self.crb = CRB(dim)
        self.DWT = DWT()
        self.IWT = IWT()
        self.conv1 = DSC(dim, dim)
        self.conv2 = DSC(dim, dim)
        self.conv3 = DSC(dim, dim)
        self.conv4 = DSC(dim, dim)
    def forward(self,x):
        ll,lh,hl,hh = self.DWT(x)
        lh_res = self.conv1(self.enhance(lh,ll)) + lh
        hl_res = self.conv2(self.enhance(hl,ll)) + hl
        hh_res = self.conv3(self.enhance(hh,ll)) + hh
        low_res = self.conv4(ll)
        res = self.IWT(low_res,lh_res,hl_res,hh_res)
        return res

if __name__ == '__main__':


