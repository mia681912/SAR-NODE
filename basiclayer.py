"""
despeckling using wavelet
coder: Ziqing Ma  time:2024/09/22
"""

import torch
import torch.nn as nn
from argas_haar import args_n
import os
import cv2

opt = args_n()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum


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

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

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


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBEM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBEM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SpaEmbe(nn.Module):
    def __init__(self, ms_channel, out_channel):
        super(SpaEmbe, self).__init__()
        k_size4 = 7
        k_size1 = 5
        k_size2 = 3
        k_size3 = 1
        # out_channel = 128
        self.conv7 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size4, padding=(k_size4 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size1, padding=(k_size1 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size2, padding=(k_size2 - 1) // 2, bias=False),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, k_size3, bias=False),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(4 * out_channel, out_channel, k_size3, padding=(k_size3 - 1) // 2, bias=False)

    def forward(self, lrhs):
        out7 = self.conv7(lrhs)
        out5 = self.conv5(lrhs)
        out3 = self.conv3(lrhs)
        out1 = self.conv1(lrhs)
        # print(out7.shape)
        # print(out5.shape)
        # print(out3.shape)
        # print(out1.shape)

        out = torch.cat([out7, out5, out3, out1], 1)
        out = self.conv(out)

        return out


class LL_denoise(nn.Module):
    def __init__(self, ll_channel, ll_feature, denoise_num, device):
        super(LL_denoise, self).__init__()

        self.in_channel = ll_channel
        self.ll_feature = ll_feature
        self.ksize5 = 3
        self.ksize3 = 3
        self.ksize1 = 3

        layers = []
        layers.append(nn.Conv2d(self.in_channel+1, self.ll_feature, 1))
        for i in range(denoise_num):
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize5, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, self.ksize3, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
            layers.append(CBEM(self.in_channel))
            layers.append(nn.Conv2d(self.ll_feature, self.ll_feature, 3, padding=1))
            layers.append(nn.BatchNorm2d(self.ll_feature))
            layers.append(nn.ReLU())
        # layers.append(nn.Conv2d(self.ll_feature, self.in_channel, self.ksize3, padding=self.ksize3 // 2))
        self.denoise = nn.Sequential(*layers)
        self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, t, image_ll):
        input = self.concat_t(image_ll, t)
        out = self.denoise(input)
        out = self.diffu_equ(out)

        return out

    def concat_t(self, h: torch.Tensor, t):
        h_shape = h.shape
        tt = torch.ones(h_shape[0], 1, h_shape[2], h_shape[3]).to(self.dev) * t
        out_ = torch.cat((h, tt), dim=1).float()  # shape =[bach_size, features+1, N_x, N_y]
        return out_

    # def diffu_equ(self, u):
    #     div = (self.bx(self.fx(u)) + self.by(self.fy(u)))
    #     return div
    def diffu_equ(self, u):
        ep = 1e-3
        # u_sig = cv2.GaussianBlur(u, (5, 5), 0.8)
        u_fx = self.fx(u)
        # u_bx = self.bx(u_sig)
        u_fy = self.fy(u)
        # u_by = self.by(u_sig)
        div = self.bx((self.fx(u) * torch.sqrt(1+u_fx**2+u_fy**2))) 
        + self.by(self.fy(u) * torch.sqrt(1+u_fx**2+u_fy**2)) 
        return div

    def m(self,out, out1):
        return ((torch.sign(out) + torch.sign(out1)) / 2) * torch.min(torch.abs(out), torch.abs(out1))

    def fx(self,u):
        v = torch.cat((u[:,:,1:,:], u[:,:,-1,:].unsqueeze(2)), dim=2) - u
        return v

    def bx(self,u):
        v = u - torch.cat((u[:,:,0,:].unsqueeze(2), u[:,:,0:-1,:]), dim=2)
        return v

    def fy(self,u):
        v = torch.cat((u[:,:,:,1:], u[:,:,:,-1].unsqueeze(3)), dim=3) - u
        return v

    def by(self,u):
        v = u - torch.cat((u[:,:,:,0].unsqueeze(3), u[:, :,:,0:-1]), dim=3)
        return v

    def cx(self,u):
        v = (torch.cat((u[:,:,1:,:], u[:,:,-2:-1,:]), dim=2) \
            - torch.cat((u[:,:,0:1,:], u[:,:,0:-1,:]), dim=2)) / 2
        return v


    def cy(self,u):
        v = (torch.cat((u[:,:,:,1:], u[:,:,:,-2:-1]), dim=3) \
            - torch.cat((u[:,:,:,0:1], u[:,:,:,0:-1]), dim=3)) / 2
        return v


class Super_Attention(nn.Module):
    def __init__(self, hh_channel, hh_feature):
        super(Super_Attention, self).__init__()

        self.in_channel = hh_channel
        self.feature = hh_feature

        self.att1 = CBEM(self.in_channel)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU(),
            nn.Conv2d(self.feature, self.feature, 3, padding=1),
            nn.BatchNorm2d(self.feature),
            nn.ReLU()
        )

    def forward(self, hh):
        out1 = self.conv2(hh)
        out1 = out1 + self.att1(out1)
        out = self.conv3(out1)

        return out
