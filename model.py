from tkinter import N
import torch
import os
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from basiclayer import *
import cv2
opt = args_n()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum

class De_solver(nn.Module):
    def __init__(self, t, N_t, solver, ll_channel, ll_feature, denoise_num, device, odeint_adjoint):
        super(De_solver, self).__init__()

        self.odeint_adjoint = odeint_adjoint
        self.t = t
        self.N_t = N_t
        self.solver = solver
        self.ODE_vector_field = LL_denoise(ll_channel, ll_feature, denoise_num, device)

    def forward(self, noise_image: torch.Tensor):
        if self.odeint_adjoint:
            odeint_solver = odeint_adjoint
        else:
            odeint_solver = odeint
        timesteps = torch.from_numpy(np.linspace(0, self.t, self.N_t + 1)).cuda()
        out = odeint_solver(func=self.ODE_vector_field, y0=noise_image, t=timesteps, method=self.solver)[-1]

        return out


class HaarSolve(nn.Module):
    def __init__(self, args):
        super(HaarSolve, self).__init__()

        # self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        self.in_channel = args.num_channel
        self.out_feature = args.out_channel

        self.img_dwt = DWT()
        self.img_iwt = IWT()

        self.conv_ll = nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1)
        self.conv_lh = nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1)
        self.conv_hl = nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1)
        self.conv_hh = nn.Conv2d(self.in_channel, self.out_feature, 3, padding=1)

        self.ll_de = De_solver(args.t, args.N_t, args.solver, self.out_feature, 
        self.out_feature, args.denoise_num, args.device, args.odeint_adjoint) 
        self.lh_sr = Super_Attention(self.out_feature, self.out_feature)
        self.hl_sr = Super_Attention(self.out_feature, self.out_feature)
        self.hh_sr = Super_Attention(self.out_feature, self.out_feature)
        self.fusion1 = nn.Conv2d(int(4 * self.out_feature), self.out_feature, 1)
        self.fusion2 = nn.Conv2d(self.out_feature, 4, 1)

        self.renew = nn.Sequential(
            nn.Conv2d(self.out_feature, self.out_feature // 2, 3, padding=1),
            nn.BatchNorm2d(self.out_feature // 2),
            nn.ReLU(),
            nn.Conv2d(self.out_feature // 2, self.out_feature, 3, padding=1),
            nn.BatchNorm2d(self.out_feature),
            nn.ReLU()
        )

    def forward(self, x):
        x_dwt = self.img_dwt(x)
        input_ll = self.conv_ll(x_dwt[:,0:1,:,:])       
        input_lh = self.conv_lh(x_dwt[:,1:2,:,:])
        input_hl = self.conv_hl(x_dwt[:,2:3,:,:])
        input_hh = self.conv_hh(x_dwt[:,3:4,:,:])

        x_de = self.ll_de(input_ll)
        x_lh = self.lh_sr(input_lh)
        x_hl = self.hl_sr(input_hl)
        x_hh = self.hh_sr(input_hh)

        x_cat = torch.cat([x_de, x_lh, x_hl, x_hh], dim=1)
        x_cat = self.fusion1(x_cat)

        out = x_cat + self.renew(x_cat)
        out = self.img_iwt(self.fusion2(out))

        return out
