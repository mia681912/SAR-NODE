"""
calculate psnr, ssim
GPU version
coder: Ziqing Ma  time:2024/09/12
"""
import torch
import numpy as np


def rmse_cal(img_tgt, img_fus):
    """

    :param img_tgt:  [n,c,h,w]
    :param img_fus:  [n,c,h,w]
    :return: average square error
    """

    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = (np.mean(abs(img_tgt - img_fus)))

    return rmse


def psnr_cal(img_tgt, img_fus):
    """

    :param img_tgt:  [n,c,h,w]
    :param img_fus:  [n,c,h,w]
    :return: average PSNR
    """
    img_tgt = img_tgt.reshape([img_tgt.shape[0]*img_tgt.shape[1], -1])  # [nc,h*w]
    img_fus = img_fus.reshape([img_fus.shape[0]*img_fus.shape[1], -1])   # [nc, h*w]
    # img_tgt = img_tgt * 255.
    # img_fus = img_fus * 255.
    mse = torch.mean(torch.pow(img_tgt - img_fus, 2), axis=1)
    img_max = 255.
    psnr = 10 * torch.log10(img_max ** 2 / mse)
    # print(psnr)

    return torch.mean(psnr)

def ssim_cal(x1, x2, max_v, k1=0.01, k2=0.03):
    """

        :param x1:  [n,c,h,w]
        :param x2:  [n,c,h,w]
        :return: average ssim
    """

    h, w = x1.shape[2], x1.shape[3]
    x1 = x1.reshape(x1.shape[0] * x1.shape[1], -1)
    x2 = x2.reshape(x2.shape[0] * x2.shape[1], -1)
    u1 = torch.mean(x1, axis=1).reshape([-1, 1])
    u2 = torch.mean(x2, axis=1).reshape([-1, 1])
    Sig1 = torch.std(x1, axis=1).reshape([-1, 1])
    Sig2 = torch.std(x2, axis=1).reshape([-1, 1])
    sig12 = torch.sum((x1 - u1) * (x2 - u2), axis=1) / (h * w - 1)
    sig12 = sig12.reshape([-1, 1])
    c1, c2 = pow(k1 * max_v, 2), pow(k2 * max_v, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))

    return torch.mean(SSIM)
