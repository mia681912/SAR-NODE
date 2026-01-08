# code for SAR-NODE
# Author: ZiQing Ma
# 2024/09/10

import torch.nn as nn
from torch.utils import data
from model import HaarSolve
from readfromh5 import DatasetFromHdf5
import matplotlib
from compute import *
matplotlib.use('Agg')
import os
import glob
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms, utils
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from argas_haar import args_n
import cv2 as cv

opt = args_n()


# ================== Pre-Define =================== #
SEED = opt.random_seed
torch.manual_seed(SEED)
seed = np.random.RandomState(44999)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum

# 文件保存路径
if not os.path.isdir(opt.save_dir):
    os.makedirs(opt.save_dir)
model_folder = os.path.join(opt.save_models, 'model')
model_folder_best = os.path.join(opt.save_models, 'model_best')


argsDict = opt.__dict__
with open(os.path.join(opt.save_dir, 'setting.txt'), 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')
    f.close()

# load parameter
lr = opt.lr
epochs = opt.epochs
batch_size = opt.batch_size

# load model
model = HaarSolve(opt).cuda()
PLoss = nn.L1Loss().cuda()


optimizer = optim.Adam(model.parameters(), lr = 1e-2,
                            weight_decay = 1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                            step_size = 5,
                                            gamma = 0.1)


def save_checkpoint(model_folder1, model, epoch):
    model_out_path = os.path.join(model_folder1, '{}.pth'.format(epoch))

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder1):
        os.makedirs(model_folder1)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train ----------------------------------
###################################################################

def train(start_epoch=0, RESUME=False):

    trainset = DatasetFromHdf5(opt.trainh5)
    # trainset = Dataset(opt.trainh5)
    trainload = data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)

    valset = DatasetFromHdf5(opt.valh5)
    # valset = Dataset(opt.valh5)
    valload = data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False)
   
    if RESUME:
        path_checkpoint = model_folder + "{}.pth".format(opt.depoch)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))

    time_s = time.time()
    lr_now = 0
    psnr_best = 0
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_valid_loss = [], []
        psnr_train, psnr_valid = [], []
        ssim_train, ssim_valid = [], []
        lr_list = []

        # ============Epoch Train=============== #
        model.train()

        for batch_count, batch_data in enumerate(trainload):
            
            img_clean = batch_data[0].cuda()
            img_noise = batch_data[1].cuda()

            # ============================train=========================================
            # print(torch.max(torch.max(img_noise)))
            pred_image = model(img_noise.cuda())
            # print(torch.max(torch.max(pred_image)))
            # =========================calculate loss========================================
            loss = PLoss(pred_image.float(), img_clean.float())
            
            epoch_train_loss.append(loss.item())
            #=========================参数更新=========================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            psnr_train.append((psnr_cal(torch.clamp(img_clean,0,255.), torch.clamp(pred_image,0,255.))).item())
            # ssim_train.append(compare_ssim(img_clean.cpu().detach().numpy().squeeze(0), pred_image.cpu().detach().numpy().squeeze(0), channel_axis=2))

            if (batch_count + 1) % 20 == 0:
                print("iter [{}]--[{}]/[{}]: train_loss is {}".format(epoch, batch_count, len(trainload), loss))

        lr_list.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()  # update lr

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_psnr = sum(psnr_train) / len(psnr_train)

        with open(os.path.join(opt.save_dir, 'train.txt'), 'a+') as fp:
            fp.write('epoch {}, train loss = {:.4f}, train PSNR = {:.4f}\n'.format(epoch, train_loss, train_psnr))
        fp.close()
        
        save_checkpoint(model_folder, model, epoch)
        print('Start validation...')
        validation_loss = 0
        validation_psnr = 0
        torch.set_grad_enabled(False)
    
        model.eval()
        j = 0

        for batch_count, batch_valid in enumerate(valload):
            img_clean = batch_valid[0].cuda()
            img_noise = batch_valid[1].cuda()
            
            img_est = model(img_noise.cuda())
            loss_valid = PLoss(img_est.float(), img_clean.float())
            epoch_valid_loss.append(loss_valid.item())
            psnr_valid.append((psnr_cal(torch.clamp(img_clean,0.,1.), torch.clamp(img_est,0.,1.))).item())
            # ssim_valid.append(compare_ssim(img_clean, img_est, channel_axis=2))
            j += 1

        valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
        psnr_valid = sum(psnr_valid) / len(psnr_valid)
        with open(os.path.join(opt.save_dir, 'valid.txt'), 'a+') as fp:
            fp.write('epoch {}, valid loss = {:.4f}, valid PSNR = {:.4f}\n'.format(epoch, valid_loss, psnr_valid))
        fp.close()

        if epoch > 1 and psnr_best<psnr_valid:
            save_checkpoint(model_folder_best, model, epoch)

        psnr_best = psnr_valid
        print('validation, validation loss is:' + str(valid_loss))

        average_test_psnr = test(epoch)
        with open(os.path.join(opt.save_dir, 'index.txt'), 'a+') as fp:
            fp.write('epoch {}, test PSNR = {:.4f}\n'.format(epoch, average_test_psnr))
        fp.close()
        torch.set_grad_enabled(True)


def test(epoch):
    save_path = ''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_img_dir = os.path.join(save_path, 'visual')
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)
    model_folder = os.path.join(opt.save_models, 'model')
    path_checkpoint = os.path.join(model_folder, '{}.pth'.format(epoch))
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model.eval()

    torch.set_grad_enabled(False)

    imgs_test = glob.glob(os.path.join(opt.test_data, '*.tif'))
    imgs_test.sort()
    # print(imgs_test)

    val_psnr = torch.zeros(len(imgs_test))
    val_ssim = torch.zeros(len(imgs_test))

    for idx, img_test in enumerate(imgs_test):
        print(idx)
        img_test = Image.open(img_test)
        img_test = img_test.convert('L')
        img_test = np.expand_dims(img_test, axis=0)
        img_test_np = np.array(img_test)
        
        noise = seed.gamma(size=img_test_np.shape, shape=opt.L, scale=1 / opt.L)
        img_noise = img_test_np * noise
        

        img_test_tensor = torch.from_numpy(img_test_np)
        img_noise = torch.from_numpy(img_noise)
        img_clean = img_test_tensor.unsqueeze(0).cuda()
        img_noise = img_noise.unsqueeze(0).cuda()
        img_est = model(img_noise.float()).detach()
        # print(torch.max(img_clean))

        psnr_test = psnr_cal(torch.clamp(img_clean,0.,255.), torch.clamp(img_est,0.,255.))
        ssim_test = ssim_cal(torch.clamp(img_clean,0.,255.), torch.clamp(img_est,0.,255.), 255.)
        with open(os.path.join(save_path, 'index.txt'), 'a+') as fp:
            fp.write("image {}: psnr={:.4f} ssim= {:.4f}\n".format(idx, psnr_test, ssim_test))
        fp.close()

        img_est1 = torch.squeeze(img_est, dim=1)
        img_est1 = torch.squeeze(img_est1, dim=0)
        img_est1 = img_est1.cpu().detach().numpy()

        img_noise1 = torch.squeeze(img_noise, dim=1)
        img_noise1 = torch.squeeze(img_noise1, dim=0)
        img_noise1 = img_noise1.cpu().detach().numpy()

        print("image {}: psnr={:.7f} \n".format(idx, psnr_test))

        val_psnr[idx] = psnr_test
        val_ssim[idx] = ssim_test

        save_img_path = os.path.join(save_img_dir, '{}.jpg'.format(idx))
        cv.imwrite(save_img_path, img_est1)
        save_img_path1 = os.path.join(save_img_dir, '{}_noise.jpg'.format(idx))
        cv.imwrite(save_img_path1, img_noise1)
    psnr_average = torch.mean(val_psnr)
    ssim_average = torch.mean(val_ssim)
    with open(os.path.join(save_path, 'index.txt'), 'a+') as fp:
        fp.write("average psnr={:.4f} ssim= {:.4f}\n".format(psnr_average, ssim_average))
    fp.close()
    return psnr_average


###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    train()


