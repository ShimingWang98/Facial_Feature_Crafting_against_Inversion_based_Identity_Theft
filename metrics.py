import os
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import time
import torch
import PIL
import torchvision.utils as tvls
from PIL import Image
from torch.autograd import grad
import torch.nn.functional as F
from math import exp
import math


# def eval_acc(Eval, fake_img, real_img, batch_size):
#     eval_fake = Eval(fake_img)
#     eval_real = Eval(real_img)
#     id_fake = torch.argmax(eval_fake, dim=1).view(-1)
#     id_real = torch.argmax(eval_real, dim=1).view(-1)
#     acc = id_real.eq(id_fake.long()).sum().item() * 1.0 / batch_size
#     return acc

def eval_acc(Eval, fake_img, label):
    batch_size = fake_img.shape[0]
    eval_fake = Eval(fake_img)
    id_fake = torch.argmax(eval_fake, dim=1).view(-1) + torch.ones(batch_size).cuda()
    id_real = label
    # print("predicted", id_fake)
    # print("real", label)
    acc = id_real.eq(id_fake.long()).sum().item() * 1.0 / batch_size
    return acc

def feature_sim(Eval, x1, x2):
    x1 = Eval(x1,toEnd=False)
    x2 = Eval(x2,toEnd=False)
    return torch.mean(torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-08)).item()



def eval_acc_top5(Eval, fake_img, label):
    batch_size = fake_img.shape[0]
    eval_fake = Eval(fake_img)
    id_fake = torch.argmax(eval_fake, dim=1).view(-1) + torch.ones(batch_size).cuda()
    id_fake_top5 = torch.topk(eval_fake, 5, dim=1)[1]
    print(id_fake_top5)
    id_real = label - torch.ones(batch_size).cuda()
    correct = 0
    for i in range(batch_size):
        if id_real[i] in id_fake_top5[i]:
            correct = correct + 1
    # acc = id_real.eq(id_fake.long()).sum().item() * 1.0 / batch_size
    acc = correct / batch_size
    return acc


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# create Gaussian Kernel
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Calculate SSIM
# Use formula directly, but get the final value using normalized Gaussian kernel rather than averaging
# Lemma: Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
def ssim(img_set1, img_set2, window_size=11, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    # ret_all = 0
    # for i in range(bs):
    img1 = img_set1
    img2 = img_set2
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    # import pdb;pdb.set_trace()
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    # ret_all = ret + ret_all
    # ret_all = ret_all / bs

    return ret
