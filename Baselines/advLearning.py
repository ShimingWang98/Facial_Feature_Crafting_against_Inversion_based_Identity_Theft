"""
Adversarial Training
Train Encoder(with prt), Decoder, and F(down stream)
"""

import os
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import itertools
from torchvision import datasets, transforms
import torch
import torch.optim as optim
from models import *
from fawkes_loader import *
import torch.nn as nn
import argparse


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-lam", "--lambda_1", type=float, help="lambda, or beta, the trade off coefficient")
    args = parser.parse_args()
    lambda_1 = args.lambda_1
    print(f'lambda is {lambda_1}')


    pub_attr = "../dataset/pub_attri.csv"
    pub_fig = "../dataset/public"
    pvt_attr = "../dataset/pvt_attri_train.csv"
    pvt_fig = '../dataset/private'
    save_model_dir = f'../params/130debug_norm/{lambda_1}'
    os.makedirs(save_model_dir, exist_ok=True)

    bs = 32
    _, pub_loader = init_dataloader(pub_attr, pub_fig, batch_size=32, n_classes=2, attriID=1, allAttri=True,
                                    normalization=True)
    _, pvt_loader = init_dataloader(pvt_attr, pvt_fig, batch_size=32, n_classes=2, attriID=1, allAttri=True,
                                    normalization=True)

    Enc = Encoder()
    F = F()
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=10)
    F.model.fc = torch.nn.Linear(in_features=512, out_features=10)
    Enc.model.load_state_dict(torch.load('../params/adversarialLearning/param_epoch_16.pt'))
    F.model.load_state_dict(torch.load('../params/adversarialLearning/param_epoch_16.pt'))

    Enc = torch.nn.DataParallel(Enc).cuda()
    F = torch.nn.DataParallel(F).cuda()

    Dec = Decoder(in_dim=64*32*32)
    Dec = nn.DataParallel(Dec).cuda()

    e_optimizer = optim.Adam(Enc.parameters())
    f_optimizer = optim.Adam(F.parameters())
    dec_optimizer = optim.Adam(Dec.parameters())

    loss_func_u = nn.BCEWithLogitsLoss()

    epoch = 500
    # lambda_1 = 0.5
    for T in range(epoch):
        total_loss_dec = 0.0
        total_loss_enc = 0.0
        cnt_1 = 0
        cnt_2 = 0

        # Update Dec^a and D while fixing Enc and f
        freeze(Enc)
        freeze(F)
        unfreeze(Dec)
        for i, (img, label) in enumerate(pub_loader): # should be pub
            if(img.shape[0]==1):
                continue
            x = img.cuda()
            # print("x.shape=",x.shape)
            z = Enc(x)
            # print("Enc(x).shape:",z.shape)
            x_hat = Dec(z)
            # print("x_hat.shape=",x_hat.shape)
            # exit()
            loss_p_a = torch.dist(x, x_hat, p=2)
            dec_optimizer.zero_grad()
            loss_p_a.backward()
            dec_optimizer.step()
            total_loss_dec = total_loss_dec + loss_p_a.item()
            cnt_1 = cnt_1 + 1
        print("epoch:{}\tdec_loss: {}, cnt {}".format(T, total_loss_dec / cnt_1, cnt_1))

        freeze(Dec)
        unfreeze(Enc)
        unfreeze(F)
        for i, (img, label) in enumerate(pvt_loader):
            # if i > 450:  # pvt training set
            #     break
            if(img.shape[0]==1):
                continue
            x = img.cuda()
            label = label[:,[0, 1, 2, 7, 9, 14, 16, 17, 19, 25]]
            y = label.cuda()
            z = Enc(x)
            pred = F(z)

            x_hat = Dec(z)

            ploss = torch.dist(x, x_hat, p=2)/(x.shape[-1]*x.shape[-2])
            uloss = loss_func_u(pred, y)
            # import pdb;pdb.set_trace()
            loss_u = (1-lambda_1)* uloss - lambda_1 * ploss
            e_optimizer.zero_grad()
            f_optimizer.zero_grad()

            loss_u.backward()
            e_optimizer.step()
            f_optimizer.step()

            total_loss_enc = total_loss_enc + loss_u.item()
            cnt_2 = cnt_2 + 1

        print("epoch:{}\tenc_loss: {}, u {} p {}".format(T, total_loss_enc / cnt_1, uloss, ploss))

        # Save models
        if T > 200 and (T+1)%50 == 0:
            torch.save(Enc.module.state_dict(), os.path.join(save_model_dir, f'Enc_l1_lambda{lambda_1}.pkl'))
            torch.save(Dec.module.state_dict(), os.path.join(save_model_dir, f'Dec_l1_lambda{lambda_1}.pkl'))
            torch.save(F.module.state_dict(), os.path.join(save_model_dir, f'F_l1_lambda{lambda_1}.pkl'))
