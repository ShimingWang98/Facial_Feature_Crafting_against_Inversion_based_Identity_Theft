"""
Adversarial training
Continue to train Decoder
Using (Enc(x), x) pairs where x is public
"""

import os
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import itertools
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import sys
sys.path.append('..')

from models import *
from loader import *

# def freeze(net):
#     for p in net.parameters():
#         p.requires_grad_(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lam", "--lambda_1", type=float, default=0.8,
                        help="lambda, or beta, the trade off coefficient")
    args = parser.parse_args()
    lambda_1 = args.lambda_1
    pub_attr = "../dataset/pub_attri.csv"
    pub_fig = "../dataset/public"

    os.makedirs(f'../params/decoder/', exist_ok=True)


    bs = 128
    # lambda_1 = 0.1
    _, pub_loader = init_dataloader(pub_attr, pub_fig, batch_size=bs, n_classes=2, attriID=1, allAttri=True,
                                    normalization=True)
    Enc = Encoder()
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(torch.load('../params/enc.pt'))
    Enc = torch.nn.DataParallel(Enc).cuda()
    freeze(Enc)
    Enc.eval()

    G = newGanGenerator(in_dim=500)
    G.load_state_dict(torch.load('../Gan/zdim_500/params/G_90.pkl'))
    G = torch.nn.DataParallel(G).cuda()
    G.eval()

    Dec = Decoder()
    Dec = nn.DataParallel(Dec).cuda()
    Dec.load_state_dict(torch.load("../params/dec.pkl"))
    Dec.train()
    dec_optimizer = optim.Adam(Dec.parameters())

    epoch = 500
    lr = 1e-3  # 5e-5, 1e-4, 5e-4, , 5e-3, 1e-2
    beta = 10
    dlr=0.0005
    flr=0.0001
    gp=10
    ncitic = 5
    print("Using beta:", beta)

    for T in range(epoch):
        total_loss_dec = 0.0
        cnt_1 = 0
        cnt_2 = 0
        for i, (img, label) in enumerate(pub_loader):

            if i == 30:  # 100
                break
            x = img.cuda()

            feature = torch.load(f"../Crafter_result/beta{beta}/features/{i}_447.pt")
            x_hat = Dec(feature)
            save_tensor_images(x_hat, "x_hat.jpg")
            save_tensor_images(x, "x.jpg")
            loss_p_a = torch.dist(x, x_hat, p=2)
            dec_optimizer.zero_grad()
            loss_p_a.backward()
            dec_optimizer.step()
            total_loss_dec = total_loss_dec + loss_p_a.item()
            cnt_1 = cnt_1 + 1
        print("epoch:{}\tdec_loss: {}".format(T, total_loss_dec / cnt_1))

        if (T + 1) % 10 == 0:
            torch.save(Dec.module.state_dict(), f'../params/decoder_adaptive/beta{beta}/decoder_epoch{T}.pkl')
    torch.save(Dec.module.state_dict(), f'../params/dec_adaptive.pkl')
