import os
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import itertools
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn
import sys
sys.path.append('..')

from attacker import inversion
from loader import *
from models import *


def loss_func(inverted_img, img):
    trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inverted_img, img = trans(inverted_img), trans(img)
    bce = torch.dist(inverted_img, img)
    # bce = torch.nn.functional.binary_cross_entropy(inverted_img, img, reduction='sum')
    return bce

if __name__ == "__main__":
    bs = 128
    z_dim = 500
    attributePath = "../dataset/identity_pub_CelebA.txt"
    figurePath = "../dataset/public"

    G = newGanGenerator(in_dim=z_dim)
    G.load_state_dict(torch.load('../params/G_90.pkl'))
    G = torch.nn.DataParallel(G).cuda()
    G.eval()
    Amor = Amortizer(nz=z_dim)
    Amor.load_state_dict(torch.load('../params/amor.pkl'))
    Amor = torch.nn.DataParallel(Amor).cuda()
    Amor.eval()

    D = GanDiscriminator()
    D.load_state_dict(torch.load('../params/D_90.pkl'))
    D = torch.nn.DataParallel(D).cuda()
    D.eval()
    Enc = Encoder()
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(torch.load('../params/enc.pt'))
    Enc = torch.nn.DataParallel(Enc).cuda()
    Enc.eval()
    D.eval()
    Enc.eval()

    beta = 50
    zlr = 1e-4

    result_dir = f'../params/adaptiveG/beta{beta}_zlr{zlr}/'
    os.makedirs(result_dir + 'z/', exist_ok=True)
    os.makedirs(result_dir + 'G/', exist_ok=True)
    os.makedirs(result_dir + 'img/', exist_ok=True)
    epoch = 50
    g_optimizer = optim.RMSprop(G.parameters(), lr=0.001)
    d_score = []
    final_loss_list = []

    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)


    for e in range(epoch):
        loss_list = []
        print("epoch", e)
        z_avg = torch.randn((8, 500)).cuda()
        save_tensor_images(G(z_avg), result_dir + 'img/img_epoch_{}.jpg'.format(e))

        for (i, (imgs, label)) in enumerate(dataloader):
            if i == 50:
                break

            imgs = imgs.cuda()
            released_feature =torch.load(f"./Crafter_result/features/{i}_447.pt")

            current_inv_z, _ = inversion(G, D, Enc, released_feature, result_dir, lr=5e-3, z_dim=z_dim, save=False)
            if i == 0:  # each epoch save the first batch
                # torch.save(current_inv_z, result_dir+'z/z_epoch_{}.pt'.format(e))
                fake_val, real_val = -torch.mean(D(G(z_avg))), torch.mean(D(imgs))
                # save_tensor_images(G(current_inv_z).detach(), result_dir+'img/inv_epoch{}.jpg'.format(e))
                print("d score is {}".format(fake_val + real_val))
                d_score.append(fake_val + real_val)


            g_optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            loss = loss_func(G(current_inv_z), imgs)
            loss_list.append(loss)
            loss.backward()
            g_optimizer.step()

        print("loss is {}".format(torch.mean(torch.tensor(loss_list))))
        final_loss_list.append(torch.mean(torch.tensor(loss_list)))
        torch.save(G.module.state_dict(), result_dir + 'G/G_epoch_{}.pkl'.format(e))
        torch.save(d_score, result_dir + 'score.pt')
        torch.save(final_loss_list, result_dir + 'loss.pt')
    
