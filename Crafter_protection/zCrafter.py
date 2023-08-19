import sys
sys.path.append('..')
import os
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import itertools
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import sys
from WhiteBoxAttack.attacker import amor_inversion
from models import *
from loader import *
from plot_loss import *


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = netD(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def f_loss(f, f_rec):
    n, d = len(f), 64 * 64 * 64
    mse = torch.norm(f.view(n, -1) - f_rec.view(n, -1), dim=1) ** 2 / d
    return mse


def solve_z(feature, netG, netEnc, use_amortized=True):
    """
    Find initial x* of feature
    """
    netG.zero_grad()
    netEnc.zero_grad()
    if use_amortized:
        z = amor_inversion(netG, D, netEnc, Amor, feature, lr=5e-3, bs=bs, iter_times=110, z_dim=z_dim, save=False) #80
        z = z.clone().detach().requires_grad_(False)
    else:
        print("without amortize")
        z = inversion(netG, D, netEnc, feature, result_dir, lr=5e-3, bs=bs, iter_times=601, z_dim=z_dim, save=False)
        z = z.clone().detach().requires_grad_(False)
    netG.zero_grad()
    netEnc.zero_grad()
    return z



def attack_batch(f0, netG, netD, netEnc, batchID=0):
    """

    :param f0: original feature
    :param netEnc: Encoder network
    :return: optimal perturbation delta
    """
    d_optimizer = optim.Adam(netD.parameters(), lr=dlr)
    u_list = []
    p_list = []
    z_avg = torch.randn(bs, z_dim).cuda().float()
    z = solve_z(f0.clone().detach(), netG, netEnc, True).clone().detach().requires_grad_(True)  # initialize
    torch.save(z.detach().clone(), result_dir+'/z_init.pt')
    z_optimizer = optim.Adam([z], lr=zlr)

    for T in range(OUTER_EPOCH):
        torch.cuda.empty_cache()

        print("Epoch ", T)
        d_batch = 32

        for d_epoch in range(ncritic):  # update Discriminator
            real_index = torch.randint(0, bs, (d_batch,))
            real = netG(z_avg[real_index]).clone().detach()

            fake_index = torch.randint(0, bs, (d_batch,))
            fake_z = z[fake_index]
            fake = netG(fake_z).clone().detach()
            neg_logit = netD(fake)
            pos_logit = netD(real)

            gp = gradient_penalty(real, fake)
            l_gp = lambda_gp
            if gp > 1:
                l_gp = 30
            EMD = neg_logit.mean() - pos_logit.mean()
            d_loss = EMD + l_gp * gp
            p_list.append(torch.mean(EMD).data.cpu())
            print(" gp {}, d loss {} ".format(gp.item(), d_loss.item()))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            del fake
            del real

        loss_u = f_loss(netEnc(netG(z)), f0)
        loss_p = -netD(netG(z)).mean()

        loss = torch.mean(loss_p + beta * loss_u)
        z_optimizer.zero_grad()
        loss.backward()
        print(f'gradient of z is {torch.norm(z.grad.data)}')
        z_optimizer.step()

        print("Epoch {}, utility loss {}".format(T, torch.mean(loss_u)))
        u_list.append(torch.mean(loss_u).data.cpu())
        # p_list.append(torch.mean(d_loss).data.cpu())
        if T > 100:
            if T%20 == 0:
                torch.save(z, result_dir + '/features_z/{}_{}.pt'.format(batchID, T))
                inv_img = netG(z).clone().detach()
                save_tensor_images(inv_img, result_dir + '/images/inv_{}_{}.jpg'.format(batchID, T))
                del inv_img

    torch.save(p_list, result_dir + '/curves/ploss.pt')
    torch.save(u_list, result_dir + '/curves/uloss.pt')
    plot_loss(p_list, result_dir + '/curves', 'privacy_{}'.format(batchID), 'EM distance')
    plot_loss(u_list, result_dir + '/curves', 'utility_{}'.format(batchID), 'feature norm')
    del p_list
    del u_list
    del netD
    del z_avg
    return


if __name__ == "__main__":
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True

    attributePath = "../dataset/eval_test.csv"
    figurePath = "../dataset/private"
    bs = 128 # 32, 26
    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)

    z_dim = 500
    ATK_LR = 0.1

    G = newGanGenerator(in_dim=z_dim)
    G.load_state_dict(torch.load('../Gan/zdim_{}/params/G_90.pkl'.format(z_dim)))
    G = torch.nn.DataParallel(G).cuda()
    G.eval()

    D = GanDiscriminator()
    D.load_state_dict(torch.load('../Gan/zdim_{}/params/D_90.pkl'.format(z_dim)))
    D = torch.nn.DataParallel(D).cuda()
    D.eval()

    Amor = Amortizer(nz=z_dim)
    Amor.load_state_dict(torch.load('../params/amor.pkl'))
    Amor = torch.nn.DataParallel(Amor).cuda()
    Amor.eval()

    Enc = Encoder()
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(torch.load('../params/enc.pt'))
    Enc = torch.nn.DataParallel(Enc).cuda()
    Enc.eval()
    
    Beta = [100] # 10, 1, 5
    Lambda_gp = [15]
    Dlr = [5e-4] # , 5e-3, 5e-2
    Zlr = [5e-3] # , 0.05, 0.005
    Ncritic = [10] # 10,
    OUTER_EPOCH = 1#500

    for lambda_gp in Lambda_gp:
        for ncritic in Ncritic:
            for beta in Beta:
                for zlr in Zlr:
                    for dlr in Dlr:
                        torch.cuda.empty_cache()
                        result_dir = f'../zCraft_result/ncritic{ncritic}_beta{beta}_dlr{dlr}_zlr{zlr}/'
                        os.makedirs(result_dir + '/original', exist_ok=True)
                        os.makedirs(result_dir + '/images', exist_ok=True)
                        os.makedirs(result_dir + '/features_z', exist_ok=True)
                        os.makedirs(result_dir + '/curves', exist_ok=True)

                        for (i, (imgs, label)) in enumerate(dataloader):
                            torch.cuda.empty_cache()
                            netD = GanDiscriminator()

                            netD = torch.nn.DataParallel(netD).cuda()
                            # if i == 1:
                            #     break
                            bs = imgs.shape[0]
                            print("Batch", i)

                            imgs = imgs.cuda()
                            save_tensor_images(imgs, result_dir + 'original/original_{}.jpg'.format(i))

                            target_feature = Enc(imgs).clone().detach().requires_grad_(False)
                            attack_batch(target_feature, G, netD, Enc, batchID=i)
                            del netD
                            del target_feature
