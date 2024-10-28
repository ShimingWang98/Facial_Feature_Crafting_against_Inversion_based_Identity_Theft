import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import grad

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import urllib3

from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gender_idx = 20
target_idx = gender_idx


def gradient_penalty(x, y, c):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = discriminator(z, c)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training [deprecated]")
    parser.add_argument("--n_iters", type=int, default=100000, help="number of iterations of training")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=4096, help="size of training dataset")
    # parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--lr", type=float, default=4e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    opt = parser.parse_args()
    print(opt)

    latent_dim = opt.latent_dim
    bs = 64

    # Loss weight for gradient penalty
    lambda_gp = 10
    print("latent dimension ", latent_dim)
    print("learning rate", opt.lr)
    # Initialize generator and discriminator
    generator = GanGenerator(z_dim=latent_dim, y_dim=2)
    generator = nn.DataParallel(generator).cuda()
    discriminator = GanDiscriminator(y_dim=2)  # 3
    discriminator = nn.DataParallel(discriminator).cuda()

    trans_n = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])
    # trans = transforms.Compose([trnas_resize, trans_tensor])
    dataset = torchvision.datasets.CelebA('../dataset/', split='train', target_type='attr', transform=trans,
                                          target_transform=None, download=False)
    torch.manual_seed(0)
    length = len(dataset)
    dataset, _ = torch.utils.data.random_split(dataset, [opt.data_num, length - opt.data_num])

    dataloader = torch.utils.data.DataLoader(dataset, bs)
    # import pdb;pdb.set_trace()

    result_dir = '../Gan/zdim_{}/images/'.format(opt.latent_dim)
    os.makedirs(result_dir, exist_ok=True)
    store_param = '../Gan/zdim_{}/params/'.format(opt.latent_dim)
    os.makedirs(store_param, exist_ok=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor

    # ----------
    #  Training
    # ----------
    batches_done = 0
    epoch = 0
    while batches_done < opt.n_iters:
        for i, (imgs, cond) in enumerate(dataloader):
            epoch += 1
            # Configure input
            real_imgs = imgs.cuda()
            cond = cond[:, target_idx].cuda()
            cond = torch.nn.functional.one_hot(cond, 2)
            # Sample noise as generator input
            z = torch.randn((imgs.shape[0], latent_dim), requires_grad=True).cuda()

            # Generate a batch of images
            fake_imgs = generator(z, cond)

            # Real images
            real_validity = discriminator(real_imgs, cond)
            fake_validity = discriminator(fake_imgs, cond)
            gp = gradient_penalty(real_imgs.data, fake_imgs.data, cond.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if (batches_done + 1) % opt.n_critic == 0:
                # Generate a batch of images
                fake_imgs = generator(z, cond)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs, cond)

                # z1, z2 = torch.randn(bs, latent_dim).cuda(), torch.randn(bs, latent_dim).cuda()
                # loss_div = -torch.mean(torch.norm(Enc(generator(z1)) - Enc(generator(z2))) / torch.norm(z1 - z2))
                g_loss = -torch.mean(fake_validity)  # + 0.01 * loss_div

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss_div: %s]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), "deprecated")
                    # loss_div.item()
                )
                print("gp is {}".format(gp))

            batches_done += 1

            if batches_done % (opt.n_critic * 1000) == 0:
                torch.save(generator.module.state_dict(), store_param + 'G_{}.pkl'.format(batches_done))
                torch.save(discriminator.module.state_dict(), store_param + 'D_{}.pkl'.format(batches_done))
                save_image(fake_imgs.detach().data, result_dir + '/img_{}.jpg'.format(batches_done))

    torch.save(generator.module.state_dict(), store_param + 'G_{}.pkl'.format(batches_done))
    torch.save(discriminator.module.state_dict(), store_param + 'D_{}.pkl'.format(batches_done))
    save_image(fake_imgs.detach().data, result_dir + '/img_{}.jpg'.format(batches_done))
