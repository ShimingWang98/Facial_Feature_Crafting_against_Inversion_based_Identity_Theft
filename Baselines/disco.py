## LFW dataset, 10 utility attributes, privacy attribute is identity, disco protection.
## Enc, F, Decoder snippet from LFW_advlearn

## Disco baseline code notes
# main.py ->
# scheduler.py run_defense_job() -> defense_train() ----> algo.forward() model.forward() algo.infer() model.backward() -> algo.backward()
# algo: edge-end (class Disco); model: server-end (resnet 18)
# all functions of algo: disco.py
# all functions of model: model_zoo.py

import os
import time
import matplotlib.pyplot as plt
import pickle
import itertools
import torchvision
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import sys
sys.path.append('..')
from loader_verbose import *
import torch.nn as nn
import argparse


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.layers = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class F(nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.layers = nn.Sequential(
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool,
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x



class Decoder(nn.Module):
    def __init__(self, in_dim=64 * 32 * 32, dim=64):
        super(Decoder, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 4, padding=1, output_padding=1),
            nn.Sigmoid())
    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.l1(y)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y



class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, channels):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = 0.99 # config["pruning_ratio"]
        self.split_layer = 5 # split_layer

        self.temp = 1/30
        self.logits = channels #config["channels"]
        self.model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(list(self.model.children())[self.split_layer:])
        self.model = nn.Sequential(*self.model)


    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #print(exponent)
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        #print(answer)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        return fmap_score

    def network_forward(self, x):
        return self.model(x)

    def forward(self, x):

        channel_score = self.get_channels_from_network(x, self.pruning_ratio)
        x = x*channel_score.unsqueeze(-1).unsqueeze(-1)
        return x


def generate_patch_params(client_model, img_size):
    img = torch.randn(1, 3, img_size, img_size)
    patch = client_model(img)
    assert patch.shape[2] == patch.shape[3]  # only square images
    return patch.shape[1], patch.shape[2]


if __name__ == "__main__":
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True
    bs = 64

    # initialize
    pub_attr = "/home/shiming/LFW_advlearn/dataset/pub_attri.csv"
    pub_fig = "/home/shiming/LFW_advlearn/dataset/public"
    _, ploader = init_dataloader(pub_attr, pub_fig, batch_size=bs, n_classes=2, attriID=1, allAttri=True,
                                    normalization=True)
    # pvt_attr = "/home/shiming/LFW_advlearn/dataset/pvt_attri_train.csv"
    # pvt_fig = '/home/shiming/LFW_advlearn/dataset/private'
    # _, ploader = init_dataloader(pvt_attr, pvt_fig, batch_size=bs, n_classes=2, attriID=1, allAttri=True,
    #                                 normalization=True)

    Enc = Encoder()  # client model
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=10)
    img_size = 128
    channels, patch_size = generate_patch_params(Enc, img_size)  # line 22, disco.py
    alpha = 0.4  # tradeoff parameter
    save_model_dir = f'params/disco_{alpha}/'
    os.makedirs(save_model_dir, exist_ok=True)

    Pruner = PruningNetwork(channels)
    F = F() # server model
    Dec = Decoder() # proxy adversary model
    F.model.fc = torch.nn.Linear(in_features=512, out_features=10)

    Pruner = torch.nn.DataParallel(Pruner).cuda()
    Enc = torch.nn.DataParallel(Enc).cuda()
    F = torch.nn.DataParallel(F).cuda()
    Dec = torch.nn.DataParallel(Dec).cuda()

    p_optimizer = optim.Adam(Pruner.parameters(), lr=3e-4)
    e_optimizer = optim.Adam(Enc.parameters(), lr=3e-4)
    f_optimizer = optim.Adam(F.parameters(), lr=3e-4)
    dec_optimizer = optim.Adam(Dec.parameters(), lr=3e-4)

    Enc.train()
    F.train()

    # Training
    for T in range(100):
        for i, (img, label) in enumerate(ploader):
            if (img.shape[0] != bs):
                continue
            x = img.cuda()
            label = label[:, [0, 1, 2, 7, 9, 14, 16, 17, 19, 25]]
            y = label.cuda()

            ## forwarding through client
            unpruned_feature_out = Enc(x)
            unpruned_feature_in = unpruned_feature_out.detach()
            unpruned_feature_in.requires_grad = True

            pruned_feature_out = Pruner(unpruned_feature_in)
            pruned_feature_in = pruned_feature_out.detach()
            pruned_feature_in.requires_grad = True

            x_recons = Dec(pruned_feature_in)
            adv_loss = torch.dist(x, x_recons, p=2)

            pruned_feature_in = pruned_feature_out.detach()
            pruned_feature_in.requires_grad = True # why

            ## forwarding through server  (model_zoo.py forward())
            pruned_feature_in.retain_grad()
            pred = F(pruned_feature_in)

            ## backwarding through server
            utility_loss = nn.BCEWithLogitsLoss()
            loss_u = utility_loss(pred, y)
            f_optimizer.zero_grad()
            loss_u.backward()
            f_optimizer.step()
            server_grad = pruned_feature_in.grad

            if i==0:
                print(f'utility loss {loss_u}; privacy loss {adv_loss}')

            ## backwarding through client, update Enc, Pruner and Adv
            e_optimizer.zero_grad()
            p_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            # first update adv
            adv_loss.backward()
            dec_optimizer.step()
            # next update pruner
            pruned_feature_out.backward(server_grad, retain_graph=True)
            unpruned_feature_out.backward(unpruned_feature_in.grad)
            e_optimizer.step()
            for params in Pruner.parameters():
                params.grad *= (1 - alpha)

            pruned_feature_out.backward(-1*alpha*pruned_feature_in.grad)
            p_optimizer.step()

        if T > 80 and (T+1)%10 == 0:
            torch.save(Enc.module.state_dict(), os.path.join(save_model_dir, f'Enc_alpha{alpha}_epoch{T}.pkl'))
            torch.save(Dec.module.state_dict(), os.path.join(save_model_dir, f'Dec_alpha{alpha}_epoch{T}.pkl'))
            torch.save(F.module.state_dict(), os.path.join(save_model_dir, f'F_alpha{alpha}_epoch{T}.pkl'))
            torch.save(Pruner.module.state_dict(), os.path.join(save_model_dir, f'Pruner_alpha{alpha}_epoch{T}.pkl'))
