import random
import sys
import os
import torch.optim as optim
from metrics import *
from models import *
from loader import *
import matplotlib.pyplot as plt
import numpy
import argparse
from facenet_pytorch import InceptionResnetV1
import sklearn.metrics
from statistics import mean
import csv

def my_auc(labelll, preddd):
    auclist = []
    labelll = labelll.detach().cpu()
    preddd = preddd.detach().cpu()
    for i in range(label.shape[1]):
        try:
            a = sklearn.metrics.roc_auc_score(labelll[:,i], preddd[:,i])
        except:
            pass
        else:
            auclist.append(a)
    return auclist

def plot_loss(loss_list, directory, name, title):
    '''
    name: filename
    title: plot title
    '''
    x = range(len(loss_list))
    plt.plot(x, loss_list)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + name + '.png')  # './curves/gp/'
    plt.close()


def inversion(G, Enc, feature_real, result_dir, name='', title='',
              lr=5e-3, momentum=0.9, iter_times=601, z_dim=500, save=True):
    '''
    White box inversion, momentum descent
    G: generator
    Enc: target network
    feature_real: feature to invert
    result_dir: where to store inverted images
    batch: to generate random seed
    name: inversion loss curve file name
    tilte: inversion loss curve title
    '''
    bs = feature_real.shape[0]

    G.eval()
    Enc.eval()
    # print(">>>> Saving to {}".format(result_dir))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    z = torch.randn(bs, z_dim).cuda().float()  # randn
    z.requires_grad = True
    v = torch.zeros(bs, z_dim).cuda().float()

    loss_list = []
    for _ in range(iter_times):
        fake = G(z)
        feature_fake = Enc(fake)
        if z.grad is not None:
            z.grad.data.zero_()
        feature_loss = torch.dist(feature_fake, feature_real, p=2)
        total_loss = 100 * feature_loss#  + 100 * prior_loss
        total_loss.backward()
        loss_list.append(feature_loss.data.float().cpu())

        v_prev = v.clone()
        gradient = z.grad.data.clone()
        v = momentum * v_prev + gradient
        z = z - lr * v
        z = z.detach()
        z.requires_grad = True
    if save:
        save_tensor_images(fake.detach(), result_dir + "/inverted.jpg", nrow=8)
        plot_loss(loss_list, result_dir, 'loss curve', 'inv loss')
    return z


def amor_inversion(G, D, Enc, Amor, feature_real, result_dir='',
                   lr=5e-3, momentum=0.9, iter_times=80, z_dim=500, save=False):
    '''
    White box inversion, using amor to set initial z
    G: generator
    Enc: target network
    feature_real: feature to invert
    '''
    G.eval()
    Amor.eval()
    Enc.eval()
    bs = feature_real.shape[0]

    z = Amor(feature_real).clone().detach().cuda()
    z.requires_grad = True
    v = torch.zeros(bs, z_dim).cuda().float()

    loss_list = []
    for _ in range(iter_times):
        fake = G(z)
        feature_fake = Enc(fake)
        if z.grad is not None:
            z.grad.data.zero_()
        feature_loss = torch.dist(feature_fake, feature_real, p=2)
        prior_loss = - D(fake).mean()
        total_loss = 100 * feature_loss + 100 * prior_loss
        total_loss.backward()
        loss_list.append(feature_loss.data.float().cpu())

        v_prev = v.clone()
        gradient = z.grad.data.clone()
        v = momentum * v_prev + gradient
        z = z - lr * v
        # z = torch.clamp(z.detach(), -1.0, 1.0).float()
        z = z.detach()
        z.requires_grad = True
    if save:
        save_tensor_images(fake.detach(), result_dir + "/inverted.jpg", nrow=8)
        plot_loss(loss_list, result_dir, 'loss curve', 'inv loss')
    return z


if __name__ == '__main__':
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-zd", "--z_dimension", type=int, default=500, help="dimension of latent z")
    parser.add_argument("-lam", "--lambda_1", type=float, default=0.5,
                        help="parameter of black box protection baseline")
    args = parser.parse_args()
    zd = args.z_dimension
    lambda_1 = args.lambda_1
    print("z dimension ", zd)

    G = newGanGenerator(in_dim=zd)
    G.load_state_dict(torch.load('../Gan/zdim_{}/params/G_90.pkl'.format(zd)))
    G = torch.nn.DataParallel(G).cuda()
    G.eval()
    D = GanDiscriminator()
    D.load_state_dict(torch.load('../Gan/zdim_{}/params/D_90.pkl'.format(zd)))
    D = torch.nn.DataParallel(D).cuda()
    D.eval()
    Enc = Encoder()
    Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(torch.load('../params/enc.pt'))
    Enc = torch.nn.DataParallel(Enc).cuda()
    Enc.eval()
    Amor = Amortizer(nz=zd)
    Amor.load_state_dict(torch.load('../params/amor.pkl'))
    Amor = torch.nn.DataParallel(Amor).cuda()
    Amor.eval()
    Dec = Decoder()
    Dec = torch.nn.DataParallel(Dec).cuda()
    Dec.load_state_dict(torch.load('../params/dec.pkl'))
    Dec.eval()
    Eval_net = Eval(1000)
    Eval_net = nn.DataParallel(Eval_net).cuda()
    Eval_net.load_state_dict(torch.load('../params/eval.pkl'))
    Eval_net.eval()

    INV, EVAL_ID, UTIL = True, True, True


    ATTACK_LIST = ['black','white'] #, 'white'
    ift_i, alpha = 120, 0.001
    # beta_list, flr_list = [2, 1, 0.2, 10], [0.01] # 5, 10, 20,   0.001
    beta_list, flr_list = [10,2, 1, 0.5], [0.01] # , 2, 1, 0.5
    for flr in flr_list:
        # f = open(f'plot/csv/ift_flr{flr}_alpha{alpha}_newfloss_noamor.csv', 'w', encoding='utf-8')
        f = open(f'plot/csv/ift_ssim.csv', 'w', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['', '', "eval acc", "", "feature sim", '', "ssim", "", "utility"])
        csv_writer.writerow(["flr", "beta", "white", "black", "white", "black", "white", "black", ""])
        for beta in beta_list:
            print(f"\n\ntesting ours_ift, beta{beta}, flr{flr}")
            for ATTACK in ATTACK_LIST:
                print('\n-----------------------'+ATTACK)
                inv_result_dir = f'../result/ift_newfloss/{flr}_{beta}/{ATTACK}/'
                # inv_result_dir = f'../result/noprt/{ATTACK}/'
                os.makedirs(inv_result_dir, exist_ok=True)
                figurePath = "../dataset/private"
                attributePath = "../dataset/eval_test.csv"
                bs = 128  # 16

                img_name_list = []
                skip = 1
                with open(attributePath, 'r') as file:
                    for line in file:
                        if skip == 1:
                            skip = 0
                            continue
                        line = re.split(r',', line)
                        img_name_list.append(line[0].strip())
                file.close()

                num_batch = 1

                if INV:
                    _, dataloader = init_dataloader(attributePath, figurePath, action=f'inv_unprotected', batch_size=bs, n_classes=38, skiprows=1, attriID=1)  # inv_unprotected/inv_ours/inv_lowkey/inv_fawkes // eval // prt

                    for (batch_id, (imgs, label)) in enumerate(dataloader):
                            torch.cuda.empty_cache()
                            imgs = imgs.cuda()
                            label = label.cuda()
                            feature = torch.load(f'Crafter_result/i{ift_i}_alpha{alpha}_flr{flr}/beta{beta}/features/0_447.pt')

                            evalacclist, loss_list, fsim_list = [], [], []
                            save_tensor_images(imgs, inv_result_dir + 'original.jpg')
                            for inv_epoch in range(3):
                                if ATTACK == 'black':
                                    inv_img = Dec(feature).detach()
                                elif ATTACK == 'white':
                                    z = inversion(G, Enc, feature, inv_result_dir, save=True)
                                    # z = amor_inversion(G, D, Enc, Amor, feature, z_dim=zd)
                                    inv_img = G(z).detach()
                                save_tensor_images(inv_img, inv_result_dir + f'{inv_epoch}_inv.jpg')
                                for num in range(bs):
                                    save_tensor_images(inv_img[num],
                                                       inv_result_dir + f'{inv_epoch}_' + img_name_list[batch_id * bs + num])

                white_acc, white_ssim, white_fsim = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
                if EVAL_ID:
                    print("Testing evaluation accuracy")
                    _, eval_dataloader = init_dataloader(attributePath, inv_result_dir, action='eval', batch_size=bs, attriID=1, skiprows=1)

                    _, original_dataloader = init_dataloader(attributePath, "../dataset/private", action='eval_fsim', batch_size=bs, n_classes=38,
                                                         attriID=1, skiprows=1) # ../result/prt_inv/noprt_inv_new
                    org_images_list = []
                    for org_images, _ in original_dataloader:
                        org_images = org_images.cuda()
                        org_images_list.append(org_images)
                         ## only for the first batch, edit later

                    acclist, ssim_list, fsim_list = [], [], []
                    with torch.no_grad():
                        for (i,(imgs, label)) in enumerate(eval_dataloader):
                            imgs = imgs.cuda()
                            label = label.cuda()
                            acclist.append(eval_acc(Eval_net, imgs, label))
                            org_images = org_images_list[i]
                            fsim_list.append(feature_sim(Eval_net, org_images, imgs))
                            ssim_list.append(ssim(imgs, org_images))

                        attack_acc, attack_feature_sim, attack_ssim = torch.mean(torch.tensor(acclist)), torch.mean(torch.tensor(fsim_list)), torch.mean(torch.tensor(ssim_list))
                        print(f'eval acc is {torch.mean(torch.tensor(acclist))}')
                        print(f'feature ssim is {torch.mean(torch.tensor(fsim_list))}')
                        print(f'image ssim is {attack_ssim}')
                        if ATTACK == 'white':
                            white_acc = attack_acc
                            white_fsim = attack_feature_sim
                            white_ssim = attack_ssim
                        else:
                            black_acc = attack_acc
                            black_fsim = attack_feature_sim
                            black_ssim = attack_ssim

                if UTIL:
                    print("Testing utility")
                    CF = F()
                    CF.model.fc = torch.nn.Linear(in_features=512, out_features=40)
                    CF.model.load_state_dict(torch.load('../params/enc.pt'))
                    CF = torch.nn.DataParallel(CF).cuda()
                    CF.eval()

                    attributePath = "../dataset/eval_test_attri.csv"
                    _, util_dataloader = init_dataloader(attributePath, figurePath, action=f'utility_original', batch_size=bs,
                                                         n_classes=2, allAttri=True, skiprows=1)

                    predlist, labellist = [], []
                    for (batch_id, (imgs, label)) in enumerate(util_dataloader):
                        torch.cuda.empty_cache()
                        imgs = imgs.cuda()
                        label = label.cuda()

                        feature = torch.load(
                            f'Crafter_result/i{ift_i}_alpha{alpha}_flr{flr}/beta{beta}/features/0_447.pt')

                        pred = CF(feature)
                        predlist.append(pred)
                        labellist.append(label)
                        break
                    pred = torch.vstack(predlist)
                    label = torch.vstack(labellist)
                    auc = my_auc(label, pred)
                    # print("auc list:", auc)
                    print("avg:", mean(auc))
                    utility = mean(auc)

            csv_writer.writerow([str(flr), str(beta), str(white_acc.data.cpu().numpy()), str(black_acc.data.cpu().numpy()), str(white_fsim.data.cpu().numpy()), str(black_fsim.data.cpu().numpy()), str(white_ssim.data.cpu().numpy()),  str(black_ssim.data.cpu().numpy()), str(utility)])
    f.close()

