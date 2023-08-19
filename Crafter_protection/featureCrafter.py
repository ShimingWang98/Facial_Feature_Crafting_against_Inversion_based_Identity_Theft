import sys
sys.path.append('..')
from WhiteBoxAttack.attacker import *
from loader import *
from models import *
import os
import torch
from tqdm import tqdm
from typing import Tuple


def f_loss(f, f_rec):
    n, d = len(f), 64 * 16 * 16
    mse = torch.norm(f.view(n, -1) - f_rec.view(n, -1), dim=1) ** 2
    mse = mse/d
    return torch.mean(mse)


def solve_z(feature, G, Enc):
    """
    Find initial x* of feature
    """
    G.zero_grad()
    Enc.zero_grad()

    print("with amortize")
    z = amor_inversion(G, D, Enc, Amor, feature, z_dim=zd)
    z = z.clone().detach().requires_grad_(True)
    G.zero_grad()
    Enc.zero_grad()
    return z


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha_grad = torch.rand(shape).cuda()
    interpolate = x + alpha_grad * (y - x)
    interpolate = interpolate.cuda()
    interpolate.requires_grad = True

    o = netD(interpolate)
    g = grad(o, interpolate, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(interpolate.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def gradient_based_ho(f0, netD, batch_id):
    f_adv = f0.clone().detach().requires_grad_(True).cuda()
    f_optimizer = torch.optim.Adam([f_adv], lr=flr)
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=5e-4)
    p_list, u_list = [], []
    z_avg = torch.randn(bs, zd).cuda().float()
    for epoch in (range(epochs)):  # Outer optimization loop  tqdm
        print(f'\nepoch {epoch}')
        z = solve_z(f_adv, G, Enc)

        d_batch = 32
        for d_epoch in range(10):  # update Discriminator
            real_index = torch.randint(0, bs, (d_batch,))
            real = G(z_avg[real_index]).clone().detach()
            fake_index = torch.randint(0, bs, (d_batch,))
            fake = G(z[fake_index]).clone().detach()
            neg_logit = netD(fake)
            pos_logit = netD(real)

            gp = gradient_penalty(real, fake)
            l_gp = lambda_gp
            if gp > 1:
                l_gp = 40 # 15
            EMD = neg_logit.mean() - pos_logit.mean()
            d_loss = EMD + l_gp * gp
            p_list.append(torch.mean(EMD).data.cpu())
            print(" gp {}, d loss {} ".format(gp.item(), d_loss.item()))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            del fake
            del real

        inv_loss = f_loss(f_adv, Enc(G(z)))*2000
        l_p = -netD(G(z)).mean()
        l_u = f_loss(f_adv, f0)# /(64*16*16)
        tot_loss = torch.mean(100*l_p + 100*beta * l_u) # 100*l_p + beta * l_u
        print(f'privacy loss is {l_p}, utility loss is {l_u}')
        hyper_grads = hypergradient(tot_loss, inv_loss, f_adv, z)
        f_optimizer.zero_grad()
        f_adv.grad = hyper_grads[0]
        print(f'check feature grad is {torch.norm(f_adv.grad)}')
        f_optimizer.step()
        u_list.append(l_u.data.cpu())
        if epoch % 3 == 0:
            save_tensor_images(G(z).detach(), f'{result_dir}/images/inv_{epoch}.jpg')
            torch.save(f_adv.detach(), f'{result_dir}/features/{batch_id}_{epoch}.pt')
    plot_loss(p_list, result_dir + '/curves', 'privacy_{}'.format(batch_id), 'EM distance')
    plot_loss(u_list, result_dir + '/curves', 'utility_{}'.format(batch_id), 'feature norm')
    return


def hypergradient(tot_loss: torch.Tensor, inv_loss: torch.Tensor, f_adv: torch.Tensor, z: torch.Tensor):
    v1 = torch.autograd.grad(tot_loss, z, retain_graph=True)
    for V in v1:
        print(f'EMD to z grad is {torch.norm(V)}')
    d_inv_d_z = torch.autograd.grad(inv_loss, z, create_graph=True)
    v2 = approxInverseHVP(v1, d_inv_d_z, z, i=i, alpha=alpha)
    v3 = torch.autograd.grad(d_inv_d_z, f_adv, grad_outputs=v2, retain_graph=True) # inv_z 对 f_adv的partial太小了
    # for V in v3:
    #     print(f'v shape is {V.shape}')
    d_tot_d_f = torch.autograd.grad(tot_loss, f_adv)
    # print(f'd_inv_d_z grad is {torch.norm(d_inv_d_z)}, v2 is {torch.norm(v2)}, v3 is {torch.norm(v3)}')
    return [d - v for d, v in zip(d_tot_d_f, v3)]


def approxInverseHVP(v: torch.Tensor, f: torch.Tensor, z: torch.Tensor, i=30, alpha=.01):
    p = v
    for j in range(i):
        grad = torch.autograd.grad(f, z, grad_outputs=v, retain_graph=True)
        if j%30 == 0:
            print(f'inner epoch {j}, grad is {torch.norm(grad[0])}, v is {torch.norm(v[0])}')
        v = [v_ - alpha * g for v_, g in zip(v, grad)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]
    p = [alpha*p_ for p_ in p]
    return p


if __name__ == "__main__":
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True
    figurePath = "../dataset/private"
    attributePath = "../dataset/eval_test.txt" #_batch1
    bs = 128  # 128
    epochs = 450# 600 #600
    zd = 500
    lambda_gp = 20
    beta_list = [10]  #0.5, 1, 2, 10
    flr, alpha = .01, .001
    i_list = [120] # ,100

    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)

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

    for beta in beta_list:
        for i in i_list:
            result_dir = f'../Crafter_result/beta{beta}/'
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(result_dir + '/original', exist_ok=True)
            os.makedirs(result_dir + '/images', exist_ok=True)
            os.makedirs(result_dir + '/features', exist_ok=True)
            os.makedirs(result_dir + '/curves', exist_ok=True)
            for (batch_id, (imgs, label)) in enumerate(dataloader):
                torch.cuda.empty_cache()
                imgs = imgs.cuda()
                save_tensor_images(imgs.detach(), result_dir+f'/original/{batch_id}.jpg')
                f0 = Enc(imgs).detach()
                netD = GanDiscriminator()
                netD = torch.nn.DataParallel(netD).cuda()
                gradient_based_ho(f0, netD, batch_id)
