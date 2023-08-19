import sys
sys.path.append('..')

from models import *
import torch
import os
from torchvision import transforms
from loader import *

device = "cuda"
torch.cuda.current_device()
torch.cuda._initialized = True

# featureSize = 64*16*16# 128*32*32 # 256 * 16 * 16
batchSize = 64
batchNum = 100
latent_dim = 700
in_channel=64
amor = Amortizer(nz=latent_dim) # WARNING: deep amor used
amor = torch.nn.DataParallel(amor).cuda()

Enc = Encoder()
Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
Enc.model.load_state_dict(torch.load('../params/enc.pt'))
Enc = torch.nn.DataParallel(Enc).cuda()
Enc.eval()
for param in Enc.parameters():
    param._requires_grad = False

G = newGanGenerator(in_dim=latent_dim)
G.load_state_dict(torch.load('../Gan/G_zdim_{}/params/G_95.pkl'.format(latent_dim)))
G = torch.nn.DataParallel(G).cuda()
G.eval()
for param in G.parameters():
    param._requires_grad = False

result_dir = '../params/amor/'
test_amor_dir = '../result/amor/'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(test_amor_dir, exist_ok=True)


# loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCEWithLogitsLoss()
def f_loss(f, f_rec):
    n, d = len(f), torch.numel(f[0])
    mse = torch.norm(f.view(n, -1) - f_rec.view(n, -1), dim=1) ** 2  # / d
    return mse


trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
epoch = 500
z_allbatch = torch.randn((batchNum, batchSize, latent_dim)).cuda()
tot_loss = []

opt = torch.optim.Adam(amor.parameters(), lr=0.01)


attributePath = "../dataset/eval_test.txt"
figurePath = "../dataset/private"
bs = 64
_, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=1000, attriID=1, skiprows=1)
for (imgs, label) in dataloader:
    imgs = imgs.cuda()
    break

for e in range(epoch):
    if (e + 1) % 50 == 0:  # regenerate the random batches
        z_allbatch = torch.randn((batchNum, batchSize, latent_dim)).cuda()
    loss_list = []
    for i in range(batchNum):
        # z = torch.randn((batchSize,100)).cuda()
        # import pdb;pdb.set_trace()
        feature = Enc(G(z_allbatch[i])).clone().detach()
        normalized_org_img = trans(G(z_allbatch[i]))
        normalized_inv_img = trans(G(amor(feature)))
        ivtF = Enc(normalized_inv_img)
        normalized_F = Enc(normalized_org_img)
        opt.zero_grad()
        # loss = loss_fn(ivtF, feature)
        # print(ivtF.shape)
        loss = torch.mean(f_loss(ivtF, normalized_F))
        loss.backward()
        # print("grad on regenerated feature is ",ivtF.grad.data[0][0][0][0])
        # print("grad on inverted z is ", ivtZ.grad.data[0][0]) # this one becomes 0 easily
        # for param in amor.parameters():
        #     print("grad on amor",param.grad.data[0][0])
        #     break
        opt.step()
        loss_list.append(loss)
    batch_loss = torch.mean(torch.Tensor(loss_list))
    print("epoch {}, loss is {}".format(e, batch_loss))
    tot_loss.append(batch_loss)
    if e % 10 == 0:
        save_tensor_images(G(amor(Enc(imgs))).detach(), test_amor_dir + 'inv_epoch{}.jpg'.format(e))
        torch.save(amor.module.state_dict(), result_dir + f"amor_zd{latent_dim}_epoch{e}.pkl")
