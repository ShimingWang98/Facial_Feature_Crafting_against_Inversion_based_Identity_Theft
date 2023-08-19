import sys
sys.path.append('..')
from models import *
import torch.nn as nn
import torch.optim as optim
import torch
from loader import *
import os
import sklearn.metrics
n_epochs = 100
bs = 128

attributePath = "../dataset/eval_test.csv"
figurePath = "../dataset/private"

result_dir = "../params/f2i/"
os.makedirs(result_dir, exist_ok=True)



_, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)


Enc = Encoder()
Enc.model.fc = torch.nn.Linear(in_features=512, out_features=40)
Enc.model.load_state_dict(torch.load('../params/enc.pt'))
Enc = torch.nn.DataParallel(Enc).cuda()
Enc.eval()

f2i = F()
f2i.load_state_dict(torch.load(os.path.join(result_dir, f"f2i_e95.pkl")))
f2i = torch.nn.DataParallel(f2i).cuda()
f2i.eval()

for beta in [0.5, 1, 2, 10]:
    print("beta:", beta)
    featurePath = f"../Crafter_result/beta{beta}/features/0_447.pt"
    f2i.module.load_state_dict(torch.load(os.path.join(result_dir, f"f2i_e295.pkl")))
    with torch.no_grad():
        predlist = []
        labellist = []
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.cuda()
            labels = labels.type(torch.int64).cuda()
            # feature = Enc(imgs)
            feature = torch.load(featurePath)
            pred = f2i(feature)
            pred = pred.argmax(axis=1)
            labellist.append(labels)
            predlist.append(pred)
            # if i > 3:
            break
        # import pdb;pdb.set_trace()
        pred = torch.cat(predlist) #vstack for 2-dimension
        label = torch.cat(labellist)
        # print((label==pred).type(torch.float64))
        print((label==pred).type(torch.float64).mean().item())
            
