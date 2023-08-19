import sys
sys.path.append('..')
from models import *
import torch.nn as nn
import torch.optim as optim
import torch
from loader import *
import os
n_epochs = 400
bs = 128

attributePath = "../dataset/eval_train.csv"
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
f2i.load_state_dict(torch.load(os.path.join(result_dir, f"f2i_e85.pkl")))
f2i = torch.nn.DataParallel(f2i).cuda()
f2i.train()

f2i_opt = optim.Adam(f2i.parameters())

loss_fn = torch.nn.CrossEntropyLoss()

for e in range(86,n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.cuda()
        labels = labels.type(torch.int64).cuda()
        with torch.no_grad():
            feature = Enc(imgs)
        pred = f2i(feature)
        f2i_opt.zero_grad()
        loss = loss_fn(pred,labels)
        loss.backward()
        f2i_opt.step()
        print(loss.item())
    if e%5 == 0:
        torch.save(f2i.module.state_dict(), os.path.join(result_dir, f"f2i_e{e}.pkl"))
        
