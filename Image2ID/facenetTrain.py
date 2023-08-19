import sys
sys.path.append('..')
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch.optim
from loader import *
from plot_loss import *


def freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
def unfreeze(m):
    for p in m.parameters():
        p.requires_grad_(True)

resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=38).cuda().eval()
freeze(resnet)
unfreeze(resnet.logits)
resnet.logits.train()

opt = torch.optim.Adam(resnet.logits.parameters())

idPath = "../dataset/eval_train.csv"
imgPath = "../dataset/private"

_, trainLoader = init_dataloader(idPath, imgPath, batch_size=256, n_classes=38, skiprows=1)
loss_fn = torch.nn.CrossEntropyLoss()

lossList = []
for epoch in range(10):
    for imgs, label in trainLoader:
        imgs = imgs.cuda()
        pred = resnet(imgs)
        opt.zero_grad()
        l = loss_fn(pred, label.type(torch.LongTensor).cuda())
        lossList.append(l.item())
        l.backward()
        opt.step()
    torch.save(resnet.logits.state_dict(),"../params/facenetFC.pkl")
    plot_loss(lossList, name='facenetEvalTrain', title="trainLoss")
