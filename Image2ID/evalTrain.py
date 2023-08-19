import sys
sys.path.append("..")
import metrics
import torch.nn as nn
import torch
import os
from torch.autograd import grad
import torch.optim as optim
from WhiteBoxAttack.attacker import inversion, amor_inversion
from models import *
import random
import time
import loader

import torchvision
import evaluate


def compute_accuracy(model, data_loader):
    acc_list = []
    for (i, (img, label)) in enumerate(data_loader):
        img = img.cuda()
        label = label.cuda()
        probs = model(img)
        # iden = torch.max(probs, 1)
        iden = torch.argmax(probs, dim = 1)# +torch.ones(img.shape[0]).cuda()
        # print(iden.shape)
        # print("label is {}",label)
        # print("pred is {}", iden)
        acc = label.eq(iden.long()).sum() / bs
        # print(acc)
        acc_list.append(acc)
    return torch.Tensor(acc_list).mean()


if __name__ == "__main__":
    device = "cuda"
    torch.cuda.current_device()
    torch.cuda._initialized = True

    # e_path = "../params/eval_param.pkl"
    Eval = evaluate.Encoder(1000)
    Eval = nn.DataParallel(Eval).cuda()

    bs, z_dim = 64, 100

    attributePath = "../dataset/eval_train.txt"
    figurePath = "../dataset/private"
    test_attributePath = "../dataset/eval_test.txt"
    test_figurePath = "../dataset/private"
    _, train_dataloader = loader.init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=1000, attriID=1,
                                                 skiprows=1)

    _, test_dataloader = loader.init_dataloader(test_attributePath, test_figurePath, batch_size=bs, n_classes=1000, attriID=1,
                                                 skiprows=1)
    lr = 0.001
    epochs = 120
    criterion = torch.nn.CrossEntropyLoss()
    print("---------------------Training------------------------------")

    e_optimizer = torch.optim.Adam(Eval.parameters(), lr=lr, weight_decay=1e-4)

    save_dir = './eval_param/celebA64/'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        start = time.time()
        loss_ave = 0
        Eval.train()
        for (i, (img, label)) in enumerate(train_dataloader):
            img = img.cuda()
            label = label.long().cuda()
            out= Eval(img)
            # print("Shape of Eval output is ", out.shape) [bs,1000]
            loss = criterion(out, label)
            loss_ave += loss
            e_optimizer.zero_grad()
            loss.backward()
            e_optimizer.step()
        end = time.time()
        interval = end - start
        loss_ave = loss_ave / (i + 1)
        print("Epoch {}, loss {}".format(epoch, loss))
        if (epoch+1)%10 == 0:
            torch.save(Eval.state_dict(), save_dir+"{}.pkl".format(epoch))

            print("----------------------Testing------------------------------")
            # compute_accuracy(Eval, train_dataloader)
            Eval.eval()
            test_acc = compute_accuracy(Eval, test_dataloader)
            print(test_acc)
