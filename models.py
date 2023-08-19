import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision

class GanGenerator(nn.Module): #in_dim=100
    def __init__(self, in_dim=100, dim=64):
        super(GanGenerator, self).__init__()

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
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class newGanGenerator(nn.Module): # in_dim=150
    def __init__(self, in_dim=150, dim=64):
        super(newGanGenerator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 4 * 8 * 8, bias=False),
            nn.BatchNorm1d(dim * 4*8*8),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 8, 8)
        y = self.l2_5(y)
        return y


class GanDiscriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(GanDiscriminator, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)
        return x


class Classifier(nn.Module):
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


class Eval(nn.Module):
    def __init__(self, n_classes):
        super(Eval, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.resnet152(pretrained=True)
        self.fc_layer_1 = nn.Linear(2048 * 2 * 2, 300)
        self.fc_layer_2 = nn.Linear(300, 200)
        self.fc_layer_3 = nn.Linear(200, self.n_classes)

    def forward(self, x, toEnd=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        if toEnd:
            x = self.fc_layer_1(x)
            x = self.fc_layer_2(x)
            x = self.fc_layer_3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_dim=64 * 16 * 16, dim=64):
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
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())
    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.l1(y)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        # Since there is no effective implementation of LayerNorm,
        # we use InstanceNorm2d instead of LayerNorm here.
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))


class Amortizer(nn.Module):
    def __init__(self, nz=500):
        super().__init__()
        self.main = nn.Sequential(
            # Image (Cx32x32)
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(1024, 512, 2, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, nz, 1)
        )

    def forward(self, x):
        out = self.main(x)
        # print(out.shape)
        out2 = self.output(out)
        out2 = out2.view(len(x), -1)
        return out2

