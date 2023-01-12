import torch
import torch.nn as nn
import mixer

ce = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AC(nn.Module):
    def __init__(self, din, nk, nact):
        super().__init__()

        self.kemb = nn.Embedding(nk, din)

        self.m = nn.Sequential(nn.Linear(din * 3, 256),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(),
                               nn.Linear(256, 256),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(),
                               nn.Linear(256, nact))

    def forward(self, st, stk, k, a):
        ke = self.kemb(k)

        h = torch.cat([st, stk, ke], dim=1)

        h = self.m(h)

        loss = ((h - a) ** 2).mean(dim=-1).sum()

        return loss


class LatentForward(nn.Module):
    def __init__(self, dim, nact):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + nact, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, dim))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=1))

    def loss(self, z, zn, a):
        zn = zn.detach()

        zpred = self.net(torch.cat([z.detach(), a.detach()], dim=1))

        loss = ((zn - zpred) ** 2).mean(dim=-1).sum()

        return loss, zpred


class Encoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.mixer = mixer.MLP_Mixer(n_layers=2, n_channel=20, n_hidden=32, n_output=256, image_size=100, patch_size=10,
                                     n_image_channel=3)

        self.m = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, 256),
                               nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, 100, 100))

        p1 = torch.arange(0, 1, 0.01).reshape((1, 1, 100, 1)).repeat((x.shape[0], 1, 1, 100)).to(x.device)
        p2 = torch.arange(0, 1, 0.01).reshape((1, 1, 1, 100)).repeat((x.shape[0], 1, 100, 1)).to(x.device)
        x = torch.cat([x, p1, p2], dim=1)
        h = self.mixer(x)

        # p1 = (x.round()==1).nonzero(as_tuple=True)[2].unsqueeze(1).float().round() / 100.0
        # p2 = (x.round()==1).nonzero(as_tuple=True)[3].unsqueeze(1).float().round() / 100.0

        # print(p1[0])
        # print(p2[0])

        # h = torch.cat([p1,p2], dim=1)

        return self.m(h)


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.enc = nn.Sequential(nn.Linear(din, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512, 512),
                                 nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512, dout))

    def forward(self, s, gt):
        # print('input', s[:10])
        # print('target', gt[:10])
        # print('input-target diff', (torch.abs(s - gt)).sum())

        # print('in probe')
        # print(s[0])
        # print(gt[0])

        sd = s.detach()

        sd = self.enc(sd)

        loss = ((sd - gt) ** 2).sum()

        abs_loss = torch.abs(sd - gt).mean()

        return loss, abs_loss
