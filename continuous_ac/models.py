import torch
import torch.nn as nn
import mixer
from pe import positionalencoding2d
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from noise import noised
import numpy as np

from vit import ViT
import random

# from batchrenorm import BatchRenorm1d

ce = nn.CrossEntropyLoss()


class GaussianFourierProjectionTime(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResMLP(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_hidden)
        self.mlp3 = nn.Linear(n_hidden, n_hidden)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_hidden)

    def forward(self, U):
        z = self.layer_norm(U)  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))  # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)  # z: [B, n_tokens, n_channel]
        Y = U + z  # Y: [B, n_tokens, n_channel]
        return Y


class AC(nn.Module):
    def __init__(self, din, nk, nact):
        super().__init__()

        self.kemb = nn.Embedding(nk, din)
        self.cemb = nn.Embedding(150, din)

        # self.m = nn.Sequential(nn.Linear(din*3, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.m = nn.Sequential(nn.Linear(din * 3, 256), ResMLP(256), ResMLP(256))

        # self.m_cond = nn.Sequential(nn.Linear(din*4, 256), ResMLP(256), ResMLP(256))

        self.acont_pred = nn.Linear(256, nact)
        self.adisc_pred = nn.Linear(256, 150)

        self.yl = nn.Sequential(nn.Linear(2, din), nn.Tanh(), nn.Linear(din, din))
        self.tl = GaussianFourierProjectionTime(din // 2)

    def forward(self, st, stk, k, a):
        loss = 0.0

        a_cont = a[:, :2]
        a_disc = a[:, 2].long()

        # noise_t = torch.rand_like(a[:,0])
        # noised_a,_,_ = noised(a, noise_t)
        # yemb = self.yl(noised_a)
        # temb = self.tl(noise_t)

        ke = self.kemb(k)

        h = torch.cat([st, stk, ke], dim=1)

        h = self.m(h)

        acont_p = self.acont_pred(h)
        adisc_p = self.adisc_pred(h)

        loss += ((acont_p - a_cont) ** 2).sum(dim=-1).mean() * 10.0

        loss += 0.01 * ce(adisc_p, a_disc)

        return loss


class LatentForward(nn.Module):
    def __init__(self, dim, nact):
        super().__init__()
        self.alin = nn.Sequential(nn.Linear(nact, dim))
        self.net = nn.Sequential(nn.Linear(dim + nact, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(),
                                 nn.Linear(512, dim))
        # self.net = nn.Sequential(ResMLP(512), ResMLP(512), nn.Linear(512, dim))

        if True:
            self.prior = nn.Sequential(nn.Linear(dim + nact, 512), nn.LayerNorm(512), nn.LeakyReLU(),
                                       nn.Linear(512, 512))
            self.posterior = nn.Sequential(nn.Linear(dim * 2 + nact, 512), nn.LayerNorm(512), nn.LeakyReLU(),
                                           nn.Linear(512, 512))
            self.decoder = nn.Sequential(nn.Linear(dim + nact + 256, 512), nn.LeakyReLU(), nn.Linear(512, 512),
                                         nn.LeakyReLU(), nn.Linear(512, dim))

    def forward(self, z, a):
        a = a[:, :2]
        zc = torch.cat([z.detach(), a], dim=1)

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(zc), 2, dim=1)
            std_prior = torch.exp(std_prior) * 0.001 + 1e-5
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            sample = prior.rsample()
            zpred = self.decoder(torch.cat([zc, sample], dim=1))
        else:
            zpred = self.net(zc)

        return zpred

    def loss(self, z, zn, a):
        a = a[:, :2]

        # zn = zn.detach()

        zc = torch.cat([z, a], dim=1)
        zpred = self.net(zc)

        loss = 0.0

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(zc.detach()), 2, dim=1)
            mu_posterior, std_posterior = torch.chunk(self.posterior(torch.cat([zc.detach(), zn.detach()], dim=1)), 2,
                                                      dim=1)
            std_prior = torch.exp(std_prior)
            std_posterior = torch.exp(std_posterior)
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            posterior = torch.distributions.normal.Normal(mu_posterior, std_posterior)
            kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()
            sample = posterior.rsample()
            zpred = self.decoder(torch.cat([zc.detach(), sample], dim=1))
            zn = zn.detach()
            loss += kl_loss * 0.01
        else:
            zpred = self.forward(z, a)

        loss += ((zn - zpred) ** 2).sum(dim=-1).mean() * 0.1

        return loss, zpred


class Encoder(nn.Module):
    def __init__(self, in_channels, encoding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.conv4 = nn.Conv2d(8, 8, 5)
        self.encoding = nn.Sequential(nn.Linear(531648, 256), nn.LeakyReLU(), nn.Linear(256, encoding_dim))
        # self.mixer = mixer.MLP_Mixer(n_layers=2, n_channel=32, n_hidden=32, n_output=32*4*4, image_size=100, patch_size=10, n_image_channel=1)

        # self.m = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))
        # self.m = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, dout))

        # self.bn1 = nn.BatchNorm1d(512)
        # #self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.GroupNorm(4,32)

        # self.m = nn.Sequential(nn.Linear(32*4*4*2, 256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, x, do_bn=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # x = x.reshape((x.shape[0], 1, 100, 100))

        # p1 = torch.arange(0,1,0.01).reshape((1, 1, 100, 1)).repeat((x.shape[0], 1, 1, 100)).cuda()
        # p2 = torch.arange(0,1,0.01).reshape((1, 1, 1, 100)).repeat((x.shape[0], 1, 100, 1)).cuda()
        # x = torch.cat([x,p1,p2],dim=1)

        # h = self.mixer(x)
        #
        # h1 = self.bn1(h)
        # h = h.reshape((h.shape[0], 32, 4, 4))
        # h2 = self.bn2(h)
        #
        # h1 = h1.reshape((h.shape[0], -1))
        # h2 = h2.reshape((h.shape[0], -1))
        #
        # h = torch.cat([h1*0.0,h2],dim=1)
        #
        # return self.m(h)

        return self.encoding(x.flatten(1))

    # def to(self, device):
    #     self.m = self.m.to(device)
    #     self.mixer = self.mixer.to(device)
    #     self.bn1 = self.bn1.to(device)
    #     self.bn2 = self.bn2.to(device)
    #     return self


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.din = din
        self.dout = dout
        self.enc = nn.Sequential(nn.Linear(din, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, dout))

        # self.enc = nn.Sequential(ResMLP(256), nn.Linear(256, dout))
        # self.enc = nn.Linear(din, dout)

    def forward(self, s):
        return s
        # return self.enc(s)

    def loss(self, s, gt):
        # print('input', s[:10])
        # print('target', gt[:10])
        # print('input-target diff', (torch.abs(s - gt)).sum())

        # print('in probe')
        # print(s[0])

        # sd = s.detach()
        sd = s

        sd = self.forward(sd)

        loss = ((sd - gt) ** 2).sum(dim=-1).mean()

        abs_loss = torch.abs(sd - gt).mean()

        return loss, abs_loss
