import torch
import torch.nn as nn
import mixer
from pe import positionalencoding2d
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from noise import noised
import numpy as np

from vit import ViT

#from batchrenorm import BatchRenorm1d

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
        z = self.layer_norm(U)                  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))             # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)                        # z: [B, n_tokens, n_channel]
        Y = U + z                               # Y: [B, n_tokens, n_channel]
        return Y

class AC(nn.Module):
    def __init__(self, din, nk, nact):
        super().__init__()

        self.kemb = nn.Embedding(nk, din)

        self.m = nn.Sequential(nn.Linear(din*5, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, nact))
        #self.m = nn.Sequential(nn.Linear(din*3, 512), ResMLP(512), ResMLP(512), nn.Linear(512, nact))

        self.yl = nn.Linear(2,din)
        self.tl = GaussianFourierProjectionTime(din//2)

    def forward(self, st, stk, k, a): 

        noise_t = torch.rand_like(a[:,0])

        noised_a,_,_ = noised(a, noise_t)

        yemb = self.yl(noised_a)
        temb = self.tl(noise_t)


        ke = self.kemb(k)

        h = torch.cat([st, stk, ke, yemb, temb], dim=1)

        h = self.m(h)


        loss = ((h - a)**2).mean(dim=-1).sum()

        return loss

class LatentForward(nn.Module):
    def __init__(self, dim, nact):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+nact, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, dim))

    def forward(self, z, a):
        return self.net(torch.cat([z.detach(),a],dim=1))

    def loss(self, z, zn, a):
        zn = zn.detach()

        zpred = self.net(torch.cat([z.detach(), a.detach()],dim=1))

        loss = ((zn - zpred)**2).mean(dim=-1).sum()

        return loss, zpred

class Encoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.mixer = mixer.MLP_Mixer(n_layers=2, n_channel=32, n_hidden=32, n_output=256, image_size=100, patch_size=10, n_image_channel=1)

        self.vit = ViT(image_size=100, patch_size=10, num_classes_1=256, dim=256, depth=6, heads=4, mlp_dim=512, channels=1)

        #self.m = nn.Sequential(ResMLP(256), ResMLP(256), nn.Linear(256, dout))
        #self.m = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, x):

        x = x.reshape((x.shape[0], 1, 100, 100))

        #x = TF.gaussian_blur(x,7)*16
        #x = TF.gaussian_blur(x,19)*32

        p1 = torch.arange(0,1,0.01).reshape((1, 1, 100, 1)).repeat((x.shape[0], 1, 1, 100)).cuda()
        p2 = torch.arange(0,1,0.01).reshape((1, 1, 1, 100)).repeat((x.shape[0], 1, 100, 1)).cuda()
        #x = torch.cat([x,p1,p2],dim=1)
        
        h = self.mixer(x)
        #h = self.vit(x)

        #p1 = (x.round()==1).nonzero(as_tuple=True)[2].unsqueeze(1).float().round() / 100.0
        #p2 = (x.round()==1).nonzero(as_tuple=True)[3].unsqueeze(1).float().round() / 100.0

        #print(p1[0])
        #print(p2[0])

        #h = torch.cat([p1,p2], dim=1)

        return h
        #return self.m(h)


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        #self.enc = nn.Sequential(nn.Linear(din, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512,512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512, dout))
        self.enc = nn.Sequential(ResMLP(256), ResMLP(256), nn.Linear(256, dout))

    def forward(self, s):
        return self.enc(s)

    def loss(self, s, gt):

        #print('input', s[:10])
        #print('target', gt[:10])
        #print('input-target diff', (torch.abs(s - gt)).sum())

        #print('in probe')
        #print(s[0])

        sd = s.detach()

        sd = self.enc(sd)

        loss = ((sd - gt)**2).sum(dim=-1).mean()

        abs_loss = torch.abs(sd - gt).mean()

        return loss, abs_loss






