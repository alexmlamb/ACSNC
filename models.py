import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

class AC(nn.Module):
    def __init__(self, din, nk, nact):
        super().__init__()

        self.kemb = nn.Embedding(nk, din)

        self.m = nn.Sequential(nn.Linear(din*3, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, nact))

    def forward(self, st, stk, k, a): 

        ke = self.kemb(k)

        h = torch.cat([st, stk, ke], dim=1)

        h = self.m(h)


        loss = ce(h, a)

        return loss

class Encoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(din, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, x):
        return self.m(x)


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(din, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, s, gt):

        sd = s.detach()

        sd = self.m(sd)

        loss = ce(sd, gt)


        acc = torch.eq(sd.argmax(1), gt).float().mean()

        return loss, acc




