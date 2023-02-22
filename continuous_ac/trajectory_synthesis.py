
'''

-Learn p(s[t : t + k] | s[t], s[t+k]).  
-We can then do an inverse-model on the trajectory as our plan.  

(N,2), (N,2) --> (N,k).  


'''

import torch
import torch.nn as nn
import random
import numpy as np

def sample_ex(S,k):

    t = random.randint(0, S.shape[0]-20)

    return S[t:t+k]

def sample_trajectory_batch(S, bs, k):
    st = []

    for b in range(bs):
        s = sample_ex(S,k)
        st.append(s)

    st = torch.Tensor(np.array(st)).cuda()

    return st


class TSynth(nn.Module):
    def __init__(self, dim, k):
        super().__init__()

        if True:
            self.prior = nn.Sequential(nn.Linear(dim*2, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,512))
            self.posterior = nn.Sequential(nn.Linear(dim*2 + dim*k, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,512))
            self.decoder = nn.Sequential(nn.Linear(dim*2 + 256, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512,dim*k))

    def make_zc(self, st, stk):
        st = st.reshape((st.shape[0], -1))
        stk = stk.reshape((stk.shape[0], -1))

        zc = torch.cat([st,stk],dim=1)

        return zc

    def forward(self, st, stk):

        zc = self.make_zc(st,stk)

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(zc), 2, dim=1)
            std_prior = torch.exp(std_prior) * 0.001 + 1e-5
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            sample = prior.rsample()
            zpred = self.decoder(torch.cat([zc, sample], dim=1))

        return zpred

    def loss(self, st, stk, straj):

        straj = straj.reshape((straj.shape[0],-1))

        loss = 0.0

        zc = self.make_zc(st,stk)

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(zc.detach()), 2, dim=1)
            mu_posterior, std_posterior = torch.chunk(self.posterior(torch.cat([zc.detach(), straj.detach()], dim=1)), 2, dim=1)
            std_prior = torch.exp(std_prior)
            std_posterior = torch.exp(std_posterior)
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            posterior = torch.distributions.normal.Normal(mu_posterior, std_posterior)
            kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()
            sample = posterior.rsample()
            zpred = self.decoder(torch.cat([zc.detach(), sample], dim=1))
            straj = straj.detach()
            loss += kl_loss * 0.01

        loss += ((straj - zpred)**2).sum(dim=-1).mean() * 0.1


        return loss, zpred




