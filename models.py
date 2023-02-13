import torch
import torch.nn as nn

from transformer import Transformer

ce = nn.CrossEntropyLoss()

class AC(nn.Module):
    def __init__(self, din, n1, n2, n3,  nact, m):
        super().__init__()

        self.m = m

        self.emb1 = nn.Embedding(n1, din)
        #self.emb2 = nn.Embedding(n2, din)


        self.emb3 = nn.Embedding(n3, din)

        

        self.action_embedding = nn.Embedding(nact, din)


        self.factors = nn.Parameter(torch.randn(1, 1, din))

        self.pos_emb = nn.Parameter(torch.randn(1, 200, din))

        self.obs_factors = nn.Parameter(torch.randn(1, m ** 2, din))

        self.transformer_enc = Transformer(din, 3, 6, 2, din, 2)
        self.transformer_dec = Transformer(din, 3, 6, 2, din, 2)
        self.decode_projection = nn.Linear(din, 3)

        self.rule_1 = nn.Sequential(nn.Linear(3 * din, 2 * din), nn.GELU(), nn.LayerNorm(2 * din), nn.Linear(2 * din, din))
        self.rule_2 = nn.Sequential(nn.Linear(din, 2 * din), nn.GELU(), nn.LayerNorm(2 * din), nn.Linear(2 * din, din))

        self.m_ = nn.Sequential(nn.Linear(1280, 256),  nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, nact))
        self.prior = nn.Sequential(nn.Linear(din, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,2 * din))
        self.posterior = nn.Sequential(nn.Linear(din*2, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,2 * din))
        #self.decoder = nn.Sequential(nn.Linear(dim + nact + 256, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512,dim))
        self.mse_loss = nn.BCELoss()

    def encode(self, x):
        B, H, W, C = x.shape
        #x = torch.split(x, 1, dim = -1)
        x_enc1 = self.emb1(x)
        #x_enc2 = self.emb2(x[1])

        x_enc = x_enc1#torch.cat((x_enc1, x_enc2), dim = -1)
        x_enc = x_enc.reshape(B, -1, x_enc.shape[-1])

        pos_emb = self.pos_emb.repeat(B, 1, 1)

        factors = self.factors.repeat(B, 1, 1)

        x = torch.cat((x_enc, factors), dim = 1)
        x = x + pos_emb[:, :x.shape[1]]
        x = self.transformer_enc(x)

        return x[:, -2:]

    
    def decode(self, x):

        obs_factors = self.obs_factors.repeat(x.shape[0], 1, 1)
        
        x = torch.cat((obs_factors, x), dim = 1)

        x = self.transformer_dec(x)

        x = self.decode_projection(x[:, :-2])

        x = x.reshape(x.shape[0], self.m , self.m, -1)

        return x
    
    def transition(self, x, a, x_next, train  = False):
        a_emb = self.action_embedding(a)
        x_split = torch.split(x, 1, dim = 1)
        x_next_split = torch.split(x_next, 1, dim = 1)

        kl_loss = 0
        if train:
            mu_prior, std_prior = torch.chunk(self.prior(x_split[1].squeeze(1)), 2, dim=1)
            mu_posterior, std_posterior = torch.chunk(self.posterior(torch.cat([x_split[1].squeeze(1), x_next_split[1].squeeze(1)], dim=1)), 2, dim=1)
            std_prior = torch.exp(std_prior)
            std_posterior = torch.exp(std_posterior)
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            posterior = torch.distributions.normal.Normal(mu_posterior, std_posterior)
            kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()
            x_transition_2 = posterior.rsample().unsqueeze(1)
        else:
            mu_prior, std_prior = torch.chunk(self.prior(x_split[1].squeeze(1)), 2, dim=1)
            std_prior = torch.exp(std_prior) * 0.001 + 1e-5
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            x_transition_2   = prior.rsample().unsqueeze(1)

        x_transition_1 = self.rule_1(torch.cat((x_split[0], a_emb.unsqueeze(1), x_split[1]), dim = -1))

        x_out = torch.cat((x_transition_1, x_transition_2), dim = 1)

        return x_out, kl_loss

    def forward(self, st, st1, stk, k, a, train = False): 

        st_enc = self.encode(st)
        stk_enc = self.encode(stk)
        st1_enc = self.encode(st1)
        st1_tr, kl_loss = self.transition(st_enc, a.clone(), st1_enc, train = train)

        #st1_dec = self.decode(st1_tr)

        
        #print(st1_dec.shape)
        #print(st1.shape)

       


        

        rec_loss = ((st1_tr - st1_enc) ** 2).mean()#ce(st1_dec.permute(0, 3, 1, 2), st1[:, :, :, 0]) #+ ce(st1_dec_2.permute(0, 3, 1, 2), st1[:, :,:,  1])




        
        ke = self.emb3(k)

        B = st_enc.shape[0]

        

        h = torch.cat([st_enc.reshape(B, -1), stk_enc.reshape(B, -1), ke], dim=1)

        h = self.m_(h)
        

        loss = ce(h, a)

        
        

        return loss, rec_loss, kl_loss, st_enc[:, 0], st_enc[:, 1]

class Encoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(din, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, x):
        return self.m(x)


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(din, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))

    def forward(self, s, gt):

        sd = s.detach()

        sd = self.m(sd)

        loss = ce(sd, gt)


        acc = torch.eq(sd.argmax(1), gt).float().mean()

        return loss, acc




