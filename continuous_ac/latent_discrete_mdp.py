import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Function


class BinarizeSigF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0.5] = 1
        output[input < 0.5] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.hardsigmoid(input)
        output = binarizeSig(output)
        return output


class LatentDiscreteMDP(nn.Module):
    def __init__(self, latent_dim, discrete_dim, action_dim, num_embeddings, embedding_dim):
        super(LatentDiscreteMDP, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.Tanh(),
                                     nn.Linear(256, discrete_dim),
                                     BinarySigmoid())

        self.decoder = nn.Sequential(nn.Linear(discrete_dim, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, latent_dim))

        self.k_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.inverse_transition = nn.Sequential(nn.Linear(2 * discrete_dim + embedding_dim, 256),
                                                nn.LeakyReLU(),
                                                nn.Linear(256, action_dim))

        self.optimizer = Adam([{'params': self.encoder.parameters()},
                               {'params': self.decoder.parameters()},
                               {'params': self.inverse_transition.parameters()}],
                              lr=1e-3)

    def update(self, latent_state, k_latent_state, k, action):
        discrete_latent = self.encoder(latent_state)
        discrete_k_latent = self.encoder(k_latent_state)

        predicted_latent_state = self.decoder(discrete_latent)
        predicted_k_latent_state = self.decoder(discrete_k_latent)

        k_embedding = self.k_embedding(k)
        pred_action = self.inverse_transition(torch.cat((discrete_latent,
                                                         discrete_k_latent,
                                                         k_embedding), dim=1))
        loss = nn.MSELoss()(predicted_latent_state, latent_state)
        loss += nn.MSELoss()(predicted_k_latent_state, k_latent_state)
        loss += nn.MSELoss()(pred_action, action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


binarizeSig = BinarizeSigF.apply
