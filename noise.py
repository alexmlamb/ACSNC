
import torch

'''
t from 0 to 1, no noise to lots of noise.  
'''
def noised(x, t):
    beta_1 = 20.0
    beta_0 = 0.0
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff[(..., ) + (None,)*(len(x.shape)-1)]) * x # complex way of expanding log_mean_coef to same shape as x (256) -> (256, 1, 1, 1) for image or (256, 1) for label
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))


    sample = mean + std[:,None] * torch.randn_like(std[:,None])


    return sample, mean, std


if __name__ == "__main__":
    x = torch.zeros((10,5)).float()

    x[:,3] += 0.1

    print(x)

    for t in [0.01]:
        tl = torch.Tensor([t]).repeat(x.shape[0])
        print(t)
        s,mu,std = noised(x,tl)

        #print(s.shape, mu.shape, std.shape, tl.shape, x.shape)
        print('std', std.mean())

        print(s)

