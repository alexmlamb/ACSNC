
import torch
from torch import nn

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes_1, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()


        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.head = nn.Sequential(
            nn.Linear(dim, num_classes_1)
        )


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.head(x)

        return x

if __name__ == "__main__":

    import mixer
    from batchrenorm import BatchRenorm1d

    #vit = ViT(image_size=100, patch_size=10, num_classes_1=2, dim=256, depth=6, heads=4, mlp_dim=512).cuda()
    mymixer = mixer.MLP_Mixer(n_layers=2, n_channel=32, n_hidden=32, n_output=32*4*4, image_size=100, patch_size=10, n_image_channel=1).cuda()

    post_net1 = nn.BatchNorm2d(32).cuda()
    post_net2 = nn.LayerNorm(32).cuda()

    post_net3 = nn.Sequential(nn.Linear(32*4*4, 256), nn.LeakyReLU(), nn.Linear(256,2)).cuda()

    opt = torch.optim.Adam(list(mymixer.parameters()) + list(post_net1.parameters()) + list(post_net2.parameters()) + list(post_net3.parameters()))

    from main import sample_batch
    from room_env import RoomEnv
    env = RoomEnv()

    X = []
    A = []
    ast = []
    est = []

    import random
    for i in range(0,500000):
        a = env.random_action()

        x, agent_state, exo_state = env.get_obs()
        env.step(a)

        A.append(a[:])
        X.append(x[:])
        ast.append(agent_state[:])
        est.append(exo_state[:])

    X = np.asarray(X).astype('float32')
    A = np.asarray(A).astype('float32')
    ast = np.array(ast).astype('float32')
    est = np.array(est).astype('float32')


    for j in range(0,10000):

        xt, xtn, xtk, k, a, astate, estate = sample_batch(X, A, ast, est, 128)
        
        xt = xt.reshape((128, 1, 100, 100))
        pred = mymixer(xt)

        pred = pred.reshape((128,32,4,4))

        if j % 100 == 0:
            post_net1.eval()
        else:
            post_net1.train()

        if True:
            pred = post_net1(pred)
        else:
            pred = post_net2(pred)

        pred = pred.reshape((128,-1))

        pred = post_net3(pred)

        loss = ((pred - astate)**2).sum(dim=-1).mean()
        abs_loss = (torch.abs(pred - astate)).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        if j % 100 == 0:
            print(j, loss.item(), abs_loss.item())

            print(astate[0])
            print(pred[0])





