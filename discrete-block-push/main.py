# from block_pull_env import BlockEnv
from block_push_env import BlockEnv

import torch.nn.functional as F

from sklearn.cluster import KMeans
from models import Encoder, Probe, AC
import torch
import numpy as np

'''
Sample 100k examples.  
Write a batch sampler to get (xt, xtk, k, agent_state, block_state).  

p(a[t] | enc(x[t]), enc(x[t+k]), k)

enc(x[t]).detach() --> agent_state[t]
enc(x[t]).detach() --> block_state[t]

encoder
inv_model
as_probe
bs_probe


'''


def sample_example(X, A, ast, bst, est):
    N = X.shape[0]
    maxk = 40
    t = random.randint(0, N - maxk - 1)
    k = random.randint(1, maxk)

    return (X[t], X[t + k], k, A[t], ast[t], bst[t], est[t])


def sample_batch(X, A, ast, bst, est, bs):
    xt = []
    xtk = []
    klst = []
    astate = []
    bstate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, bst, est)
        xt.append(lst[0])
        xtk.append(lst[1])
        klst.append(lst[2])
        alst.append(lst[3])
        astate.append(lst[4])
        bstate.append(lst[5])
        estate.append(lst[6])

    xt = torch.Tensor(np.array(xt)).cuda()
    xtk = torch.Tensor(np.array(xtk)).cuda()
    klst = torch.Tensor(np.array(klst)).long().cuda()
    alst = torch.Tensor(np.array(alst)).long().cuda()
    astate = torch.Tensor(np.array(astate)).long().cuda()
    bstate = torch.Tensor(np.array(bstate)).long().cuda()
    estate = torch.Tensor(np.array(estate)).long().cuda()

    return xt, xtk, klst, alst, astate, bstate, estate


if __name__ == "__main__":

    env = BlockEnv()

    ac = AC(512, nk=45, nact=5).cuda()
    enc = Encoder(env.m ** 2 * 2, 512).cuda()
    a_probe = Probe(512, env.m ** 2).cuda()
    b_probe = Probe(512, env.m ** 2).cuda()
    e_probe = Probe(512, env.m ** 2).cuda()

    opt = torch.optim.Adam \
        (list(ac.parameters()) + list(enc.parameters()) + list(a_probe.parameters()) + list(b_probe.parameters()),
         lr=0.0001)

    X = []
    A = []
    ast = []
    bst = []
    est = []

    import random

    for i in range(0, 1000000):
        a = random.randint(0, 4)
        # env.render()

        x, agent_state, block_state, exo_state = env.get_obs()
        env.step(a)

        A.append(a)
        X.append(x)
        ast.append(agent_state)
        bst.append(block_state)
        est.append(exo_state)

    X = np.asarray(X).astype('float32')
    A = np.asarray(A).astype('int64')
    ast = np.array(ast).astype('int64')
    bst = np.array(bst).astype('int64')
    est = np.array(est).astype('int64')

    eave = []

    for j in range(0, 100000):
        xt, xtk, k, a, astate, bstate, estate = sample_batch(X, A, ast, bst, est, 256)

        st, vql_1 = enc(xt)
        stk, vql_2 = enc(xtk)

        # print(a[10:20])
        # print(astate[10:20])

        ac_loss = ac(st, stk, k, a)
        ap_loss, ap_acc = a_probe(st, astate)
        bp_loss, bp_acc = b_probe(st, bstate)
        ep_loss, ep_acc = e_probe(st, estate)

        loss = ac_loss + ap_loss + bp_loss + ep_loss + vql_1 + vql_2
        loss.backward()

        opt.step()
        opt.zero_grad()

        if j % 100 == 0:
            print(j, ac_loss.item(), 'A_acc', ap_acc.item(), 'B_acc', bp_acc.item(), 'E_acc', ep_acc.item())

    h = []
    all_astate = []
    all_bstate = []

    enc.eval()

    for j in range(0, 490000, 256):
        h.append(enc(torch.Tensor(X[j: j + 256]).cuda())[0].data.cpu())

        all_astate.append(torch.Tensor(ast[j: j + 256]))
        all_bstate.append(torch.Tensor(bst[j: j + 256]))

    h = torch.cat(h, dim=0)

    all_astate = torch.cat(all_astate, dim=0)
    all_bstate = torch.cat(all_bstate, dim=0)

    ns = 1500

    print('kmeans start')
    kmeans = KMeans(n_clusters=ns, max_iter=300, n_init=10, verbose=1).fit(h)
    print('kmeans end')

    print('h shape', h.shape)

    na = 5
    counts = np.zeros((ns, na, ns))

    c2s = {}

    for j in range(0, h.shape[0] - 1):
        print('C', kmeans.labels_[j], 'a', A[j], 'A1', (all_astate[j] // env.m).item(), 'A2',
              (all_astate[j] % env.m).item(), 'B1', (all_bstate[j] // env.m).item(), 'B2',
              (all_bstate[j] % env.m).item())

        c2s[kmeans.labels_[j]] = (
        int((all_astate[j] // env.m).item()), int((all_astate[j] % env.m).item()), int((all_bstate[j] // env.m).item()),
        int((all_bstate[j] % env.m).item()))

        s = kmeans.labels_[j]
        sn = kmeans.labels_[j + 1]
        a = A[j]

        counts[s, a, sn] += 1

    np.save('counts.npy', counts)
    import pickle

    pickle.dump(c2s, open('dict.pkl', 'wb'))
