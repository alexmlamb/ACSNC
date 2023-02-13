

#from block_pull_env import BlockEnv
from block_traj_env import BlockEnv


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

def sample_example(X, A, ast, bst): 
    N = X.shape[0]
    maxk = 20
    t = random.randint(0, N - maxk - 1)
    k = random.randint(1, maxk)

    return (X[t], X[t + 1], X[t+k], k, A[t], ast[t], bst[t])

def sample_batch(X, A, ast, bst, bs):

    xt = []
    xt1 = []
    xtk = []
    klst = []
    astate = []
    bstate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, bst)
        xt.append(lst[0])
        xt1.append(lst[1])
        xtk.append(lst[2])
        klst.append(lst[3])
        alst.append(lst[4])
        astate.append(lst[5])
        bstate.append(lst[6])


    xt = torch.Tensor(np.array(xt)).cuda()
    xt1 = torch.Tensor(np.array(xt1)).cuda()
    xtk = torch.Tensor(np.array(xtk)).cuda()
    klst = torch.Tensor(np.array(klst)).long().cuda()
    alst = torch.Tensor(np.array(alst)).long().cuda()
    astate = torch.Tensor(np.array(astate)).long().cuda()
    bstate = torch.Tensor(np.array(bstate)).long().cuda()

    return xt, xt1, xtk, klst, alst, astate, bstate

if __name__ == "__main__":

    env = BlockEnv()

    ac = AC(256, n1=3, n2 = 3, n3 = 21, nact=5, m = env.m).cuda()
    a_probe_1 = Probe(256, env.m**2).cuda()
    a_probe_2 = Probe(256, env.m**2).cuda()
    b_probe_1 = Probe(256, env.m**2).cuda()
    b_probe_2 = Probe(256, env.m**2).cuda()
    c_probe = Probe(256 * 2, env.m**2).cuda()
    d_probe = Probe(256 * 2, env.m**2).cuda()

    opt = torch.optim.Adam(list(ac.parameters()) + list(a_probe_1.parameters()) + list(a_probe_2.parameters())+ list(b_probe_1.parameters()) + list(b_probe_2.parameters()) + list(c_probe.parameters()) + list(d_probe.parameters()), lr = 1e-4)

    X = []
    A = []
    ast = []
    bst = []
    est = []

    import random
    for i in range(0,500000):
        a = random.randint(0,4)
        #env.render()

        x, agent_state, block_state = env.get_obs()
        env.step(a)

        A.append(a)
        X.append(x)
        ast.append(agent_state)
        bst.append(block_state)

    X = np.asarray(X).astype('float32')
    A = np.asarray(A).astype('int64')
    ast = np.array(ast).astype('int64')
    bst = np.array(bst).astype('int64')

    for j in range(0, 200000):
        xt, xt1, xtk, k, a, astate, bstate = sample_batch(X, A, ast, bst, 256)

        

        ac_loss, rec_loss, kl_loss, st1, st2 = ac(xt.long(), xt1.long(), xtk.long(), k, a, train = True)
        
        with torch.no_grad():
            _, _, _, st1, st2 = ac(xt.long(), xt1.long(), xtk.long(), k, a, train = False)
        
        
        
        ap_loss1, ap_acc1 = a_probe_1(st1, astate)
        ap_loss2, ap_acc2 = a_probe_2(st2, astate)
        
        bp_loss1, bp_acc1 = b_probe_1(st1, bstate)
        bp_loss2, bp_acc2 = b_probe_2(st2, bstate)

        both_loss1, both_acc_1 = c_probe(torch.cat((st1, st2), dim = -1), astate)
        both_loss2, both_acc_2 = d_probe(torch.cat((st1, st2), dim = -1), bstate)
        
        loss = ac_loss + rec_loss + kl_loss + ap_loss1 + bp_loss1 + ap_loss2 + bp_loss2 + both_loss1 + both_loss2
        
        loss.backward()

        opt.step()
        opt.zero_grad()
        
        if j % 100 == 0:
            print(j, ac_loss.item(), rec_loss.item())
            print("Agent  Obstacle")
            print("Agent:" + str(ap_acc1.item()) + " " + str(ap_acc2.item()))
            print("Obstacle:" + str(bp_acc1.item()) + " " + str(bp_acc2.item()))


            print("BOTH:")
            print(str(both_acc_1.item()) + " " + str(both_acc_2.item()))



