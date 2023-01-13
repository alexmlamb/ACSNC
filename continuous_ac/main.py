

from room_env import RoomEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from models import Encoder, Probe, AC, LatentForward
import torch
import numpy as np
import random

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

def sample_example(X, A, ast, est): 
    N = X.shape[0]
    maxk = 10 #5
    t = random.randint(0, N - maxk - 1)
    k = random.randint(1, maxk)

    return (X[t], X[t+1], X[t+k], k, A[t], ast[t], est[t])

def sample_batch(X, A, ast, est, bs):

    xt = []
    xtn = []
    xtk = []
    klst = []
    astate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, est)
        xt.append(lst[0])
        xtn.append(lst[1])
        xtk.append(lst[2])
        klst.append(lst[3])
        alst.append(lst[4])
        astate.append(lst[5])
        estate.append(lst[6])


    xt = torch.Tensor(np.array(xt)).cuda()
    xtn = torch.Tensor(np.array(xtn)).cuda()
    xtk = torch.Tensor(np.array(xtk)).cuda()
    klst = torch.Tensor(np.array(klst)).long().cuda()
    alst = torch.Tensor(np.array(alst)).cuda()
    astate = torch.Tensor(np.array(astate)).cuda()
    estate = torch.Tensor(np.array(estate)).cuda()

    return xt, xtn, xtk, klst, alst, astate, estate

if __name__ == "__main__":

    env = RoomEnv()

    ac = AC(256, nk=45, nact=2).cuda()
    enc = Encoder(100*100, 256).cuda()
    forward = LatentForward(256, 2).cuda()
    a_probe = Probe(256, 2).cuda()
    b_probe = Probe(256, 2).cuda()
    e_probe = Probe(256, 2).cuda()

    from ema_pytorch import EMA

    ema_enc = EMA(enc, beta = 0.99)
    ema_forward = EMA(forward, beta = 0.99, update_after_step=5000)
    ema_a_probe = EMA(a_probe.enc, beta = 0.99, update_after_step=5000)

    opt = torch.optim.Adam(list(ac.parameters()) + list(enc.parameters()) + list(a_probe.parameters()) + list(b_probe.parameters()) + list(forward.parameters()))

    X = []
    A = []
    ast = []
    est = []

    import random
    for i in range(0,50000):
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


    for j in range(0, 200000):
        ac.train()
        enc.train()
        a_probe.train()
        forward.train()
        xt, xtn, xtk, k, a, astate, estate = sample_batch(X, A, ast, est, 128)
        astate = torch.round(astate,decimals=3)

        #print('-----')

        xjoin = torch.cat([xt,xtn,xtk],dim=0)
        sjoin = enc(xjoin)
        st, stn, stk = torch.chunk(sjoin, 3, dim=0)


        #st = enc(xt)
        #stk = enc(xtk)
        #stn = enc(xtn)


        ac_loss = ac(st, stk, k, a)
        ap_loss, ap_abserr = a_probe.loss(st, astate)
        ep_loss = ap_loss*0.0

        z_loss, z_pred = forward.loss(st, stn, a)

        #raise Exception()

        loss = ac_loss + ap_loss + ep_loss + z_loss
        loss.backward()

        opt.step()
        opt.zero_grad()
        
        ema_forward.update()
        ema_a_probe.update()
        ema_enc.update()

        if j % 100 == 0:
            print(j, ac_loss.item(), 'A_loss', ap_abserr.item(), 'Asqr_loss', ap_loss.item())

            #print('forward test')
            #print('true[t]', astate[0])
            #print('s[t]', a_probe.enc(st)[0], 'a[t]', a[0])
            #print('s[t+1]', a_probe.enc(stn)[0], 'z[t+1]', a_probe.enc(z_pred)[0])

        #ema_a_probe.eval()
        #ema_forward.eval()
        #ema_enc.eval()


        def vectorplot(a_use, name):

            #make grid
            action = []
            xl = []
            for a in range(2,99,5):
                for b in range(2,99,10):
                    action.append(a_use)
                    true_s = [a*1.0/100, b*1.0/100]
                    x = env.synth_obs(ap=true_s)
                    xl.append(x)


            action = torch.Tensor(np.array(action)).cuda()
            xl = torch.Tensor(xl).cuda()
            print(xl.shape, action.shape)
            zt = enc(xl)
            ztn = forward(zt, action)
            st_inf = a_probe(zt)
            stn_inf = a_probe(ztn)
            print('st', st_inf[30], 'stn', stn_inf[30])

            px = st_inf[:,0]
            py = stn_inf[:,1]
            pu = stn_inf[:,0]-st_inf[:,0]
            pv = stn_inf[:,1]-st_inf[:,1]

            plt.quiver(px.data.cpu(),py.data.cpu(),0.5*pu.data.cpu(),0.5*pv.data.cpu())
            plt.title(name + " " + str(a_use))
            plt.savefig('vectorfield_%s.png' % name)
            
            plt.clf()


            return xl, action

        def squareplot(x_r, a_r):


            true_s = [0.4,0.4]
            xl = env.synth_obs(ap=true_s)
            xl = torch.Tensor(xl).cuda().unsqueeze(0)
            
            xl = torch.cat([xl, x_r], dim=0)

            zt = enc(xl)

            st_lst = []

            a_lst = [[0.1,0.0],[0.1,0.0],[0.0,0.1],[0.0,0.1],[-0.1,0.0],[-0.1,0.0],[0.0,-0.1],[0.0,-0.1],[0.0,0.0],[0.0,0.0]]
            for a in a_lst:
                action = torch.Tensor(np.array(a)).cuda().unsqueeze(0)
                action = torch.cat([action, a_r], dim=0)
                st = a_probe(zt)
                st_lst.append(st.data.cpu()[0:1])
                zt = forward(zt, action)
                print('st', st[0:1])
                print('action', a)

            st_lst = torch.cat(st_lst, dim=0)

            true_sq = np.array([[0.4,0.4],[0.5,0.4],[0.6,0.4],[0.6,0.5],[0.6,0.6],[0.5,0.6],[0.4,0.6],[0.4,0.5],[0.4,0.4],[0.4,0.4]])

            plt.plot(st_lst[:,0].numpy(), st_lst[:,1].numpy())
            plt.plot(true_sq[:,0], true_sq[:,1])
            plt.ylim(0,1)
            plt.xlim(0,1)

            plt.title("Square Plan")
            plt.savefig('vectorfield_plan.png')
            plt.clf()



        if True and j % 1000 == 0:
            vectorplot([0.0, 0.1], 'up')
            vectorplot([0.0, -0.1], 'down')
            vectorplot([-0.1, 0.0], 'left')
            vectorplot([0.1, 0.0], 'right')
            vectorplot([0.1, 0.1], 'up-right')
            x_r, a_r = vectorplot([-0.1, -0.1], 'down-left')

            squareplot(x_r, a_r)




