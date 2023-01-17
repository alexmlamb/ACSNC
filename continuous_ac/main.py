from room_env import RoomEnv
import matplotlib
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from models import Encoder, Probe, AC, LatentForward
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import wandb

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


def sample_example(X, A, ast, est, max_k=5):
    N = X.shape[0]
    t = random.randint(0, N - max_k - 1)
    k = random.randint(1, max_k)

    return (X[t], X[t + 1], X[t + k], k, A[t], ast[t], est[t])


def sample_batch(X, A, ast, est, bs, max_k=5):
    xt = []
    xtn = []
    xtk = []
    klst = []
    astate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, est, max_k=max_k)
        xt.append(lst[0])
        xtn.append(lst[1])
        xtk.append(lst[2])
        klst.append(lst[3])
        alst.append(lst[4])
        astate.append(lst[5])
        estate.append(lst[6])

    xt = torch.Tensor(np.array(xt)).to(device)
    xtn = torch.Tensor(np.array(xtn)).to(device)
    xtk = torch.Tensor(np.array(xtk)).to(device)
    klst = torch.Tensor(np.array(klst)).long().to(device)
    alst = torch.Tensor(np.array(alst)).to(device)
    astate = torch.Tensor(np.array(astate)).to(device)
    estate = torch.Tensor(np.array(estate)).to(device)

    return xt, xtn, xtk, klst, alst, astate, estate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name', default='acsnc',
                            help='name of the wandb project')
    wandb_args.add_argument('--use-wandb', action='store_true',
                            help='use Weight and bias visualization lib')
    # training args
    train_args = parser.add_argument_group('wandb setup')
    train_args.add_argument("--opr", default="generate-data",
                            choices=['generate-data', 'train', 'cluster-latent'])
    train_args.add_argument("--latent-dim", default=256, type=int)
    train_args.add_argument("--k_embedding_dim", default=45, type=int)
    train_args.add_argument("--max_k", default=5, type=int)

    # process arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb init
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in train_args._group_actions})

    # Train
    env = RoomEnv()

    ac = AC(din=args.latent_dim, nk=args.k_embedding_dim, nact=2).to(device)
    enc = Encoder(100 * 100, args.latent_dim).to(device)
    forward = LatentForward(args.latent_dim, 2).to(device)
    a_probe = Probe(args.latent_dim, 2).to(device)
    b_probe = Probe(args.latent_dim, 2).to(device)
    e_probe = Probe(args.latent_dim, 2).to(device)

    if args.opr == 'generate-data':
        X = []
        A = []
        ast = []
        est = []

        for i in tqdm(range(0, 500000)):
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

        pickle.dump({'X': X, 'A': A, 'ast': ast, 'est': est}, open('dataset.p', 'wb'))

        print('data generated and stored in dataset.p')
    elif args.opr == 'train':
        dataset = pickle.load(open('dataset.p', 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        from ema_pytorch import EMA

        ema_enc = EMA(enc, beta=0.99)
        ema_forward = EMA(forward, beta=0.99)
        ema_a_probe = EMA(a_probe.enc, beta=0.99)

        opt = torch.optim.Adam(list(ac.parameters())
                               + list(enc.parameters())
                               + list(a_probe.parameters())
                               + list(b_probe.parameters())
                               + list(forward.parameters()))

        for j in range(0, 200000):
            ac.train()
            enc.train()
            a_probe.train()
            forward.train()
            xt, xtn, xtk, k, a, astate, estate = sample_batch(X, A, ast, est, 128, max_k=args.max_k)
            astate = torch.round(astate, decimals=3)

            # print('-----')

            # xjoin = torch.cat([xt,xtn,xtk],dim=0)
            # sjoin = enc(xjoin)
            # st, stn, stk = torch.chunk(sjoin, 3, dim=0)

            st = enc(xt)
            stk = enc(xtk)

            stn = ema_enc(xtn)

            ac_loss = ac(st, stk, k, a)
            ap_loss, ap_abserr = a_probe.loss(st, astate)
            ep_loss = ap_loss * 0.0

            z_loss, z_pred = forward.loss(st, stn, a)

            # raise Exception()

            loss = ac_loss + ap_loss + ep_loss + z_loss
            loss.backward()

            opt.step()
            opt.zero_grad()

            ema_forward.update()
            ema_a_probe.update()
            ema_enc.update()

            if j % 100 == 0:
                print(j, ac_loss.item(), 'A_loss', ap_abserr.item(), 'Asqr_loss', ap_loss.item())
                if args.use_wandb:
                    wandb.log(
                        {'update': j,
                         'ac-loss': ac_loss.item(),
                         'a-loss': ap_abserr.item(),
                         'asqr-loss': ap_loss.item()})

                # print('forward test')
                # print('true[t]', astate[0])
                # print('s[t]', a_probe.enc(st)[0], 'a[t]', a[0])
                # print('s[t+1]', a_probe.enc(stn)[0], 'z[t+1]', a_probe.enc(z_pred)[0])

            ema_a_probe.eval()


            # ema_forward.eval()
            # ema_enc.eval()

            def vectorplot(a_use, name):

                # make grid
                action = []
                xl = []
                for a in range(2, 99, 5):
                    for b in range(2, 99, 10):
                        action.append(a_use)
                        true_s = [a * 1.0 / 100, b * 1.0 / 100]
                        x = env.synth_obs(ap=true_s)
                        xl.append(x)

                action = torch.Tensor(np.array(action)).to(device)
                xl = torch.Tensor(xl).to(device)
                print(xl.shape, action.shape)
                zt = ema_enc(xl)
                ztn = ema_forward(zt, action)
                st_inf = ema_a_probe(zt)
                stn_inf = ema_a_probe(ztn)
                print('st', st_inf[30], 'stn', stn_inf[30])

                px = st_inf[:, 0]
                py = stn_inf[:, 1]
                pu = stn_inf[:, 0] - st_inf[:, 0]
                pv = stn_inf[:, 1] - st_inf[:, 1]

                plt.quiver(px.data.cpu(), py.data.cpu(), 0.5 * pu.data.cpu(), 0.5 * pv.data.cpu())
                plt.title(name + " " + str(a_use))
                plt.savefig('vectorfield_%s.png' % name)

                plt.clf()

                return xl, action


            def squareplot(x_r, a_r):

                true_s = [0.4, 0.4]
                xl = env.synth_obs(ap=true_s)
                xl = torch.Tensor(xl).to(device).unsqueeze(0)

                xl = torch.cat([xl, x_r], dim=0)

                zt = ema_enc(xl)

                st_lst = []

                a_lst = [[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1], [-0.1, 0.0], [-0.1, 0.0], [0.0, -0.1],
                         [0.0, -0.1],
                         [0.0, 0.0], [0.0, 0.0]]
                for a in a_lst:
                    action = torch.Tensor(np.array(a)).to(device).unsqueeze(0)
                    action = torch.cat([action, a_r], dim=0)
                    st = ema_a_probe(zt)
                    st_lst.append(st.data.cpu()[0:1])
                    zt = ema_forward(zt, action)
                    print('st', st[0:1])
                    print('action', a)

                st_lst = torch.cat(st_lst, dim=0)

                true_sq = np.array(
                    [[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6], [0.4, 0.6], [0.4, 0.5],
                     [0.4, 0.4], [0.4, 0.4]])

                plt.plot(st_lst[:, 0].numpy(), st_lst[:, 1].numpy())
                plt.plot(true_sq[:, 0], true_sq[:, 1])
                plt.ylim(0, 1)
                plt.xlim(0, 1)

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

                if args.use_wandb:
                    wandb.log({
                        'vectorfields/down': wandb.Image("vectorfield_down.png"),
                        'vectorfields/up': wandb.Image("vectorfield_up.png"),
                        'vectorfields/left': wandb.Image("vectorfield_left.png"),
                        'vectorfields/right': wandb.Image("vectorfield_right.png"),
                        'vectorfields/up-right': wandb.Image("vectorfield_up-right.png"),
                        'vectorfields/plan': wandb.Image("vectorfield_plan.png"),
                        'update': j
                    })

                # save
                torch.save({'ac': ac.state_dict(),
                            'enc': enc.state_dict(),
                            'forward': forward.state_dict(),
                            'a_probe': a_probe.state_dict(),
                            'b_probe': b_probe.state_dict(),
                            'e_probe': e_probe.state_dict()}, 'model.p')

    elif args.opr == 'cluster-latent':

        # load model
        model = torch.load('model.p', map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        a_probe.load_state_dict(model['a_probe'])
        enc.eval()
        a_probe.eval()

        # load-dataset
        dataset = pickle.load(open('dataset.p', 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # generate latent-states and ground them
        latent_states = []
        predicted_grounded_states = []
        for i in range(0, 100000, 256):
            with torch.no_grad():
                _latent_state = enc(torch.FloatTensor(X[i:i + 256]).to(device))
                latent_states += _latent_state.cpu().numpy().tolist()
                predicted_grounded_states += a_probe(_latent_state).cpu().numpy().tolist()

        predicted_grounded_states = np.array(predicted_grounded_states)
        grounded_states = np.array(ast[:len(latent_states)])
        latent_states = np.array(latent_states)

        # clustering
        kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto").fit(latent_states)
        predicted_labels = kmeans.predict(latent_states)

        # visualize and save
        plt.scatter(x=grounded_states[:, 0],
                    y=grounded_states[:, 1],
                    c=predicted_labels,
                    marker='.')
        plt.savefig('latent_cluster.png')
        plt.clf()
        plt.scatter(x=grounded_states[:, 0],
                    y=predicted_grounded_states[:, 0],
                    marker='.')
        plt.savefig('ground_vs_predicted_state.png')
        if args.use_wandb:
            wandb.log({'latent-cluster': wandb.Image("latent_cluster.png"),
                       'grounded-vs-predicted-state':
                           wandb.Image("ground_vs_predicted_state.png")})
    else:
        raise ValueError()
