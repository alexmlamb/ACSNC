from room_env import RoomEnv
import os, time
from os.path import join
from datetime import datetime
from room_obstacle_env import RoomObstacleEnv
from room_polygon_obstacle_env import RoomPolygonObstacleEnv
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
from emprical_mdp import EmpiricalMDP
from ema_pytorch import EMA
import os
import copy
import time
import torch.nn as nn
from dijkstra import make_ls, DP_goals

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


def abstract_path_sampler(empirical_mdp, abstract_horizon):
    plan = {'states': [], 'actions': []}

    init_state = np.random.choice(empirical_mdp.unique_states)
    plan['states'].append(init_state)

    while len(plan['states']) != abstract_horizon:
        current_state = plan['states'][-1]
        next_state_candidates = []
        for state in empirical_mdp.unique_states:
            if not np.isnan(empirical_mdp.transition[current_state][state]).all() and state != current_state:
                next_state_candidates.append(state)
        next_state = np.random.choice(next_state_candidates)
        plan['actions'].append(empirical_mdp.transition[current_state, next_state])
        plan['states'].append(next_state)

    return plan


def obs_sampler(dataset_obs, dataset_agent_states, state_labels, abstract_state):
    # np.random.seed(0)
    _filtered_obs = dataset_obs[state_labels == abstract_state]
    _filtered_agent_states = dataset_agent_states[state_labels == abstract_state]
    index = np.random.choice(range(len(_filtered_obs)))
    return _filtered_obs[index], _filtered_agent_states[index]


def sample_example(X, A, ast, est, max_k):
    N = X.shape[0]
    t = random.randint(0, N - max_k - 1)
    k = random.randint(1, max_k)

    return (X[t], X[t + 1], X[t + k], k, A[t], ast[t], est[t])


def sample_batch(X, A, ast, est, bs, max_k):
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


class LatentWrapper(nn.Module):
    def __init__(self, latent) -> None:
        super().__init__()
        self.latent = latent
        self.nu = 2

    def forward(self, z, a):
        return self.latent(z, a, detach = False)

class Cluster_Transform:
    def __init__(self, enc, fwd_dyn, device = 'cuda') -> None:
        self.enc = enc
        self.fwd_dyn = fwd_dyn 
        self.device = device

    def cluster_label_transform(self, X):
        # generate label [z, f(z, a_1), f(z, a_2), f(z, a_3), f(z, a_4)] from observations X for clustering
        # with z = enc(X) being the latent state and f(.,.) being the latent forward dynamics
        img = torch.FloatTensor(X).to(self.device)
        z = self.enc(img)
        
        actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(self.device)
        output = z
        for i in range(4):
            y = self.fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))
            output = torch.cat((output, y.detach()), axis = 1)

        return output

    def cluster_label_transform_latent(self, z):
        # augment the latent state z for clustering
        actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(self.device)
        output = z
        for i in range(4):
            y = self.fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))
            output = torch.cat((output, y.detach()), axis = 1)

        return output

# def cluster_label_transform(X, enc, fwd_dyn, device = 'cuda'):
#     # generate label [z, f(z, a_1), f(z, a_2), f(z, a_3), f(z, a_4)] from observations X for clustering
#     # with z = enc(X) being the latent state and f(.,.) being the latent forward dynamics
#     img = torch.FloatTensor(X).to(device)
#     z = enc(img)
    
#     actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(device)
#     output = z
#     for i in range(4):
#         y = fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))
#         output = torch.cat((output, y.detach()), axis = 1)

#     return output

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
                            choices=['generate-data', 'train', 'cluster-latent',
                                     'generate-mdp',
                                     'debug-abstract-random-plans',
                                     'debug-dijkstra-plans',
                                     'debug-dijkstra-plans-for-all-states',
                                     'trajectory-synthesis',
                                     'qplanner', 'traj_opt', 'high-low-plan'])
    train_args.add_argument("--latent-dim", default=256, type=int)
    train_args.add_argument("--num-data-samples", default=500000, type=int)
    train_args.add_argument("--k_embedding_dim", default=45, type=int)
    train_args.add_argument("--max_k", default=2, type=int)
    train_args.add_argument("--do-mixup", action='store_true', default=False)

    train_args.add_argument("--env", default='polygon-obs', choices=['rat', 'room', 'obstacle', 'polygon-obs'])

    train_args.add_argument('--exp_id', default = 'test', type = str)
    train_args.add_argument('--from_to', default = 0, nargs = "+", type = int)
    train_args.add_argument('--scaling_factor', default = 1.0, type = float)
    train_args.add_argument("--seed", default=0, type=int)

    # process arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # wandb init
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in train_args._group_actions})

    # Train
    if args.env == 'rat':
        from rat_env import RatEnvWrapper

        env = RatEnvWrapper()
    elif args.env == 'room':
        env = RoomEnv()
    elif args.env == 'obstacle':
        env = RoomObstacleEnv()
    elif args.env == 'polygon-obs':
        env = RoomPolygonObstacleEnv()

    ac = AC(din=args.latent_dim, nk=args.k_embedding_dim, nact=2).to(device)
    enc = Encoder(100 * 100, args.latent_dim).to(device)
    forward = LatentForward(args.latent_dim, 2).to(device)
    a_probe = Probe(args.latent_dim, 2).to(device)
    b_probe = Probe(args.latent_dim, 2).to(device)
    e_probe = Probe(args.latent_dim, 2).to(device)
    ema_enc = EMA(enc, beta=0.99)
    ema_forward = EMA(forward, beta=0.99)
    ema_a_probe = EMA(a_probe.enc, beta=0.99)

    field_folder = os.path.join(os.getcwd(), "fields")  # + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')
    plan_folder = os.path.join(os.getcwd(), "fields")  # + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')
    dataset_path = os.path.join(os.getcwd(), 'data', 'dataset.p')
    model_path = os.path.join(os.getcwd(), 'data', 'model.p')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    os.makedirs(field_folder, exist_ok=True)
    os.makedirs(plan_folder, exist_ok=True)

    if args.opr == 'generate-data':
        X = []
        A = []
        ast = []
        est = []

        for i in tqdm(range(0, args.num_data_samples)):
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

        pickle.dump({'X': X, 'A': A, 'ast': ast, 'est': est}, open(dataset_path, 'wb'))

        print(f'data generated and stored in {dataset_path}')
    elif args.opr == 'train':
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        opt = torch.optim.Adam(list(ac.parameters())
                               + list(enc.parameters())
                               + list(a_probe.parameters())
                               + list(b_probe.parameters())
                               + list(forward.parameters()), lr=0.0001)

        colors = iter(plt.cm.inferno_r(np.linspace(.25, 1, 200000)))
        print('Num samples', X.shape[0])

        print('Run K-mneas')
        kmeans = KMeans(n_clusters=20, verbose=1).fit(A)
        print(' K-Means done')

        A = np.concatenate([A, kmeans.labels_.reshape((A.shape[0], 1))], axis=1)

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

            # do_bn = (j < 5000)

            st = enc(xt)
            stk = enc(xtk)

            stn = enc(xtn)

            ac_loss = ac(st, stk, k, a)
            ap_loss, ap_abserr = a_probe.loss(st, astate)
            ep_loss = ap_loss * 0.0

            z_loss, z_pred = forward.loss(st, stn, a)

            # raise Exception()

            loss = 0

            if args.do_mixup:
                # add probing loss in mixed hidden states.
                mix_lamb = random.uniform(0, 1)
                mix_ind = torch.randperm(st.shape[0])

                st_mix = st * mix_lamb + st[mix_ind] * (1 - mix_lamb)
                # astate_mix = astate*mix_lamb + astate[mix_ind]*(1-mix_lamb)
                # ap_loss_mix, _ = a_probe.loss(st_mix, astate_mix)
                # loss += ap_loss_mix

                stn_mix = stn * mix_lamb + stn[mix_ind] * (1 - mix_lamb)
                z_loss_mix, _ = forward.loss(st_mix, stn_mix, a, do_detach=False)

                loss += z_loss_mix

            loss += ac_loss + ap_loss + ep_loss + z_loss
            loss.backward()

            opt.step()
            opt.zero_grad()

            ema_forward.update()
            ema_a_probe.update()
            ema_enc.update()

            if j % 100 == 0:
                print(j, 'AC_loss', ac_loss.item(), 'A_loss', ap_abserr.item(), 'Asqr_loss', ap_loss.item(), 'Z_loss',
                      z_loss.item())
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
            ema_forward.eval()
            ema_enc.eval()


            def vectorplot(a_use, name):

                fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
                fontdict = {'fontsize': 28, 'fontweight': 'bold'}

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

                # plot the quivers
                ax1.grid('on')
                ax1.plot(px.data.cpu(), py.data.cpu(), linewidth=1, color=next(colors))
                ax1.quiver(px.data.cpu(), py.data.cpu(), 0.5 * pu.data.cpu(), 0.5 * pv.data.cpu())
                ax1.set_title(name + " " + str(a_use))

                ax1.set_ylabel(rf"y (pixels)", fontdict=fontdict)
                ax1.set_xlabel(rf"x (pixels)", fontdict=fontdict)
                ax1.tick_params(axis='both', which='major', labelsize=28)
                ax1.tick_params(axis='both', which='minor', labelsize=18)
                ax1.set_title(rf"State Trajectories: {name} {a_use}.", fontdict=fontdict)
                ax1.legend(loc="center left", fontsize=8)

                fig.savefig(join(field_folder, rf"field_{name}.jpg"), dpi=79, bbox_inches='tight', facecolor='None')

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.clf()
                time.sleep(.01)

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

                fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                fontdict = {'fontsize': 28, 'fontweight': 'bold'}

                ax.grid('on')

                ax.plot(st_lst[:, 0].numpy(), st_lst[:, 1].numpy(), linewidth=2, color=next(colors))
                ax.plot(true_sq[:, 0], true_sq[:, 1], linewidth=2, color="magenta")
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 1)
                ax.set_ylabel(rf"y (pixels)", fontdict=fontdict)
                ax.set_xlabel(rf"x (pixels)", fontdict=fontdict)
                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.tick_params(axis='both', which='minor', labelsize=18)

                ax.set_title("Square Plan", fontdict=fontdict)
                fig.savefig(join(plan_folder, f"plan.jpg"), dpi=79, bbox_inches='tight', facecolor='None')
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
                        'fields/down': wandb.Image(join(field_folder, "field_down.jpg")),
                        'fields/up': wandb.Image(join(field_folder, "field_up.jpg")),
                        'fields/left': wandb.Image(join(field_folder, "field_left.jpg")),
                        'fields/right': wandb.Image(join(field_folder, "field_right.jpg")),
                        'fields/up-right': wandb.Image(join(field_folder,
                                                            "field_up-right.jpg")),
                        'fields/plan': wandb.Image(join(plan_folder,
                                                        "plan.jpg")),
                        'update': j
                    })

                # save
                torch.save({'ac': ac.state_dict(),
                            'enc': enc.state_dict(),
                            'forward': forward.state_dict(),
                            'a_probe': a_probe.state_dict(),
                            'b_probe': b_probe.state_dict(),
                            'e_probe': e_probe.state_dict()}, model_path)
    elif args.opr == 'cluster-latent':

        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        a_probe.load_state_dict(model['a_probe'])
        enc = enc.eval().to(device)
        a_probe = a_probe.eval().to(device)

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)

        cluster_trans = Cluster_Transform(enc, forward, device = device)

        # load-dataset
        dataset = pickle.load(open('data/dataset.p', 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        
        # manually check obstacle detection
        plt.figure()
        k = 0
        grounded_traj = ast[k*100:(k+30)*100]

        # plot the obstacle
        if isinstance(env, RoomPolygonObstacleEnv):
            for obs in env.obs_lst:
                x_coords,y_coords = obs.exterior.xy
                plt.plot(x_coords, y_coords, color = 'k') 

        plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color = 'blue', linewidth = 0.3)
        plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], s = 2, color = 'blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.savefig('polygon_obstacle_detection_check.png', dpi = 600, format = 'png')
        plt.clf()

        print('data loaded')

        # generate latent-states and ground them
        aug_latent_states = []
        predicted_grounded_states = []
        for i in range(0, X.shape[0], 256):
            with torch.no_grad():
                _aug_latent_state = cluster_trans.cluster_label_transform(X[i:i+256])
                # _latent_state = enc(torch.FloatTensor(X[i:i + 256]).to(device))
                aug_latent_states += _aug_latent_state.cpu().numpy().tolist()
                predicted_grounded_states += a_probe(_aug_latent_state[:,:256]).cpu().numpy().tolist()

        predicted_grounded_states = np.array(predicted_grounded_states)
        grounded_states = np.array(ast[:len(aug_latent_states)])
        aug_latent_states = np.array(aug_latent_states)
        latent_state = aug_latent_states[:,:256]

        print('about to run kmeans')

        # clustering
        kmeans = KMeans(n_clusters=50, random_state=0).fit(aug_latent_states)
        predicted_labels = kmeans.predict(aug_latent_states)
        centroids = a_probe(torch.FloatTensor(kmeans.cluster_centers_[:,:256]).to(device)).cpu().detach().numpy()

        # visualize and save
        kmean_plot_fig = plt.figure()
        plt.scatter(x=grounded_states[:, 0],
                    y=grounded_states[:, 1],
                    c=predicted_labels,
                    marker='.')
        plt.scatter(x=centroids[:, 0],
                    y=centroids[:, 1],
                    marker="*")

        for centroid_i, centroid in enumerate(centroids):
            plt.text(centroid[0], centroid[1], str(centroid_i), horizontalalignment='center', fontsize=8, color='black')

        pickle.dump(
            {'kmeans': kmeans, 'kmeans-plot': copy.deepcopy(kmean_plot_fig), 'grounded-cluster-center': centroids},
            open('kmeans_info.p', 'wb'))
        plt.savefig(join(field_folder, 'latent_cluster.png'))
        plt.clf()

        plt.scatter(x=grounded_states[:, 0],
                    y=predicted_grounded_states[:, 0],
                    marker='.')
        plt.savefig(join(field_folder, 'ground_vs_predicted_state.png'))
        if args.use_wandb:
            wandb.log({'latent-cluster': wandb.Image(join(field_folder, "latent_cluster.png")),
                       'grounded-vs-predicted-state': wandb.Image(join(field_folder, "ground_vs_predicted_state.png"))})
            wandb.save(glob_str='kmeans.p', policy='now')
    elif args.opr == 'generate-mdp':
        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()
        
        forward.load_state_dict(model['forward'])
        forward.eval().to(device)
        cluster_trans = Cluster_Transform(enc, forward, device = device)

        # load clustering
        kmeans_info = pickle.load(open('kmeans_info.p', 'rb'))
        kmeans = kmeans_info['kmeans']
        kmeans_fig = kmeans_info['kmeans-plot']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # generate latent-states and find corresponding label
        latent_states, states_label = [], []
        for i in range(0, len(X), 256):
            with torch.no_grad():
                _aug_latent_state = cluster_trans.cluster_label_transform(X[i:i+256])
                latent_states += _aug_latent_state[:,:256].cpu().numpy().tolist()
                states_label += kmeans.predict(_aug_latent_state.cpu().numpy().tolist()).tolist()

        next_state = np.array(states_label[1:])
        next_state = next_state[np.abs(A[:-1]).sum(1) < 0.1]
        states_label = np.array(states_label[:-1])[np.abs(A[:-1]).sum(1) < 0.1]
        A = A[:-1]
        A = A[np.abs(A).sum(1) < 0.1]

        print(states_label.shape, A.shape, next_state.shape)

        empirical_mdp = EmpiricalMDP(state=states_label,
                                     action=A,
                                     next_state=next_state,
                                     reward=np.zeros_like(A))

        # draw action vectors on cluster-mdp
        for cluster_i, cluster_center in enumerate(grounded_cluster_centers):
            for action in [_ for _ in empirical_mdp.transition[empirical_mdp.unique_states_dict[cluster_i]] if
                           not np.isnan(_).all()]:
                plt.quiver(cluster_center[0], cluster_center[1], action[0], action[1])
                print('quiver', cluster_center[0], cluster_center[1], action[0], action[1])
        plt.savefig(join(field_folder, 'latent_cluster_with_action_vector.png'))
        plt.clf()

        transition_img = empirical_mdp.visualize_transition(save_path=join(field_folder, 'transition_img.png'))
        # save
        pickle.dump(empirical_mdp, open('empirical_mdp.p', 'wb'))

        # save
        if args.use_wandb:
            wandb.log({'mdp': wandb.Image(join(field_folder, "transition_img.png"))})
            # wandb.log({'latent-cluster-with-action-vector': wandb.Image(join(field_folder,'latent_cluster_with_action_vector.png'))})
            wandb.save(glob_str='empirical_mdp.p', policy="now")

    elif args.opr == 'debug-abstract-random-plans':

        # load abstract mdp
        empirical_mdp = pickle.load(open('empirical_mdp.p', 'rb'))

        # load clustering
        kmeans_info = pickle.load(open('kmeans_info.p', 'rb'))
        kmeans = kmeans_info['kmeans']
        # kmeans_fig = kmeans_info['kmeans-plot']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load dynamics
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        X = X[np.abs(A).sum(1) < 0.1]
        ast = ast[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        # generate random plans over abstract states
        abstract_plans = [
            abstract_path_sampler(empirical_mdp, abstract_horizon=2),
            abstract_path_sampler(empirical_mdp, abstract_horizon=2),
            abstract_path_sampler(empirical_mdp, abstract_horizon=3),
            abstract_path_sampler(empirical_mdp, abstract_horizon=3),
            abstract_path_sampler(empirical_mdp, abstract_horizon=4),
            abstract_path_sampler(empirical_mdp, abstract_horizon=4),
            abstract_path_sampler(empirical_mdp, abstract_horizon=5),
            abstract_path_sampler(empirical_mdp, abstract_horizon=5),
            abstract_path_sampler(empirical_mdp, abstract_horizon=10),
            abstract_path_sampler(empirical_mdp, abstract_horizon=10),
            abstract_path_sampler(empirical_mdp, abstract_horizon=15),
            abstract_path_sampler(empirical_mdp, abstract_horizon=15)]

        # rollout
        max_rollout_steps = 1000
        debug_plan_plots_dir = os.path.join(os.getcwd(), 'debug_plots')
        os.makedirs(debug_plan_plots_dir, exist_ok=True)
        for plan_i, plan in enumerate(abstract_plans):

            # initial-step
            obs, true_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=plan['states'][0])
            step_count = 0
            visited_states = [plan['states'][0]]

            while step_count < max_rollout_steps:

                step_action = plan['actions'][len(visited_states) - 1]

                # step into env.
                env.agent_pos = true_agent_state
                env.step(step_action)
                next_obs, _, _ = env.get_obs()
                step_count += 1
                with torch.no_grad():
                    _latent_state = enc(torch.FloatTensor(next_obs).to(device).unsqueeze(0)).cpu().numpy().tolist()
                next_state = kmeans.predict(_latent_state)[0]

                # check for cluster switch
                if next_state != visited_states[-1]:
                    if plan['states'][len(visited_states)] == next_state:  # check if the plan matches
                        visited_states.append(next_state)
                    else:
                        visited_states.append(next_state)
                        break  # if the next-abstract state does not match with actual plan

                # transition
                obs = next_obs

                if len(visited_states) == len(plan['states']):
                    break

            # log success/failure
            if visited_states == plan['states']:
                print(f'{plan_i}:  Success: '
                      f'\n\t => Abstract Horizon: {len(plan["states"])}'
                      f'\n\t => Low-Level Steps: {step_count} '
                      f'\n\t => Original Plan: 'f'{plan["states"]}'
                      f'\n\t => Executed Plan: {visited_states} \n')
            else:
                print(f'{plan_i}: Failure: '
                      f'\n\t => Abstract Horizon: {len(plan["states"])}'
                      f'\n\t => Low-Level Steps: {step_count}'
                      f'\n\t => Original Plan: {plan["states"]}'
                      f'\n\t  => Executed Plan: {visited_states}')

            kmeans_fig = copy.deepcopy(kmeans_info['kmeans-plot'])
            original_plan_data = np.array([grounded_cluster_centers[x] for x in plan["states"]])
            plt.plot(original_plan_data[:, 0], original_plan_data[:, 1], color='black', label='original-plan')
            plt.scatter([original_plan_data[0, 0]], [original_plan_data[0, 1]], marker="o", color='black', )
            plt.scatter([original_plan_data[-1, 0]], [original_plan_data[-1, 1]], marker="s", color='black')

            visited_plan_data = np.array([grounded_cluster_centers[x] for x in visited_states])
            plt.plot(visited_plan_data[:, 0] + 0.02, visited_plan_data[:, 1] + 0.02, color='red', label='executed-plan')
            plt.scatter([visited_plan_data[0, 0] + 0.02], [visited_plan_data[0, 1] + 0.02], marker="o", color='red')
            plt.scatter([visited_plan_data[-1, 0] + 0.02], [visited_plan_data[-1, 1] + 0.02], marker="s", color='red')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=False)
            plt.savefig(os.path.join(debug_plan_plots_dir, f'{plan_i}.png'))
            plt.clf()

    elif args.opr == 'debug-dijkstra-plans':
        dijkstra_plan_dir = os.path.join(os.getcwd(), 'dijkstra-plans')
        os.makedirs(dijkstra_plan_dir, exist_ok=True)

        # load abstract mdp
        empirical_mdp = pickle.load(open('empirical_mdp.p', 'rb'))

        # load clustering
        kmeans_info = pickle.load(open('kmeans_info.p', 'rb'))
        kmeans = kmeans_info['kmeans']
        kmeans_fig = kmeans_info['kmeans-plot']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load dynamics
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        X = X[np.abs(A).sum(1) < 0.1]
        ast = ast[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        num_states, num_actions, _ = empirical_mdp.discrete_transition.shape
        ls, _ = make_ls(torch.Tensor(empirical_mdp.discrete_transition), num_states, num_actions)

        for plan_i, (init_state, goal_state, dp_step_use) in enumerate([(1, 47, 1),
                                                                        (31, 22, 1),
                                                                        (2, 39, 1),
                                                                        (12, 46, 1)]):

            current_state = init_state
            obs, true_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=init_state)
            executed_plan = [current_state]
            max_steps = 100
            step_count = 0
            distance_to_goal = np.inf
            obs_history = [copy.deepcopy(true_agent_state)]
            while current_state != goal_state and step_count < max_steps:
                step_count += 1
                distance_to_goal, g, step_action_idx = DP_goals(ls, init_state=current_state, goal_index=goal_state,
                                                                dp_step=dp_step_use, code2ground={})

                step_action = empirical_mdp.discrete_action_space[step_action_idx]

                env.agent_pos = true_agent_state
                env.step(step_action)
                next_obs, next_agent_pos, _ = env.get_obs()

                with torch.no_grad():
                    _latent_state = enc(torch.FloatTensor(next_obs).to(device).unsqueeze(0)).cpu().numpy().tolist()
                next_state = kmeans.predict(_latent_state)[0]

                obs = next_obs
                current_state = next_state
                true_agent_state = next_agent_pos

                executed_plan.append(current_state)
                obs_history.append(copy.deepcopy(next_agent_pos))

                print(next_agent_pos)
                # print('a', step_action)
                # print('g', g)
                # print('d', distance_to_goal)

            obs_history = np.array(obs_history)

            kmeans_fig = copy.deepcopy(kmeans_info['kmeans-plot'])
            executed_plan_data = np.array([grounded_cluster_centers[x] for x in executed_plan])
            plt.plot(executed_plan_data[:, 0], executed_plan_data[:, 1], color='black', label='high-level trajectory')
            plt.plot(obs_history[:, 0], obs_history[:, 1], color='pink', label='low-level trajectory')
            plt.scatter(obs_history[:, 0], obs_history[:, 1], color='pink')

            init_ground_state_x, init_ground_state_y = grounded_cluster_centers[init_state]
            goal_ground_state_x, goal_ground_state_y = grounded_cluster_centers[goal_state]
            plt.scatter([init_ground_state_x], [init_ground_state_y], marker="o", color='red')
            plt.scatter([goal_ground_state_x], [goal_ground_state_y], marker="s", color='red')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=False)
            plt.savefig(os.path.join(dijkstra_plan_dir, f'{plan_i}.png'))
            plt.clf()
            print(executed_plan)

    elif args.opr == 'debug-dijkstra-plans-for-all-states':
            dijkstra_plan_dir = os.path.join(os.getcwd(), 'dijkstra-plans-for-all-states')
            os.makedirs(dijkstra_plan_dir, exist_ok=True)

            # load abstract mdp
            empirical_mdp = pickle.load(open('empirical_mdp.p', 'rb'))

            # load clustering
            kmeans_info = pickle.load(open('kmeans_info.p', 'rb'))
            kmeans = kmeans_info['kmeans']
            kmeans_fig = kmeans_info['kmeans-plot']
            grounded_cluster_centers = kmeans_info['grounded-cluster-center']

            # load dynamics
            model = torch.load(model_path, map_location=torch.device('cpu'))
            enc.load_state_dict(model['enc'])
            enc.eval()

            # load-dataset
            dataset = pickle.load(open(dataset_path, 'rb'))
            X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
            X = X[np.abs(A).sum(1) < 0.1]
            ast = ast[np.abs(A).sum(1) < 0.1]
            A = A[np.abs(A).sum(1) < 0.1]

            num_states, num_actions, _ = empirical_mdp.discrete_transition.shape
            ls, _ = make_ls(torch.Tensor(empirical_mdp.discrete_transition), num_states, num_actions)

            num_states, num_actions, _ = empirical_mdp.discrete_transition.shape
            ls, _ = make_ls(torch.Tensor(empirical_mdp.discrete_transition), num_states, num_actions)

            vectors = []
            for plan_i, (init_state, goal_state, dp_step_use) in enumerate([(_, 47, 1) for _ in range(num_states) if _ !=47]):

                current_state = init_state
                obs, true_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=init_state)
                executed_plan = [current_state]
                max_steps = 100
                step_count = 0
                distance_to_goal = np.inf
                obs_history = [copy.deepcopy(true_agent_state)]

                distance_to_goal, g, step_action_idx = DP_goals(ls, init_state=current_state, goal_index=goal_state,
                                                                dp_step=dp_step_use, code2ground={})

                step_action = empirical_mdp.discrete_action_space[step_action_idx]
                plt.quiver(grounded_cluster_centers[init_state][0], grounded_cluster_centers[init_state][1], step_action[0], step_action[1])

            kmeans_fig = copy.deepcopy(kmeans_info['kmeans-plot'])
            plt.scatter([grounded_cluster_centers[47][0]], [grounded_cluster_centers[47][1]], marker="o", color='red', s=20)
            plt.savefig(os.path.join(dijkstra_plan_dir, f'action_direction_to_goal_47.png'))



    elif args.opr == 'trajectory-synthesis':

        from trajectory_synthesis import sample_trajectory_batch, TSynth

        dataset = pickle.load(open('dataset.p', 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        maxmove = 0
        for j in range(0, len(ast) - 7):
            s = torch.Tensor(np.array(ast[j: j + 6]))
            mm = (s[-1] - s[0]).sum().item()
            diff = (s[0]).sum().item()

            if mm > maxmove and diff < 0.1:
                maxmove = mm
                print(s)

        print('maxmove', maxmove)

        k = 10
        tsynth = TSynth(dim=2, k=k).cuda()
        opt = torch.optim.Adam(tsynth.parameters(), lr=0.0001)

        for i in range(0, 300000):

            s = sample_trajectory_batch(ast, 256, k)
            loss, spred = tsynth.loss(s[:, 0:1], s[:, -1:], s)

            loss.backward()
            opt.step()
            opt.zero_grad()

            if i % 1000 == 0:
                print('-------------------------')
                print(i, loss)
                s0_test = torch.ones((1, 2)).cuda() * 0.0
                sk_test = torch.ones((1, 2)).cuda() * 1.0

                traj = tsynth(s0_test, sk_test)

                print('true start', s0_test)
                print('k', k)
                print('true end', sk_test)

                print('synth traj')
                print(traj.reshape((1, k, 2)))

    elif args.opr == 'qplanner':

        dataset = pickle.load(open('dataset.p', 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        kmeans = KMeans(n_clusters=10, verbose=0).fit(A)
        print(' K-Means done')

        A = np.concatenate([A, kmeans.labels_.reshape((A.shape[0], 1))], axis=1)

        print('A shape', A.shape)

        from qplan import QPlan

        myq = QPlan()

    elif args.opr == 'low-level-plan':

        model = torch.load('model.p', map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()

    elif args.opr == 'traj_opt':
        # load abstract mdp
        mdp_path = os.path.join(os.getcwd(), 'empirical_mdp.p')
        empirical_mdp = pickle.load(open(mdp_path, 'rb'))

        # load models
        model_path = os.path.join(os.getcwd(), 'model.p')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()

        forward.load_state_dict(model['forward'])
        forward.eval()

        a_probe.load_state_dict(model['a_probe'])

        # load clustering
        kmeans = pickle.load(open('kmeans_info.p', 'rb'))
        grounded_cluster_centers = a_probe(torch.FloatTensor(kmeans.cluster_centers_).to(device)).cpu().detach().numpy()

        # load-dataset
        dataset_path = os.path.join(os.getcwd(), 'dataset.p')
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        X = X[np.abs(A).sum(1) < 0.1]
        ast = ast[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        # initialization
        exp_id = args.exp_id
        traj_opt_data_path = os.path.join(os.getcwd(), 'high-low-data', f'{exp_id}.p')
        traj_opt_fig_dir = os.path.join(os.getcwd(), 'high-low-data')

        # specify start state and goal state
        from_to = args.from_to
        if not isinstance(from_to, list):
            from_to = [19, 27]

        # initial mdp state
        init_mdp_state = from_to[0]
        init_obs, init_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=init_mdp_state)
        init_lat_state = enc(torch.FloatTensor(init_obs).to(device).unsqueeze(0))

        nz = init_lat_state.size(1)
        nu = ast.shape[1]

        target_mdp_state = from_to[1]
        target_obs, target_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=target_mdp_state)
        target_lat_state = enc(torch.FloatTensor(target_obs).to(device).unsqueeze(0))

        scaling_factor = args.scaling_factor
        target_lat_state = init_lat_state + scaling_factor*(target_lat_state - init_lat_state)

        target_gt_agent_state = a_probe(target_lat_state)[0]
        target_mdp_state = kmeans.predict(target_lat_state.detach().cpu())[0]

        # initialize planning parameters
        n_batch, T, N = 50, 10, 10
        u_min, u_max = -0.2, 0.2

        # latent space dynamics
        dynamics = LatentWrapper(forward)
        dynamics.eval()

        from traj_opt.hj_prox.hj_prox_alg import Tracking_Cost, HJ_Prox_Optimizer
        tracking_cost_fcn = Tracking_Cost(dynamics, init_lat_state, target_lat_state)

        init_actions = u_min + (u_max - u_min)*torch.rand((n_batch, N, nu)).to(device)

        # closed-loop simulation
        lat_state_log = torch.zeros((T+1, 1, nz)).to(device)
        control_log = torch.zeros((T, 1, nu)).to(device)
        gt_agent_state_log = np.zeros((T+1, init_gt_agent_state.shape[0]))
        probe_lat_state_log = torch.zeros((T+1, init_gt_agent_state.shape[0])).to(device)

        z_t = init_lat_state
        lat_state_log[0] = z_t
        gt_agent_state_log[0] = copy.deepcopy(init_gt_agent_state)
        probe_lat_state_log[0] = a_probe(init_lat_state)

        mpc_time = []
        gt_agent_state = copy.deepcopy(init_gt_agent_state)

        t_param = 0.1
        for t in tqdm(range(T), desc = 'hj_mpc'):
            start_time = time.time()
            tracking_cost_fcn = Tracking_Cost(dynamics, z_t, target_lat_state)
            hj_optimizer = HJ_Prox_Optimizer(tracking_cost_fcn, init_actions, t_param, x_min = u_min, x_max = u_max)
            output_action, action_list = hj_optimizer.grad_descent(iter_num = 10, x_init = init_actions, t = t_param)

            rollout_costs = tracking_cost_fcn(output_action)
            selected_action = output_action[rollout_costs[0].argmin().item()]

            action = selected_action[0:1, :]

            control_log[t] = action

            print(f'agent state: {gt_agent_state}, goal state: {target_gt_agent_state}')
            print(f'action: {action}')

            env.agent_pos = gt_agent_state
            env.step(action[0].detach().cpu().numpy())
            next_obs, next_agent_pos, _ = env.get_obs()

            gt_agent_state_log[t+1] = copy.deepcopy(next_agent_pos)
            gt_agent_state = next_agent_pos

            next_lat_state = enc(torch.FloatTensor(next_obs).unsqueeze(0).to(device))
            z_t = next_lat_state

            lat_state_log[t+1] = z_t

            # TODO: update u_init in an adaptive manner
            init_actions = u_min + (u_max - u_min)*torch.rand((n_batch, N, nu)).to(device)
            init_actions[0,:] = torch.cat((selected_action[1:,:], torch.zeros((1, nu)).to(device)), dim = 0)

            run_time = time.time() - start_time
            mpc_time.append(run_time)

        mpc_data = {'grounded_states': gt_agent_state_log, 'actions': control_log, 'mpc_time': mpc_time,
                    'target_grounded_state': target_gt_agent_state, 'lat_state_log': lat_state_log}
        torch.save(mpc_data, traj_opt_data_path)

        # plot the trajectory
        plt.figure()
        grounded_traj = gt_agent_state_log
        target_grounded_state_np = target_gt_agent_state.detach().cpu().numpy()
        action_log = control_log.detach().cpu().numpy()

        # plot the obstacle
        plt.plot(np.array([0.501, 0.501]), np.array([0.001, 0.401]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.501, 0.501]), np.array([0.601, 1.01]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.201, 0.801]), np.array([0.401, 0.401]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.201, 0.801]), np.array([0.601, 0.601]), color = 'k', linewidth = 4)

        plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color = 'pink', label = 'HJ Prox')
        plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], color = 'pink')
        plt.scatter(grounded_traj[0, 0], grounded_traj[0, 1], marker = 'o', color = 'k', label = 'init')
        plt.scatter(target_grounded_state_np[0], target_grounded_state_np[1], marker = 's', color = 'r', label = 'target')
        plt.legend(loc='lower center', ncol=3, fancybox=True, shadow=False)
        plt.title(f'init. state: {init_mdp_state}, target state: {target_mdp_state}, runtime: {sum(mpc_time)}')
        # for i in range(grounded_traj.shape[0]-1):
        #     action = action_log[i, b_num, :]
        #     plt.quiver(grounded_traj[i, 0], grounded_traj[i, 1], action[0], action[1], scale = 1.0)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.savefig(os.path.join(traj_opt_fig_dir, f'{exp_id}.png'))
        plt.clf()

    elif args.opr == 'high-low-plan':
        # load abstract mdp
        mdp_path = os.path.join(os.getcwd(), 'empirical_mdp.p')
        empirical_mdp = pickle.load(open(mdp_path, 'rb'))

        # load models
        model_path = os.path.join(os.getcwd(), 'data', 'model.p')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval().to(device)

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)

        cluster_trans = Cluster_Transform(enc, forward, device = device)

        a_probe.load_state_dict(model['a_probe'])

        # load clustering
        kmeans_info = pickle.load(open('kmeans_info.p', 'rb'))
        kmeans = kmeans_info['kmeans']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load-dataset
        dataset_path = os.path.join(os.getcwd(), 'data', 'dataset.p')
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        
        # SC: need to double check
        X = X[:-1]
        ast = ast[:-1]
        A = A[:-1]
        
        X = X[np.abs(A).sum(1) < 0.1]
        ast = ast[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        # initialization
        exp_id = args.exp_id
        traj_opt_data_path = os.path.join(os.getcwd(), 'high_low_data', f'{exp_id}.p')
        traj_opt_fig_dir = os.path.join(os.getcwd(), 'high_low_data')
        os.makedirs(traj_opt_fig_dir, exist_ok = True)

        # specify start state and goal state
        from_to = args.from_to
        if not isinstance(from_to, list):
            from_to = [7,6]

        # initial mdp state
        init_mdp_state = from_to[0]
        init_obs, init_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=init_mdp_state)
        init_lat_state = enc(torch.FloatTensor(init_obs).to(device).unsqueeze(0))

        nz = init_lat_state.size(1)
        nu = ast.shape[1]

        target_mdp_state = from_to[1]
        target_obs, target_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=target_mdp_state)
        target_lat_state = enc(torch.FloatTensor(target_obs).to(device).unsqueeze(0))

        scaling_factor = args.scaling_factor
        target_lat_state = init_lat_state + scaling_factor*(target_lat_state - init_lat_state)

        target_gt_agent_state = a_probe(target_lat_state)[0]

        aug_target_lat_state = cluster_trans.cluster_label_transform_latent(target_lat_state)
        target_mdp_state = kmeans.predict(aug_target_lat_state.detach().cpu())[0]

        # initialize low level planning parameters
        n_batch, T, N = 10, 15, 3
        u_min, u_max = -0.2, 0.2

        # latent space dynamics
        dynamics = LatentWrapper(forward)
        dynamics.eval()
    
        from traj_opt.high_level_planner.dijkstra_planner import Dijkstra_Planner
        dijkstra_planner = Dijkstra_Planner(empirical_mdp)

        from traj_opt.hj_prox.hj_prox_alg import Tracking_Cost, HJ_Prox_Optimizer
        tracking_cost_fcn = Tracking_Cost(dynamics, init_lat_state, target_lat_state)

        init_actions = u_min + (u_max - u_min)*torch.rand((n_batch, N, nu)).to(device)

        # closed-loop simulation
        lat_state_log = torch.zeros((T+1, 1, nz)).to(device)
        control_log = torch.zeros((T, 1, nu)).to(device)
        gt_agent_state_log = np.zeros((T+1, init_gt_agent_state.shape[0]))
        probe_lat_state_log = torch.zeros((T+1, init_gt_agent_state.shape[0])).to(device)
        executed_mdp_states = []
        planned_mdp_states = []

        z_t = init_lat_state
        lat_state_log[0] = z_t
        gt_agent_state_log[0] = copy.deepcopy(init_gt_agent_state)
        probe_lat_state_log[0] = a_probe(init_lat_state)

        mpc_time = []
        gt_agent_state = copy.deepcopy(init_gt_agent_state)

        t_param = 0.1
        for t in tqdm(range(T), desc = 'hj_mpc'):
            start_time = time.time()

            # call high level planner
            aug_z_t = cluster_trans.cluster_label_transform_latent(z_t)
            current_mdp_state = kmeans.predict(aug_z_t.detach().cpu())[0]
            executed_mdp_states.append(current_mdp_state)

            if current_mdp_state != target_mdp_state:
                next_mdp_state = dijkstra_planner.step(current_mdp_state, target_mdp_state)
                planned_mdp_states.append(next_mdp_state)

                next_lat_state = kmeans.cluster_centers_[next_mdp_state][:256]
                next_lat_state = torch.FloatTensor(next_lat_state).unsqueeze(0).to(device)
            else:
                next_lat_state = target_lat_state
                planned_mdp_states.append(target_mdp_state)

            tracking_cost_fcn = Tracking_Cost(dynamics, z_t, next_lat_state)
            hj_optimizer = HJ_Prox_Optimizer(tracking_cost_fcn, init_actions, t_param, x_min = u_min, x_max = u_max)
            output_action, action_list = hj_optimizer.grad_descent(iter_num = 10, x_init = init_actions, t = t_param)

            rollout_costs = tracking_cost_fcn(output_action)
            selected_action = output_action[rollout_costs[0].argmin().item()]

            action = selected_action[0:1, :]

            control_log[t] = action

            print(f'agent state: {gt_agent_state}, goal state: {target_gt_agent_state}')
            print(f'action: {action}')

            env.agent_pos = gt_agent_state
            env.step(action[0].detach().cpu().numpy())
            next_obs, next_agent_pos, _ = env.get_obs()

            gt_agent_state_log[t+1] = copy.deepcopy(next_agent_pos)
            gt_agent_state = next_agent_pos

            next_lat_state = enc(torch.FloatTensor(next_obs).unsqueeze(0).to(device))
            z_t = next_lat_state

            lat_state_log[t+1] = z_t

            # TODO: update u_init in an adaptive manner
            init_actions = u_min + (u_max - u_min)*torch.rand((n_batch, N, nu)).to(device)
            init_actions[0,:] = torch.cat((selected_action[1:,:], torch.zeros((1, nu)).to(device)), dim = 0)

            run_time = time.time() - start_time
            mpc_time.append(run_time)

        mpc_data = {'grounded_states': gt_agent_state_log, 'actions': control_log, 'mpc_time': mpc_time,
                    'target_grounded_state': target_gt_agent_state, 'lat_state_log': lat_state_log, 
                    'executed_mdp_states': executed_mdp_states, 'planned_mdp_states': planned_mdp_states}
        
        torch.save(mpc_data, traj_opt_data_path)

        # plot the trajectory
        plt.figure()
        grounded_traj = gt_agent_state_log
        target_grounded_state_np = target_gt_agent_state.detach().cpu().numpy()
        action_log = control_log.detach().cpu().numpy()

        # plot the obstacle
        plt.plot(np.array([0.501, 0.501]), np.array([0.001, 0.401]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.501, 0.501]), np.array([0.601, 1.01]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.201, 0.801]), np.array([0.401, 0.401]), color = 'k', linewidth = 4)
        plt.plot(np.array([0.201, 0.801]), np.array([0.601, 0.601]), color = 'k', linewidth = 4)

        plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color = 'pink', label = 'HJ Prox')
        plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], color = 'pink')
        plt.scatter(grounded_traj[0, 0], grounded_traj[0, 1], marker = 'o', color = 'k', label = 'init')
        plt.scatter(target_grounded_state_np[0], target_grounded_state_np[1], marker = 's', color = 'r', label = 'target')
        plt.legend(loc='lower center', ncol=3, fancybox=True, shadow=False)
        plt.title(f'init. state: {init_mdp_state}, target state: {target_mdp_state}, runtime: {sum(mpc_time)}')
        # for i in range(grounded_traj.shape[0]-1):
        #     action = action_log[i, b_num, :]
        #     plt.quiver(grounded_traj[i, 0], grounded_traj[i, 1], action[0], action[1], scale = 1.0)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.savefig(os.path.join(traj_opt_fig_dir, f'{exp_id}.png'))
        plt.clf()

    else:
        raise ValueError()
