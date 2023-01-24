from room_env import RoomEnv
import os, time
from os.path import join
from datetime import datetime
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
                            choices=['generate-data', 'train', 'cluster-latent', 'generate-mdp',
                                     'debug-abstract-plans'])
    train_args.add_argument("--latent-dim", default=256, type=int)
    train_args.add_argument("--k_embedding_dim", default=45, type=int)
    train_args.add_argument("--max_k", default=2, type=int)
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
    env = RoomEnv()

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

            do_bn = (j < 5000)

            st = enc(xt, do_bn)
            stk = enc(xtk, do_bn)

            stn = enc(xtn, do_bn)

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
                print(j, 'AC_loss', ac_loss.item(), 'A_loss', ap_abserr.item(), 'Asqr_loss', ap_loss.item(), 'Z_loss',
                      z_loss.item())
                if args.use_wandb:
                    wandb.log(
                        {'update': j,
                         'ac-loss': ac_loss.item(),
                         'a-loss': ap_abserr.item(),
                         'asqr-loss': ap_loss.item()})

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
        enc.eval()
        a_probe.eval()

        # load-dataset
        dataset = pickle.load(open('data/dataset.p', 'rb'))
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
        kmeans = KMeans(n_clusters=50, random_state=0).fit(latent_states)
        predicted_labels = kmeans.predict(latent_states)
        pickle.dump(kmeans, open('kmeans.p', 'wb'))

        # visualize and save
        plt.scatter(x=grounded_states[:, 0],
                    y=grounded_states[:, 1],
                    c=predicted_labels,
                    marker='.')
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

        # load clustering
        kmeans = pickle.load(open('kmeans.p', 'rb'))

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # generate latent-states and find corresponding label
        latent_states, states_label = [], []
        for i in range(0, len(X), 256):
            with torch.no_grad():
                _latent_state = enc(torch.FloatTensor(X[i:i + 256]).to(device))
                latent_states += _latent_state.cpu().numpy().tolist()
                states_label += kmeans.predict(_latent_state.cpu().numpy().tolist()).tolist()

        next_state = np.array(states_label[1:])
        next_state = next_state[np.abs(A[:-1]).sum(1) < 0.1]
        states_label = np.array(states_label)[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        print(states_label.shape, A.shape, next_state.shape)

        empirical_mdp = EmpiricalMDP(state=states_label,
                                     action=A,
                                     next_state=next_state,
                                     reward=np.zeros_like(A))
        # save 
        pickle.dump(empirical_mdp, open('empirical_mdp.p', 'wb'))
        transition_img = empirical_mdp.visualize_transition(save_path=join(field_folder, 'transition_img.png'))
        if args.use_wandb:
            wandb.log({'mdp': wandb.Image(join(field_folder, "transition_img.png"))})
            wandb.save(glob_str='empirical_mdp.p', policy="now")

    elif args.opr == 'debug-abstract-plans':
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
            _filtered_obs = dataset_obs[state_labels == abstract_state]
            _filtered_agent_states = dataset_agent_states[state_labels == abstract_state]
            index = np.random.choice(range(len(_filtered_obs)))
            return _filtered_obs[index], _filtered_agent_states[index]


        # load abstract mdp
        empirical_mdp = pickle.load(open('empirical_mdp.p', 'rb'))

        # load clustering
        kmeans = pickle.load(open('kmeans.p', 'rb'))

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
        for plan in abstract_plans:

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
                print(f'Success: '
                      f'\n\t => Abstract Horizon: {len(plan["states"])}'
                      f'\n\t => Low-Level Steps: {step_count} '
                      f'\n\t => Original Plan: 'f'{plan["states"]}'
                      f'\n\t => Executed Plan: {visited_states} \n')
            else:
                print(f'Failure: '
                      f'\n\t => Abstract Horizon: {len(plan["states"])}'
                      f'\n\t => Low-Level Steps: {step_count}'
                      f'\n\t => Original Plan: {plan["states"]}'
                      f'\n\t  => Executed Plan: {visited_states} \n')


    elif args.opr == 'low-level-plan':

        model = torch.load('model.p', map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()


    else:
        raise ValueError()
