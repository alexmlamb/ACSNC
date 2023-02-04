
class ValueIteration:
    def __init__(self, state_space=None, env=None, model=None, num_actions=None, gamma=None, one_hot=True):
        """
        :param state_space: list of states for the mdp
        :param transition_fn: callback transition function : f(s,a) => s'
        :param reward_fn: callback reward function : f(s) => r
        :param num_actions: number of actions in mdp
        :param gamma: discount factor
        """
        
        self.state_space = state_space
        self.gamma = gamma
        self.one_hot = one_hot #whether states in the environment are one-hot encoded
        self.env=env
       
        if state_space is not None:
            self.discrete_agent_model=model 
            self.agent_model = self.discrete_agent_model.agent_model
            self.hx_vae = self.discrete_agent_model.hx_vae
            self.num_actions = num_actions
            self.state_value = {}
            self.action_map = {}
            
            self.state_possible = {}
            self.transition_space = [{} for a in range(num_actions)] #list of dictionaries
            self.reward_transition= {} #list of dictionaries
            self.terminal_map = {}
        
        
    def train(self, tolerance=1e-5, max_iter=100000, verbose=False):
        """
        :param tolerance : Tolerance for VI convergence check
        :param max_iter : Maximum iterations allowed
        :param verbose : logs progress
        """
        print("in train")
        i=0
        #self.fill_transition_table_by_sampling()
        self.fill_transition_table_by_sampling()
        while True:
            i=i+1
            delta = 0 #maximum change to the value function map at this iteration
            for state in self.state_possible:
                max_val = -np.inf
                is_terminal = self.terminal_map[tuple(state)] #is this a terminal state?
               
                #v(s) = max_a {r(s) + sum_{nextstates} [p(s'/s,a)*v(s')}]
                for action in range(self.num_actions):
                    
                    reward_s=self.reward_transition[tuple(state)]
                    
                    #print(is_terminal)
                    if not is_terminal:
                        transition_val = reward_s + self.gamma*self.state_value[self.transition_space[action][tuple(state)]]
                    else:
                        transition_val = reward_s
                        
                    if transition_val>=max_val:
                        max_val=transition_val
                        self.action_map[tuple(state)]=action
                        #max_val=max(max_val,transition_val)
                
                delta = max(delta,abs(self.state_value[tuple(state)] - max_val))
                self.state_value[tuple(state)]=max_val
    
            if(delta< tolerance):
                print('delta is {}'.format(delta))
                break
            if verbose:
                print('Iter:{} Delta:{}'.format(i, delta))
                
    def fill_transition_table_by_sampling(self):
        
        episodes = 200
        env= self.env
        set_unique = set()
        
        for ep in range(episodes):
            
            curstate = torch.FloatTensor(env.reset(random=True))
            _,_,_,_,curstate_bottleneck,_,=self.agent_model.forward(curstate,None,hx_vae=self.hx_vae)
            done = False
            
            if ep%10==0:
                print('collecting from episode {}'.format(ep))
                
            while not done :
                
                action=np.random.randint(4)
                nextstate, reward, done, info = env.step(action)
                #set_unique.add(tuple(nextstate))
                #set_unique.add(tuple(curstate))
                
                action_vec=np.zeros([self.num_actions])
                action_vec[action]=1
                action_vec=torch.FloatTensor(action_vec).reshape(-1,self.num_actions)
                
                #now, we need to get the discrete bottleneck corresponding to this nextstate
                
                _,_,_,_,nextstate_bottleneck,_=self.agent_model.forward(torch.FloatTensor(nextstate),None,hx_vae=self.hx_vae)
               
                #self.state_possible.add(tuple(nextstate_bottleneck.detach().numpy().astype('int32').tolist()))  # adding to the set of possible states. ig
                
                #if len(self.state_possible) !=len(set_unique):
                    #print('WRONG')
                #if reward==1:
                    #print('reward')
                    #print(nextstate_bottleneck)
                
                self.transition_space[action][tuple(curstate_bottleneck.detach().numpy().astype('int32').tolist())]=tuple(nextstate_bottleneck.detach().numpy().astype('int32').tolist())
                self.reward_transition[tuple(nextstate_bottleneck.detach().numpy().astype('int32').tolist())]= reward
                self.terminal_map[tuple(nextstate_bottleneck.detach().numpy().astype('int32').tolist())] = done
                
                curstate_bottleneck = nextstate_bottleneck
                curstate = nextstate
                self.state_possible[tuple(curstate_bottleneck.detach().numpy().astype('int32').tolist())]=-1  # adding to the set of possible states. ignore
                
                    #print('hi')
        for k in self.state_possible:
            self.state_value[k] = 0 #initialize all states to zero value
    def test(self, env, discrete_model, episodes, render=False, verbose=True):
        """
        :param env: instance of the environment
        :param discrete_model:
        :param episodes: No. of episodes for evaluation
        :param render: if true, renders the test episodes
        :param verbose: logs the progress
        :return: average test performance
        """
        print("in test")
        perf=[]
        for ep in range(episodes):
            
            curstate = env.reset()
            curstate = torch.FloatTensor(curstate)
            #start_state=curstate
            done = False
            ep_steps=0

            while not done:
                
                #if render:
                    #env.render()
                
                _,_,_,_,discretized,_,=self.agent_model.forward(curstate,None,hx_vae=self.hx_vae)
                
                action = self.action_map[tuple(discretized.detach().numpy().astype('int32').tolist())]
                #print(action)
                nextstate, reward, done, info = env.step(action)
                ep_steps+=1
                curstate=torch.FloatTensor(nextstate)

            if verbose:
                logger.info('Testing ==> Ep : {}, Steps for reaching goal: {}'.format(ep, ep_steps))
                
            perf.append(ep_steps)
        return np.mean(perf).astype('int32')
    
    
    def test_policy_mappings(self):
        
        test_data_size=1000
        #env_test=ContinuousGridWorld()
        
        curstate=[]
        discrete_size=np.sqrt(test_data_size)
        for i in range(int(discrete_size)):
            for j in range(int(discrete_size)):
                
                obs=np.array([i*0.035,j*0.035])
            #obs=env_test.reset(random=True)
            
#            action_rand = env_test.action_space.sample()
#            action_rand_onehot=np.zeros([4])
#            action_rand_onehot[action_rand]=1
#            #actions.append(action_rand_onehot)
                curstate.append(obs)
            
       
        curstates=torch.FloatTensor(curstate)
        #actions = torch.FloatTensor(actions)
        
        predicted_states,predicted_rewards,_,_,discretized,probs=self.agent_model.forward(curstates,None,hx_vae=self.hx_vae)
        
        policy_mappings = np.zeros([discretized.shape[0]])
        for i in range(discretized.shape[0]):
            policy_mappings[i] = self.action_map[tuple(discretized[i].detach().numpy().astype('int32').tolist())]
        
        plt.scatter(curstates[:,0].data.numpy(),curstates[:,1].data.numpy(),c=policy_mappings)
        plt.show()
       
    def visualize_value(self,env):
        
        size_maze= env.reset().shape[0]
        vals_arr=np.zeros([size_maze])
        
        for i in range(500): #make sure all states are sampled
            
            curstate=torch.FloatTensor(env.reset(random=True))
            _,_,_,_,discretized,_,=self.agent_model.forward(curstate,None,hx_vae=self.hx_vae)
            
            state_idx = torch.argmax(curstate).item() #where the index is 1
            val= self.state_value[tuple(discretized.detach().numpy().astype('int32').tolist())]
            vals_arr[state_idx]=val
            
        vals_arr=vals_arr.reshape(-1,int(np.sqrt(size_maze)))
        #print(vals_arr)
        vals_arr[-1,-1]=np.max(vals_arr) #setting goal state value manually, since we never sample during training
        plt.imshow(vals_arr)
        plt.show()

    def save(self, path):
        pickle.dump(self.__dict__, open(path, 'wb'))

    def restore(self, path):
        self.__dict__.update(pickle.load(open(path, 'rb')))
