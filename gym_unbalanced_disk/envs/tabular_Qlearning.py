#%%
import pickle
import gym, gym_unbalanced_disk, time, pickle
from gym import spaces
import numpy as np
from collections import defaultdict

#%%
def argmax(a):
    #random argmax
    a = np.array(a)
    return np.random.choice(np.arange(len(a),dtype=int)[a==np.max(a)])

def roll_mean(ar,start=2000,N=50): #smoothing if needed
    s = 1-1/N
    k = start
    out = np.zeros(ar.shape)
    for i,a in enumerate(ar):
        k = s*k + (1-s)*a
        out[i] = k
    return out

def normalization(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

# basic Qlearn
def Qlearn(env, nsteps=5000, alpha=0.2,eps=0.2, gamma=0.99):
    Qmat = defaultdict(float) #any new argument set to zero
    env_time = env
    while not isinstance(env_time,gym.wrappers.time_limit.TimeLimit):
        env_time = env_time.env
    ep_lengths = []
    ep_lengths_steps = []
    
    obs = env.reset()
    print('goal reached time:')
    for z in range(nsteps):

        if np.random.uniform()<eps:
            action = env.action_space.sample()
        else:
            action = argmax([Qmat[obs,i] for i in range(env.action_space.n)])

        obs_new, reward, done, info = env.step(action)

        if done and not info.get('TimeLimit.truncated', False): #terminal state = done and not by timeout
            #saving results:
            # print(env_time._elapsed_steps, end=' ')
            ep_lengths.append(env_time._elapsed_steps)
            ep_lengths_steps.append(z)
            
            #updating Qmat:
            A = reward - Qmat[obs,action] # adventage or TD
            Qmat[obs,action] += alpha*A
            obs = env.reset()
        else: #done by timeout or not done
            A = reward + gamma*max(Qmat[obs_new, action_next] for action_next in range(env.action_space.n)) - Qmat[obs,action]
            Qmat[obs,action] += alpha*A
            obs = obs_new
            
            if info.get('TimeLimit.truncated',False): #done by timeout
                #saving results:
                ep_lengths.append(env_time._elapsed_steps)
                ep_lengths_steps.append(z)
                print(z)
                print('out', end=' ')
                
                #reset:
                obs = env.reset()
    
    return Qmat, np.array(ep_lengths_steps), np.array(ep_lengths)

# state discretize
class Discretize(gym.Wrapper):
    def __init__(self, env, nvec_obs=10, nvec_act=10):
        super(Discretize, self).__init__(env) #sets self.env
        if isinstance(nvec_obs,int): #nvec in each dimention
            self.nvec_obs = [nvec_obs]*np.prod(env.observation_space.shape,dtype=int)
        else:
            self.nvec_obs = nvec_obs
        self.nvec_obs = np.array(nvec_obs) #(Nobs,) array
        self.observation_space = gym.spaces.MultiDiscrete(self.nvec_obs) # theta, omega

        # Limit theta to [-pi,pi], omega to [-40,40]
        olow = np.array([-np.pi, env.observation_space.low[1]]) 
        ohigh = np.array([np.pi, env.observation_space.high[1]])
        self.olow, self.ohigh  = olow, ohigh

        self.nvec_act = nvec_act
        self.action_space = gym.spaces.Discrete(self.nvec_act) 
        self.alow, self.ahigh = env.action_space.low, env.action_space.high

    def discretize(self,observation):
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec_obs).astype(int)) #b)

    def step(self, action):
        action = self.opt_action(action)
        observation, reward, done, info = self.env.step(action) #b)
        reward = self.get_reward(observation, action)
        observation = self.discretize(observation)

        return observation, reward, done, info 

    def reset(self):
        return self.discretize(self.env.reset()) #b)

    def random_action(self):
        action = self.env.action_space.sample()
        return ((action - self.alow)/(self.ahigh - self.alow)*self.nvec_act).astype(int) # discretize action

    def opt_action(self, action):
        action = action / self.nvec_act *(self.ahigh - self.alow) + self.alow
        return action

    def get_reward(self, observation, action):

        alpha = 15
        beta  = 0.1
        gamma = 0.01
        theta = normalization(observation[0])
        reward = np.pi - np.abs(theta) 
        # reward = observation[0]**2
        return - alpha*reward**2 - beta*observation[1]**2 - gamma*action**2


# Start training and use different discrete observation space
def train_Qmat(env):

    Qmat, _, _ = Qlearn(env, nsteps=130000, alpha=0.2, eps=0.2, gamma=0.99)

    # save model
    # with open('model/Qmat_opt_tabular','wb') as Q_function:
    #    pickle.dump(Qmat,Q_function)

    return Qmat

#%%
if __name__ == '__main__':
    nvec_act = 10
    nvec_obs = 36
    max_episode_steps = 1000

    env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.)
    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps) #c)
    env = Discretize(env,nvec_obs=nvec_obs, nvec_act=nvec_act) #c)

    Qmat = train_Qmat(env)

    try:
        obs = env.reset()
        for i in range(1000):
            time.sleep(1/48)
            action = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
            obs, reward, done, info = env.step(action)
            env.render()
            print(f'obs = {obs}, reward = {reward}')
    finally:
        env.close()

# %%
