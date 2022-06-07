import time
import torch
import torch.nn as nn
import gym, gym_unbalanced_disk, time
import numpy as np
import matplotlib.pyplot as plt

def normalization(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

class Discretize(gym.Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretize, self).__init__(env) #sets self.env
        self.nvec = nvec

        self.action_space = gym.spaces.Discrete(self.nvec)
        self.alow, self.ahigh = env.action_space.low, env.action_space.high

    def step(self, action):
        action = self.opt_action(action)
        observation, _, done, info = self.env.step(action)
        observation[0] = normalization(observation[0])
        reward = self.get_reward(observation, action=action)

        return np.array(observation), reward, done, info    # Use minus reward here

    def reset(self):
        return np.array(self.env.reset())

    def random_action(self):
        action = self.env.action_space.sample()
        return ((action - self.alow)/(self.ahigh - self.alow)*self.nvec).astype(int)

    def opt_action(self, action):
        action = action / self.nvec *(self.ahigh - self.alow) + self.alow
        return action

    def get_reward(self, observation, action):
        theta = normalization(observation[0])
        omega = observation[1]

        alpha, beta, gamma = 100, 0.15, 0.15
        reward = alpha*theta**2 - beta*omega**2 - gamma*action**2
        return reward

class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size=40):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        #define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)  #a)
        self.critic_linear2 = nn.Linear(hidden_size, 1) #a)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size) #a)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions) #a)

    def actor(self, state, return_logp=False):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.actor_linear1(state)) #a)
        h = self.actor_linear2(hidden) #a=)
        h = h - torch.max(h,dim=1,keepdim=True)[0] #for additional numerical stability
        logp = h - torch.log(torch.sum(torch.exp(h),dim=1,keepdim=True)) #log of the softmax
        if return_logp:
            return logp
        else:
            return torch.exp(logp) #by default it will return the probability

    def critic(self, state):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.critic_linear1(state)) #a)
        return self.critic_linear2(hidden)[:,0] #a) #no activation function

    def forward(self, state):
        #state has shape (Nbatch, Nobs)
        return self.critic(state), self.actor(state)

# Randomly choose action and restore tuple in experienced pool
def rollout(actor_crit, env, N_rollout=10_000):
    #save the following (use .append)
    Start_state = [] #hold an array of (x_t)
    Actions = [] #hold an array of (u_t)
    Rewards = [] #hold an array of (r_{t+1})
    End_state = [] #hold an array of (x_{t+1})
    Terminal = [] #hold an array of (terminal_{t+1})
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        obs = env.reset()
        for i in range(N_rollout):
            action = np.random.choice(env.action_space.n,p=pi(obs)) #b=)

            Start_state.append(obs)
            Actions.append(action)

            obs_next, reward, done, info = env.step(action)
            terminal = done and not info.get('TimeLimit.truncated', False)

            Terminal.append(terminal)
            Rewards.append(reward)
            End_state.append(obs_next)

            if done:
                obs = env.reset()
            else:
                obs = obs_next

    #error checking:
    assert len(Start_state)==len(Actions)==len(Rewards)==len(End_state)==len(Terminal), f'error in lengths: {len(Start_state)}=={len(Actions)}=={len(Rewards)}=={len(End_state)}=={len(Dones)}'
    return np.array(Start_state), np.array(Actions), np.array(Rewards), np.array(End_state), np.array(Terminal).astype(int)

def eval_actor(actor_crit, env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        rewards_acc = 0
        obs = env.reset()
        while True:
            action = np.argmax(pi(obs)) #b=)
            obs, reward, done, info = env.step(action)
            rewards_acc += reward
            if done:
                return rewards_acc

def show(actor_crit,env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        try:
            obs = env.reset()
            env.render()
            time.sleep(1)
            while True:
                action = np.argmax(pi(obs)) #b=)
                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)
                time.sleep(1/60)
                env.render()
                if done:
                    time.sleep(0.5)
                    break
        finally: #this will always run even when an error occurs
            env.close()

# -------------------------------------------------------------------------
# Training function
def A2C_rollout(actor_crit, optimizer, env, alpha_actor=0.5, alpha_entropy=0.5, gamma=0.98, \
                N_iterations=21, N_rollout=20000, N_epochs=10, batch_size=32, N_evals=10):
    best = -float('inf')
    torch.save(actor_crit.state_dict(),'actor-crit-checkpoint')
    try:
        for iteration in range(N_iterations):
            print(f'rollout iteration {iteration}')

            #2. rollout
            Start_state, Actions, Rewards, End_state, Terminal = rollout(actor_crit, env, N_rollout=N_rollout)

            #Data conversion, no changes required
            convert = lambda x: [torch.tensor(xi,dtype=torch.float32) for xi in x]
            Start_state, Rewards, End_state, Terminal = convert([Start_state, Rewards, End_state, Terminal])
            Actions = Actions.astype(int)

            print('starting training on rollout information...')
            for epoch in range(N_epochs):
                for i in range(batch_size,len(Start_state)+1,batch_size):
                    Start_state_batch, Actions_batch, Rewards_batch, End_state_batch, Terminal_batch = \
                    [d[i-batch_size:i] for d in [Start_state, Actions, Rewards, End_state, Terminal]]

                    #Advantage:
                    Vnow = actor_crit.critic(Start_state_batch) #c=)
                    Vnext = actor_crit.critic(End_state_batch) #c=)
                    A = Rewards_batch + gamma*Vnext*(1-Terminal_batch) - Vnow #c=)

                    action_index = np.stack((np.arange(batch_size),Actions_batch),axis=0) #to filter actions
                    logp = actor_crit.actor(Start_state_batch,return_logp=True)[action_index] #c=)
                    p = torch.exp(logp) #c=)

                    L_value_function = torch.mean(A**2) #c=)
                    L_policy = -(A.detach()*logp).mean() #c=) #detach A, the gradient should only to through logp
                    L_entropy = -torch.mean((-p*logp),0).sum() #c=)

                    Loss = L_value_function + alpha_actor*L_policy + alpha_entropy*L_entropy #c=)

                    optimizer.zero_grad()
                    Loss.backward()
                    optimizer.step()


                score = np.mean([eval_actor(actor_crit, env) for i in range(N_evals)])

                print(f'iteration={iteration} epoch={epoch} Average Reward per episode:',score)
                print('\t Value loss:  ',L_value_function.item())
                print('\t Policy loss: ',L_policy.item())
                print('\t Entropy:     ',-L_entropy.item())

                if score>best:
                    best = score
                    print('################################# \n new best',best,'saving actor-crit... \n#################################')
                    torch.save(actor_crit.state_dict(),'actor-crit-checkpoint')

            print('loading best result')
            actor_crit.load_state_dict(torch.load('actor-crit-checkpoint'))
    finally: #this will always run even when using the a KeyBoard Interrupt.
        print('loading best result')
        actor_crit.load_state_dict(torch.load('actor-crit-checkpoint'))



def main():

    # Define environment parameters
    nvec = 10
    max_episode_steps = 1000 #c)

    env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.)
    env = gym.wrappers.time_limit.TimeLimit(env,max_episode_steps=max_episode_steps) #c)
    env = Discretize(env, nvec)

    # Define training parameters
    gamma = 0.99
    N_iterations = 5
    N_rollout = 20000
    N_epochs = 5
    N_evals = 10
    alpha_actor = 0.3
    alpha_entropy = 0.5
    lr = 5e-3


    assert isinstance(env.action_space,gym.spaces.Discrete), 'action space requires to be discrete'

    actor_crit = ActorCritic(env, hidden_size=40)
    optimizer = torch.optim.Adam(actor_crit.parameters(), lr=lr) #low learning rate
    A2C_rollout(actor_crit, optimizer, env, alpha_actor=alpha_actor, alpha_entropy=alpha_entropy,\
                             gamma=gamma, N_iterations=N_iterations, N_rollout=N_rollout, N_epochs=N_epochs, \
                             N_evals=N_evals)

    plt.plot([eval_actor(actor_crit, env) for i in range(100)],'.')
    plt.show()
    show(actor_crit,env)

if __name__ == '__main__':
    main()
