#%%
from UnbalancedDisk import *
import gym, time
import numpy as np
from matplotlib import pyplot as plt
import pickle

Log_space = False

#%%
def argmax(a):
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])

def roll_mean(ar,start=400,N=25): #smoothing if needed
    s = 1-1/N
    k = start
    out = np.zeros(ar.shape)
    for i,a in enumerate(ar):
        k = s*k + (1-s)*a
        out[i] = k
    return out

def make_radial_basis_network(env,nvec,scale):
    # env: is the given enviroment
    # nvec: is the given number of grid points in each dimention.
    # scale: is the sigma_c in the equation
    if isinstance(nvec,int):
        nvec = [nvec]*env.observation_space.shape[0]
    
    # This creates a grid of points c_i the lower bound to the upper bound with nvec number of samples in each dimention
    low, high = env.observation_space.low, env.observation_space.high # get upper and lower bound
    Xvec = [np.linspace(l,h,num=ni) for l,h,ni in zip(low,high,nvec)] # calculate the linspace in all directions
    c_points = np.array(np.meshgrid(*Xvec)) # meshgrid all the linspaces together (Nx, X1, X2, X3, ...) 
    c_points = np.moveaxis(c_points, 0, -1) #transform to (X1, X2, X3, ..., Nobs) 
    c_points = c_points.reshape((-1,c_points.shape[-1])) #flatten into the size (Nc, Nobs)
    dx = np.array([X[1]-X[0] for X in Xvec]) # spacing (related to the B matrix)
    
    def basis_fun(obs):
        #this function should return the vector containing all phi_i of all c_points
        obs = np.array(obs) #(Nobs)
        
        dis = (c_points-obs[None,:])/dx[None,:] #dim = (Nbasis, Nobs)
        exp_arg = np.sum(dis**2,axis=1)/(2*scale**2) #squared distance to every point #b)
        Z = -exp_arg+np.min(exp_arg) #b) for numerical stability you can add the minimum.
        R = np.exp(Z) #b)
        return R/np.sum(R) #b)
    
    return basis_fun #returns a basis function

def Qlearn(env, basis_fun, epsilon=0.1, alpha=0.1, gamma=0.99, nsteps=100_000, verbose=True):
    #theta = (Nbasis, Na)
    #basis_fun(state) -> (Nbasis)
    #Q(s,.) = basis_fun(state)@theta
    env_time = env
    while not isinstance(env_time,gym.wrappers.time_limit.TimeLimit):
        env_time = env_time.env
    ep_length = []
    ep_length_id = []
    
    
    obs = env.reset() #d=)
    #init theta:
    Nbasis = basis_fun(obs).shape[0] #d=)
    theta = np.zeros((Nbasis, env.action_space.n))#d=)

    
    Q = lambda s: basis_fun(s)@theta #short-hand such that you can call Q(obs)
    
    for z in range(nsteps):
        if np.random.random()<epsilon: #d)
            u = env.action_space.sample() #d)
        else: #d)
            u = argmax(Q(obs)) #d)
        
        obs_next, reward, done, info = env.step(u) #d=)
        terminal = done and not info.get('TimeLimit.truncated', False) #terminial state
        
        if terminal:
            TD = Q(obs)[u] - reward #d)
        else:
            TD = Q(obs)[u] - (reward + gamma*np.max(Q(obs_next)))#d)
        
        #update theta
        theta[:,u] -= alpha*TD*basis_fun(obs)#d)
        
        if done:
            if verbose: #print result only when verbose is set to True
                print(env_time._elapsed_steps, end=' ') 
            ep_length.append(env_time._elapsed_steps)#time-keeping
            ep_length_id.append(z)
            
            obs = env.reset() #d=)
        else:
            obs = obs_next #d=)

    return theta, np.array(ep_length_id), np.array(ep_length)

# discretize action
class Discretize(gym.Wrapper):
    def __init__(self, env, nvec = 20):
        super(Discretize, self).__init__(env) #sets self.env
        self.nvec = nvec
        self.action_space = gym.spaces.Discrete(self.nvec)
        self.alow, self.ahigh = env.action_space.low, env.action_space.high

    def step(self, action):
        action = self.opt_action(action)
        return self.env.step(action) #b)

    def reset(self):
        return self.env.reset()

    def random_action(self):
        action = self.env.action_space.sample()
        return ((action - self.alow)/(self.ahigh - self.alow)*self.nvec).astype(int) # discretize action

    def opt_action(self, action):
        action = action / self.nvec *(self.ahigh - self.alow) + self.alow
        return action

def visualize_theta(env, theta, basis_fun):
    # for a given enviroment, theta matrix (Nbasis, Naction) and basis_fun(obs) -> (Nbasis,) 
    # it visualizes the max Q value in state-space.
    low, high = env.observation_space.low, env.observation_space.high
    nvec = [100,120]
    Xvec = [np.linspace(l,h,num=ni) for l,h,ni in zip(low,high,nvec)] # calculate the linspace in all directions
    c_points = np.array(np.meshgrid(*Xvec)) # meshgrid all the linspaces together (Nx, X1, X2, X3, ...) 
    c_points = np.moveaxis(c_points, 0, -1) #transform to (X1, X2, X3, ..., Nobs) 
    c_points = c_points.reshape((-1,c_points.shape[-1])) #flatten into the size (Nc, Nobs)
    maxtheta = np.array([np.max(basis_fun(ci)@theta) for ci in c_points]).reshape((nvec[1],nvec[0]))
    
    plt.contour(Xvec[0],Xvec[1],maxtheta)
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.colorbar()
    plt.show()

def train_theta(env,nvec,scale,alpha):
    basis_fun = make_radial_basis_network(env,nvec,scale)
    theta, _, _ = Qlearn(env, basis_fun, epsilon=0.2, alpha = alpha, nsteps=100_000)

    # save model
    # with open('model/theta_opt_basis', 'wb') as theta_opt:  
    #    pickle.dump(theta, theta_opt)

    return basis_fun, theta

#%%
# theta normalization: pi = - pi
def normalization(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

class UnbalancedDisk_limit(UnbalancedDisk):
    """limit theta to [-pi,pi]"""
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_limit, self).__init__(umax=umax, dt=dt)
        low = [-np.pi,-40.]
        high = [np.pi,40.]
        weight_omega, weight_action = 0.1, 1e-3
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))
        self.reward_fun = lambda self: (-15*(np.pi-np.abs(normalization(self.th)))**2 - weight_omega *self.omega **2 - weight_action*self.u**2).item()

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([normalization(self.th_noise), self.omega_noise]) #change anything here

#%%
if __name__ == '__main__':
    max_episode_steps = 500
    scale = 0.5
    alpha = 0.2
    nvec = 18

    env = UnbalancedDisk_limit(dt=0.025, umax=3.)
    env = gym.wrappers.time_limit.TimeLimit(env,max_episode_steps)
    env = Discretize(env,24)

    basis_fun, theta = train_theta(env,nvec,scale,alpha)

    with open('model/theta_opt_best','wb') as theta_opt:
        pickle.dump(theta,theta_opt)

    # with open('model/theta_opt_test', 'rb') as theta_opt:  
    #    theta = pickle.load(theta_opt)

    Qfun = lambda s: basis_fun(s)@theta

    try:
        obs = env.reset() #b)
        env.render() #b)
        for i in range(1000): #b)
            time.sleep(1/24)
            action = argmax(Qfun(obs)) #b)
            obs, reward, done, info = env.step(action) #b)
            env.render() #b)
    finally: #this will always run even when an error occurs
        env.close()
    


# %%
