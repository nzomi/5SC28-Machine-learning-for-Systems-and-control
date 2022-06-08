import gym, gym_unbalanced_disk, time
import numpy as np
from stable_baselines3 import PPO, A2C, SAC
from UnbalancedDisk import *

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



max_episode_steps = 500
env = UnbalancedDisk_limit(dt=0.025, umax=3.)
env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps)

model = SAC.load('SAC_12000.zip')
#model = SAC("MlpPolicy", env, learning_rate=0.01, verbose=1)
#model.learn(total_timesteps=12000)
#model.save('SAC_12000')


obs = env.reset()
try:
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1/50)
        if done:
            env.reset()
finally:
    env.close()