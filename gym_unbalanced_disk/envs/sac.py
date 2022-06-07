import gym, gym_unbalanced_disk, time
import numpy as np
from stable_baselines3 import PPO, A2C, SAC

class normalized_angle_disk(gym_unbalanced_disk.UnbalancedDisk):
    def __init__(self, umax=3., dt=0.025):
        super(normalized_angle_disk, self).__init__(umax=umax, dt=dt)
        low = [-np.pi, -40.]
        high = [np.pi, 40.]
        self.observation_space = gym.spaces.Box(low=np.array(
            low, dtype=np.float32), high=np.array(high, dtype=np.float32), shape=(2,))
        self.reward_fun = lambda self: (-15*angle_normalize(np.pi-angle_normalize(self.th))**2 - 0.1 * self.omega ** 2 - 0.001*self.u**2).item()
        #self.reward_fun = lambda self: (-15*(np.pi-np.abs(angle_normalize(self.th)))**2 - 0.1 * self.omega ** 2 - 0.001*self.u**2).item()


    def get_obs(self):
        self.th_noise = self.th + \
            np.random.normal(loc=0, scale=0.001)  # do not edit
        self.omega_noise = self.omega + \
            np.random.normal(loc=0, scale=0.001)  # do not edit
        # change anything here
        return np.array([angle_normalize(self.th_noise), self.omega_noise])


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


max_episode_steps = 500
env = normalized_angle_disk(dt=0.025, umax=3.)
env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps)

model = SAC("MlpPolicy", env, learning_rate=0.01, verbose=1)
model.learn(total_timesteps=10000)

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