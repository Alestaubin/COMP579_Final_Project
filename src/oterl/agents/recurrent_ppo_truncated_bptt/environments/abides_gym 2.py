import gym
import numpy as np
import time
import abides_gym

class SqueezeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        assert old_space.shape[-1] == 1, "Expected shape (n, 1)"
        self.observation_space = gym.spaces.Box(
            low=np.squeeze(old_space.low),
            high=np.squeeze(old_space.high),
            dtype=old_space.dtype
        )

    def observation(self, obs):
        return np.squeeze(obs)
    
class AbidesGym:
    """
    Overview:
        Wrapper for the Abides-gym environment.
    """
    def __init__(self):
        self._env = gym.make('markets-daily_investor-v0', background_config='rmsc04',) 
        self._env = SqueezeObsWrapper(self._env)

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()