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
    def __init__(self, env = None, testing = False):
        if not env: 
            self._env = gym.make('markets-execution-v0',
                   background_config='rmsc04', 
                   starting_cash = 10_000_000,
                   timestep_duration="1S",
                   order_fixed_size= 20,
                   execution_window= "00:30:00",
                   parent_order_size= 20_000,
                   debug_mode=True)
            print("Using default Abides environment (markets-execution-v0) with config: background_config='rmsc04', starting_cash = 10_000_000, timestep_duration=1S, order_fixed_size= 20, execution_window= 00:30:00, parent_order_size= 20_000, debug_mode=True")
        else:
            self._env = env
        self._env = SqueezeObsWrapper(self._env)
        self.testing = testing

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

        if not self.testing:
            # if training, return the reward and done flag
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