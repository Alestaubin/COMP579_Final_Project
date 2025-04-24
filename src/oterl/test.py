


import argparse
import torch
import gym
import numpy as np
import os
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO as SKRL_PPO
from skrl.agents.torch.dqn import DQN as SKRL_DQN
import pickle
from oterl.agents.recurrent_ppo_truncated_bptt.utils import create_env
from oterl.agents.recurrent_ppo_truncated_bptt.model import ActorCriticModel
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from oterl.agents.TWAP import TWAPAgent
from skrl.utils.model_instantiators.torch import deterministic_model
from oterl.agents.baselines.skrl_models import Policy, Value
from oterl.agents.baselines.cfg_utils import get_ppo_cartpole_cfg

env = gym.make('markets-execution-v0',
            background_config='rmsc04', 
            starting_cash = 10_000_000,
            timestep_duration="1S",
            order_fixed_size= 20,
            execution_window= "00:30:00",
            parent_order_size= 20_000,
            debug_mode=True)
action = 1
done = False
state = env.reset()
print(f"Initial State: {state}")

while not done:
    state, reward, done, info = env.step(action)
    print(f"State: {state}, Action: {action}, reward: {reward}")
