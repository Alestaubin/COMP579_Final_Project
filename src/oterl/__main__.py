import abides_gym
import gym
from oterl.utils.cfg_utils import get_ppo_cartpole_cfg
from oterl.models import Policy, Value
from oterl.training.trainer import train_agent
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env

def main():
  print('Initializing rmsc04 gym environment...')

  env = gym.make(
    'markets-daily_investor-v0',
    background_config='rmsc04',
  )
  env = wrap_env(env)
  seed = 0 

  env.reset()
  env.seed(seed)

  for timesteps in [500_000, 1_000_000, 2_000_000]:  # Different training durations
    print(f"Training PPO with {timesteps} timesteps and seed {seed}")
    train_agent(PPO, env, cfg=None, timesteps=timesteps, seed=seed)
