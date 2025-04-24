import gymnasium as gym
import skrl
import torch
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model
from skrl.envs.wrappers.torch import GymWrapper

from oterl.agents.baselines.cfg_utils import get_ppo_cartpole_cfg, get_dqn_cfg
from oterl.agents.baselines.skrl_models import Policy, Value

def load_skrl_agent(agent_class, checkpoint_path, env, device="cpu"):
        models = {}
        if agent_class == PPO:
            models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
            models["value"] = Value(env.observation_space, env.action_space, device)
            cfg = get_ppo_cartpole_cfg(env, device)
        elif agent_class == DQN:
            models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                                    action_space=env.action_space,
                                                    device=device,
                                                    clip_actions=False,
                                                    network=[{
                                                        "name": "net",
                                                        "input": "STATES",
                                                        "layers": [64, 64],
                                                        "activations": "relu",
                                                    }],
                                                    output="ACTIONS")
            cfg = DQN_DEFAULT_CONFIG.copy()
            #cfg["exploration"] = {"noise": False}  # disable exploration for evaluation
        else:
            raise ValueError("Unsupported agent class")

        agent = agent_class(models=models,
                            memory=None,
                            cfg=cfg,
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            device=device)

        agent.load(checkpoint_path)
        return agent

def train_agent(agent_class, env, cfg=None, timesteps=500_000, seed=0, device=None):
  if not isinstance(env, GymWrapper):
    raise ValueError("env must be a GymWrapper instance")
  # Set random seed for reproducibility
  torch.manual_seed(seed)
  # Define models (simple feedforward network)
  models = {}
  if agent_class == DQN:
    memory = RandomMemory(
      memory_size=200000, num_envs=1, device=device, replacement=False
    )
    models['q_network'] = deterministic_model(
      observation_space=env.observation_space,
      action_space=env.action_space,
      device=device,
      clip_actions=False,
      network=[
        {
          'name': 'net',
          'input': 'STATES',
          'layers': [64, 64],
          'activations': 'relu',
        }
      ],
      output='ACTIONS',
    )
    models['target_q_network'] = deterministic_model(
      observation_space=env.observation_space,
      action_space=env.action_space,
      device=device,
      clip_actions=False,
      network=[
        {
          'name': 'net',
          'input': 'STATES',
          'layers': [64, 64],
          'activations': 'relu',
        }
      ],
      output='ACTIONS',
    )
    for role, model in models.items():
      model.init_state_dict(role)
    for model in models.values():
      model.init_parameters(method_name='normal_', mean=0.0, std=0.1)
    # get the DQN configuration
    cfg = get_dqn_cfg()
  elif agent_class == PPO:
    memory = RandomMemory(memory_size=1024, num_envs=1, device=device)
    models['policy'] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
    models['value'] = Value(env.observation_space, env.action_space, device)
    cfg = get_ppo_cartpole_cfg(env, device)
  else:
    raise ValueError('Invalid agent class')

  # Initialize agent
  agent = agent_class(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
  )

  # Configure and run the RL trainer
  cfg_trainer = {'timesteps': timesteps, 'headless': True}
  trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
  trainer.train()
