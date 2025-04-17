import gymnasium as gym
import torch
import skrl
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model

from oterl.utils.cfg_utils import get_ppo_cartpole_cfg
from oterl.models import Policy, Value

def train_agent(agent_class, env, cfg=None, timesteps=500_000, seed=0, device="cpu"):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    # Get observation and action space sizes
    observation_space_size = env.observation_space.shape[0]
    print(f"Observation space size: {observation_space_size}")
    # Get action space size
    action_space_size = env.action_space.n
    print(f"Action space size: {action_space_size}")
    # Define device
    device = torch.device(device)
    # Define models (simple feedforward network)
    models = {}

    if agent_class == DQN:
        memory = RandomMemory(memory_size=200000, num_envs=env.num_envs, device=device, replacement=False)
        cfg = DQN_DEFAULT_CONFIG.copy()
        cfg["learning_starts"] = 10_000  # Start learning after 10k steps: fill the experience replay buffer
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
        models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
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
        # initialize models' lazy modules
        for role, model in models.items():
            model.init_state_dict(role)

        # initialize models' parameters (weights and biases)
        for model in models.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

        # configure and instantiate the agent (visit its documentation to see all the options)
        # https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
        cfg = DQN_DEFAULT_CONFIG.copy()
        cfg["learning_starts"] = 100
        cfg["exploration"]["final_epsilon"] = 0.04
        cfg["exploration"]["timesteps"] = 1500
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 1000
        cfg["experiment"]["checkpoint_interval"] = 50000
        cfg["experiment"]["directory"] = "runs/torch/CartPole"
    elif agent_class == PPO:
        memory = RandomMemory(memory_size=1024, num_envs=1, device=device)
        models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
        models["value"] = Value(env.observation_space, env.action_space, device)    
        cfg = get_ppo_cartpole_cfg(env, device)
    else:
        raise ValueError("Invalid agent class")

    # Initialize agent
    agent = agent_class(models=models, 
                        memory=memory, 
                        cfg=cfg, 
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        device=device)

    # Configure and run the RL trainer
    cfg_trainer = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()

