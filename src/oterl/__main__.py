import yaml
import argparse
import gym
import torch
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.dqn import DQN
from skrl.envs.wrappers.torch import wrap_env
from oterl.utils.yaml_parser import YamlParser
from oterl.agents.recurrent_ppo_truncated_bptt.trainer import PPOTrainer
from oterl.agents.baselines.trainer import train_agent
import time


def load_config(config_path):
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)



def main():
    import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print('Initializing rmsc04 gym environment...')
    env = gym.make('markets-execution-v0', 
                   background_config='rmsc04', 
                   starting_cash = 10_000_000,
                   timestep_duration="1S",
                   order_fixed_size= 20,
                   execution_window= "00:30:00",
                   parent_order_size= 20_000,
                   debug_mode=False,
    )

    seed = cfg.get("seed", 0)
    agent = cfg.get("agent")
    agent_name = agent["name"]
    cfg_path = agent["config_path"]

    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine. Please set device to 'cpu' or another supported backend.")
    elif device_str == "cpu" and torch.cuda.is_available():
        print("Warning: CUDA is available but 'cpu' is selected. Consider using 'cuda' for better performance.")
    elif device_str == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine. Please set device to 'cpu' or 'cuda'.")

    if device_str == "cuda":
        print("Using CUDA for training.")
        device = torch.device("cuda")
    elif device_str == "cpu":
        print("Using CPU for training.")
        device = torch.device("cpu")
    elif device_str == "mps":
        print("Using Metal Performance Shaders (MPS) for training.")
        device = torch.device("mps")
    else:
        raise ValueError(f"Unknown device: {device_str}. Please set device to 'cpu', 'cuda', or 'mps'.")    # Set random seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")

    # Vanilla PPO Baseline
    if agent_name == "PPO":
        timesteps = int(cfg.get("timesteps", 1e6))
        print(f"Training PPO with {timesteps} timesteps")
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(PPO, env, cfg=None, timesteps=timesteps, seed=seed)
    # DQN Baseline
    elif agent_name == "DQN":
        timesteps = int(cfg.get("timesteps", 1e6))
        print(f"Training DQN with {timesteps} timesteps")
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(DQN, env, cfg=None, timesteps=timesteps, seed=seed)
    
    # Recurrent PPO with Truncated BPTT
    elif agent_name == "RPPO":
        if not cfg_path:
            raise ValueError("Missing 'agent_config_path' in config for RPPO")
        agent_cfg = YamlParser(cfg_path).get_config()
        print(f"Training RPPO with config: \n{agent_cfg}")
        timestamp = time.time()
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(timestamp))
        trainer = PPOTrainer(agent_cfg, run_id=timestamp, device=device)
        trainer.run_training()
        trainer.close()

    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    main()
