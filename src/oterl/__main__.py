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
    env = gym.make('markets-daily_investor-v0', background_config='rmsc04')

    seed = cfg.get("seed", 0)
    agent = cfg.get("agent")
    agent_name = agent["name"]
    cfg_path = agent["config_path"]
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")

    print(f"\nTraining {agent} with seed {seed}")
    # Vanilla PPO Baseline
    if agent_name == "PPO":
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(PPO, env, cfg=None, timesteps=None, seed=seed)
    # DQN Baseline
    elif agent_name == "DQN":
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(DQN, env, cfg=None, timesteps=None, seed=seed)
    
    # Recurrent PPO with Truncated BPTT
    elif agent_name == "RPPO":
        if not cfg_path:
            raise ValueError("Missing 'agent_config_path' in config for RPPO")
        agent_cfg = YamlParser(cfg_path).get_config()
        trainer = PPOTrainer(agent_cfg, run_id="rppo_run", device=device)
        trainer.run_training()
        trainer.close()

    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    main()
