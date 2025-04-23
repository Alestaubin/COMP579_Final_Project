import yaml
import argparse
import gym
import torch
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.dqn import DQN
from skrl.envs.wrappers.torch import wrap_env
from oterl.utils.parser import YamlParser
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
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")

    print(f"\nTraining {agent} with seed {seed}")
    # Vanilla PPO Baseline
    if agent == "PPO":
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(PPO, env, cfg=None, timesteps=None, seed=seed)
    # DQN Baseline
    elif agent == "DQN":
        env = wrap_env(env)
        env.reset()
        env.seed(seed)
        train_agent(DQN, env, cfg=None, timesteps=None, seed=seed)
    
    # Recurrent PPO with Truncated BPTT
    elif agent == "RPPO":
        agent_cfg_path = cfg.get("agent_config_path")
        if not agent_cfg_path:
            raise ValueError("Missing 'agent_config_path' in config for RPPO")
        agent_cfg = YamlParser(agent_cfg_path).get_config()
        trainer = PPOTrainer(agent_cfg, run_id="rppo_run", device=device)
        trainer.run_training()
        trainer.close()

    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    main()
