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
<<<<<<< HEAD
  print('Initializing rmsc04 gym environment...')
  config_path = '/Users/alexst-aubin/Library/CloudStorage/GoogleDrive-alex.staubin2@gmail.com/My Drive/McGill/McGill_Winter25/COMP579/FinalProject/code/src/oterl/config.yaml'
  cfg = load_config(config_path)
               
  env = gym.make(
    'markets-daily_investor-v0',
    background_config='rmsc04',
  )
  seed = cfg.get('seed', 0)

  for timesteps in cfg["timesteps"]:
      for agent in cfg["agents"]:
          print(f"Training {agent} with {timesteps} timesteps and seed {cfg.get('seed', 0)}")
          if agent == "PPO":
            # The SKRL library requires the environment to be wrapped in a specific way
            env = wrap_env(env)
            env.reset()
            env.seed(seed)
            train_agent(PPO, env, cfg=None, timesteps=timesteps, seed=cfg.get("seed", 0))
          elif agent == "DQN":
            # The SKRL library requires the environment to be wrapped in a specific way
            env = wrap_env(env)
            env.reset()
            env.seed(seed)
            train_agent(DQN, env, cfg=None, timesteps=timesteps, seed=cfg.get("seed", 0))
          elif agent == "RPPO":
            max_eps_length = 1000

            config = {
                "PPO":{
                    "critic_coef": 1,
                    "policy_kl_range":0.0008,
                    "policy_params": 20,
                    "gamma":0.998,
                    "gae_lambda":0.95,
                    "value_clip": 0.2,

                },
                "LSTM":{
                    "max_eps_length":max_eps_length + 50,
                    "seq_length":-1,
                    "hidden_size":64,
                    "embed_size": 64,
                },
                "entropy_coef":{
                    "start": 0.01,
                    "end": 0,
                    "step": 100_000
                },
                "lr":1e-3,
                "num_epochs": 3,
                "num_game_per_batch":1,
                "max_grad_norm": 0.5,
                "n_mini_batch": 4,
                "rewards": [0,1,0], # [lose,win,not_done]
                "set_detect_anomaly": True,
                "normalize_advantage": True,
            }
            writer_path = "runs"
            save_path = "models"
            trainer = Trainer(config=config,env=env,writer_path = writer_path,save_path=save_path)
            trainer.train()
          else:
            print(f"Unknown agent: {agent}")
            break
=======
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
>>>>>>> jonathan
