from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL


def get_ppo_cartpole_cfg(env, device):
  cfg = PPO_DEFAULT_CONFIG.copy()
  cfg['rollouts'] = 1024  # Memory size
  cfg['learning_epochs'] = 10
  cfg['mini_batches'] = 32
  cfg['discount_factor'] = 0.99
  cfg['lambda'] = 0.95
  cfg['learning_rate'] = 1e-3
  cfg['learning_rate_scheduler'] = KLAdaptiveRL
  cfg['learning_rate_scheduler_kwargs'] = {'kl_threshold': 0.008}
  cfg['grad_norm_clip'] = 0.5
  cfg['ratio_clip'] = 0.2
  cfg['value_clip'] = 0.2
  cfg['clip_predicted_values'] = False
  cfg['entropy_loss_scale'] = 0.01
  cfg['value_loss_scale'] = 0.5
  cfg['kl_threshold'] = 0
  cfg['mixed_precision'] = True
  cfg['state_preprocessor'] = RunningStandardScaler
  cfg['state_preprocessor_kwargs'] = {'size': env.observation_space, 'device': device}
  cfg['value_preprocessor'] = RunningStandardScaler
  cfg['value_preprocessor_kwargs'] = {'size': 1, 'device': device}
  # Logging to TensorBoard and checkpoints
  cfg['experiment']['write_interval'] = 500
  cfg['experiment']['checkpoint_interval'] = 5000
  cfg['experiment']['directory'] = 'runs/torch/CartPole'
  return cfg

def get_dqn_cfg():
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg['learning_starts'] = (
      10_000  # Start learning after 10k steps: fill the experience replay buffer
    )
    cfg['learning_starts'] = 100
    cfg['exploration']['final_epsilon'] = 0.04
    cfg['exploration']['timesteps'] = 1500
    # tensorboard logging and checkpoints
    cfg['experiment']['write_interval'] = 1000
    cfg['experiment']['checkpoint_interval'] = 50000
    cfg['experiment']['directory'] = 'runs/torch/CartPole'
    return cfg
