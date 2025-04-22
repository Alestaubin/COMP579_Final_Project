import torch
import torch.nn.functional as F
import numpy as np
from oterl.agents.recurrent_ppo.rolloutBuffer import RolloutBuffer
from oterl.agents.recurrent_ppo.distribution import Distribution
from oterl.agents.recurrent_ppo.stat import RunningMeanStd
from tqdm import tqdm

class Agent():
    """Agent class representing an AI agent"""
    def __init__(self, env, model, config):
        """
        Overview:
            Initializes the Agent instance.

        Arguments:
            - env: (`object`): The environment object.
            - model: (`object`): The model object.
            - config: (`dict`): Configuration settings for the agent.
        """
        super().__init__()
        self.env            = env
        self.observation_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n
        
        self.model          = model
        self.reward         = config["rewards"]
        self.max_eps_length = config["LSTM"]["max_eps_length"]
        self.p_state        = torch.zeros(1, 1, config["LSTM"]["hidden_size"])
        self.v_state        = torch.zeros(1, 1, config["LSTM"]["hidden_size"])
        self.hidden_size    = config["LSTM"]["hidden_size"]
        self.rollout        = RolloutBuffer(config, self.observation_space_size, self.action_space_size)
        self.distribution   = Distribution()
        self.rms            = RunningMeanStd(shape=(self.observation_space_size))

    def reset_hidden(self):
        """
        Overview:
            Resets the hidden state and candidate state.
        """
        self.p_state = torch.zeros(1, 1, self.hidden_size)
        self.v_state = torch.zeros(1, 1, self.hidden_size)

    @torch.no_grad()
    def play(self, state, testing=False):
        """
        Overview:
            Agent's play function.

        Arguments:
            - state: (`np.array`): The current state.
            - per: (`List`): The per file.

        Returns:
            - action: (`int`): The agent's chosen action.
            - per: (`List`): The per file.
        """
        self.model.eval()
        tensor_state        = self.rms.normalize((torch.tensor(state, dtype=torch.float32)))

        policy, value, p_state, v_state = self.model(tensor_state.reshape(1, -1), self.p_state, self.v_state)
        tensor_state        = tensor_state.view(-1)
        policy              = policy.squeeze()

        action_mask = torch.ones(self.action_space_size, dtype=torch.float32)
        action, log_prob    = self.distribution.sample_action(policy, action_mask)

        action = int(action) # convert to int, else step won't work
        # print("action:", action)
        obs, reward, done, info = self.env.step(action)
        # print("obs:", obs, "reward:", reward, "done:", done)

        if not testing:
            if not done:  # game not over
                if self.rollout.step_count < self.max_eps_length:
                    self.rollout.add_data(
                        state        = tensor_state,
                        p_state      = self.p_state.squeeze(),
                        v_state      = self.v_state.squeeze(),
                        action       = action,
                        value        = value.item(),
                        reward       = reward * 1.0,  # use the raw reward from step()
                        done         = 0,
                        valid_action = torch.ones(self.action_space_size, dtype=torch.float32),
                        prob         = log_prob,
                        policy       = policy
                    )
                self.rollout.step_count += 1
            else:  # game over
                if self.rollout.step_count < self.max_eps_length:
                    self.rollout.add_data(
                        state        = tensor_state,
                        p_state      = self.p_state.squeeze(),
                        v_state      = self.v_state.squeeze(),
                        action       = action,
                        value        = value.item(),
                        reward       = reward * 1.0,
                        done         = 1,
                        valid_action = torch.ones(self.action_space_size, dtype=torch.float32),
                        prob         = log_prob,
                        policy       = policy
                    )

                self.rollout.batch["dones_indices"][self.rollout.game_count] = self.rollout.step_count
                self.rollout.game_count += 1
                self.rollout.step_count = 0

        self.p_state, self.v_state = p_state, v_state

        if done:
            self.reset_hidden()
        
        return obs, reward, done, info 
    
    def run(self, num_games: int) -> float:
        """
        Overview:
            Runs the custom environment and returns the win rate.

        Arguments:
            - num_games: (`int`): The number of games to run.

        Returns:
            - win_rate: (`float`): The win rate of the agent.
        """
        for i in tqdm(range(num_games), desc="Simulating games"):
            # breakpoint()
            obs = self.env.reset()
            done = False
            while not done:
                obs, reward, done, info= self.play(obs)

        self.rms.update()
        
        win_rate = 1
        return win_rate