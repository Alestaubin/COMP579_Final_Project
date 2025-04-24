import argparse
import torch
import gym
import numpy as np
import pickle
from agents.recurrent_ppo_truncated_bptt.utils import create_env
from agents.recurrent_ppo_truncated_bptt.model import ActorCriticModel
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from agents.TWAP import TWAPAgent
from skrl.utils.model_instantiators.torch import deterministic_model
from agents.baselines.skrl_models import Policy, Value
from agents.baselines.cfg_utils import get_ppo_cartpole_cfg
from agents.recurrent_ppo_truncated_bptt.environments.abides_gym import AbidesGym
from agents.baselines.trainer import load_skrl_agent
from skrl.envs.wrappers.torch import GymWrapper

# Example usage:
# uv run tester.py --model_path "src/models/2025-04-23-21-25-41_20.nn" --agent "RPPO"
# uv run tester.py --model_path "../runs/torch/CartPole/25-04-23_18-12-48-540371_PPO/checkpoints/best_agent.pt" --agent "PPO"

class AgentTester:
    def __init__(self, model_path, agent_name):
        self.model_path = model_path
        self.agent_name = agent_name.upper()
        self.env = gym.make('markets-execution-v0',
                   background_config='rmsc04', 
                   starting_cash = 10_000_000,
                   timestep_duration="1S",
                   order_fixed_size= 20,
                   execution_window= "00:30:00",
                   parent_order_size= 20_000,
                   debug_mode=True)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

        # TWAP-specific parameters (will leave as the default environment set up)
        self.total_shares = 20000 # number of shares that need to be executed by the agent
        self.execution_window_sec = 1800 # number of time steps available to our agent to execute the entire order
        self.time_discretization = 30 # TWAP executes trades at every 30 seconds

        self.load_agent()

    def load_agent(self):
        if self.agent_name == "PPO":
            self.env = wrap_env(self.env)
            self.agent = load_skrl_agent(PPO, self.model_path, self.env, self.device)
        elif self.agent_name == "DQN":
            self.env = wrap_env(self.env)
            self.agent = load_skrl_agent(DQN, self.model_path, self.env, self.device)
        elif self.agent_name == "TWAP":
            self.agent = TWAPAgent(self.total_shares, self.execution_window_sec, self.time_discretization)
        
        elif self.agent_name == "RPPO":
            # Load model and config
            state_dict, self.config = pickle.load(open(self.model_path, "rb"))
            # the RPPO agent requires a custom wrapper 
            self.env = AbidesGym(self.env, testing=True)
            self.model = ActorCriticModel(self.config, self.env.observation_space, (self.env.action_space.n,))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            # Init recurrent cell
            hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
            if self.config["recurrence"]["layer_type"] == "gru":
                self.recurrent_cell = hxs
            elif self.config["recurrence"]["layer_type"] == "lstm":
                self.recurrent_cell = (hxs, cxs)

        else:
            raise ValueError(f"Unsupported agent: {self.agent_name}")
    
    def act(self, state, time_step=None, max_steps=None):
        if self.agent_name == "RPPO":
            # Forward model
            policy, value, self.recurrent_cell = self.model(torch.tensor(np.expand_dims(state, 0)), self.recurrent_cell, self.device, 1)
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            return action
        elif self.agent_name == "TWAP":
            current_time_sec = state["current_time"][-1] / 1e9 # converting ns array to seconds
            return self.agent.get_action(current_time_sec)
        else:
            return self.agent.act(state, timestep=time_step, timesteps=max_steps)[0]

    def run_episode(self):
        print(self.env)
        if isinstance(self.env, GymWrapper):
            state, _ = self.env.reset()
        else:
            state = self.env.reset()
        print("Initial state:", state)
        done = False
        self.states = []
        self.infos = []

        max_steps = 10000
        t = 0
        while not done and t < max_steps:
            action = self.act(state, time_step=t, max_steps=max_steps)
            t += 1
            if isinstance(self.env, GymWrapper):
                state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                state, _, done, info = self.env.step(action)
            print(info)

            self.infos.append(info)
            self.states.append(state)
    
    #@TODO: Implement metric evaluation
    def evaluate_metrics(self):
        """
        Placeholder for metric evaluation.
        Args:
            data_dict (dict): Dict with keys 'states' and 'rewards'
        """
        pass  # Implement metric evaluation logic here

    def test(self):
        self.run_episode()
        self.evaluate_metrics()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained agent model")
    parser.add_argument('--agent', type=str, required=True, choices=["PPO", "DQN", "RPPO"], help="Agent type")
    args = parser.parse_args()

    tester = AgentTester(model_path=args.model_path, agent_name=args.agent)
    tester.test()


if __name__ == "__main__":
    main()