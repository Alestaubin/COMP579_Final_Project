import argparse
import torch
import gym
import numpy as np
import os
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO as SKRL_PPO
from skrl.agents.torch.dqn import DQN as SKRL_DQN
import pickle
from agents.recurrent_ppo_truncated_bptt.utils import create_env
from agents.recurrent_ppo_truncated_bptt.model import ActorCriticModel
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from agents.TWAP import TWAPAgent
from skrl.utils.model_instantiators.torch import deterministic_model
<<<<<<< HEAD
from oterl.agents.baselines.skrl_models import Policy, Value
from oterl.agents.baselines.cfg_utils import get_ppo_cartpole_cfg
from oterl.agents.recurrent_ppo_truncated_bptt.environments.abides_gym import AbidesGym
=======
from agents.baselines.skrl_models import Policy, Value
from agents.baselines.cfg_utils import get_ppo_cartpole_cfg
>>>>>>> refs/remotes/origin/AddMetrics

# Example usage:
# uv run tester.py --model_path "../models/my_run.nn" --agent "RPPO"

class AgentTester:
    def __init__(self, model_path, agent_name):
        self.model_path = model_path
        self.agent_name = agent_name.upper()
        breakpoint()
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

    def load_skrl_agent(self, agent_class, checkpoint_path, env, device="cpu"):
        models = {}
        if agent_class == PPO:
            self.env = wrap_env(env)
            models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
            models["value"] = Value(env.observation_space, env.action_space, device)
            cfg = get_ppo_cartpole_cfg(env, device)
        elif agent_class == DQN:
            self.env = wrap_env(env)
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


    def load_agent(self):
        if self.agent_name == "PPO":
            self.agent = self.load_skrl_agent(PPO, self.model_path, self.env, self.device)
        elif self.agent_name == "DQN":
            self.agent = self.load_skrl_agent(DQN, self.model_path, self.env, self.device)
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
    
    def act(self, state, time_step=None):
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
            return self.agent.act(state, time_step=time_step)[0]

    def run_episode(self):
        print(self.env)
        state = self.env.reset()
        done = False
        self.states = []
        self.infos = []
        reward = 0
        while not done:
            action = self.act(state)
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