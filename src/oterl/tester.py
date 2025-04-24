import argparse
import torch
import gym
import numpy as np
import os
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO as SKRL_PPO
from skrl.agents.torch.dqn import DQN as SKRL_DQN
import pickle
from oterl.agents.recurrent_ppo_truncated_bptt.utils import create_env
from oterl.agents.recurrent_ppo_truncated_bptt.model import ActorCriticModel
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO
from skrl.utils.model_instantiators.torch import deterministic_model
from oterl.agents.baselines.skrl_models import Policy, Value
from oterl.agents.baselines.cfg_utils import get_ppo_cartpole_cfg

# Example usage:
# uv run tester.py --model_path "../models/my_run.nn" --agent "RPPO"

class AgentTester:
    def __init__(self, model_path, agent_name):
        self.model_path = model_path
        self.agent_name = agent_name.upper()
        self.env = wrap_env(gym.make('markets-daily_investor-v0', background_config='rmsc04'))
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
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
        elif self.agent_name == "RPPO":
            # Load model and config
            state_dict, self.config = pickle.load(open(self.model_path, "rb"))
            # the RPPO agent requires a custom wrapper 
            self.env = create_env({"type": "Abides"}, render=False)
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
        else:
            return self.agent.act(state, time_step=time_step)[0]

    def run_episode(self):
        state = self.env.reset()
        done = False
        data = {"states": [], "rewards": []}
        reward = 0
        while not done:
            action = self.act(state)
            print(f"State: {state}, Action: {action}, reward: {reward}")
            state, reward, done, _ = self.env.step(action)
            data["states"].append(state)
            data["rewards"].append(reward)

        print('DATA', data)
        return data
    
    def evaluate_metrics(self, actions):
        """
        Evaluates trading performance metrics based on a list of action dictionaries.
        
        Args:
            actions (list): List of dictionaries, each containing:
                - "last_transaction": Price of the last transaction
                - "best_bid": Current best bid price
                - "best_ask": Current best ask price
                - "current_time": Current timestamp
                - "holdings": Current inventory position
                - "parent_size": Total size of the parent order
                - "pnl": Current profit and loss
                - "reward": PnL normalized by parent order size
                
        Returns:
            dict: Dictionary containing calculated performance metrics
        """
        import numpy as np
        
        if not actions:
            print("No actions provided for evaluation")
            return {}
        
        # Extract data series from the action dictionaries
        last_transactions = np.array([action.get("last_transaction", 0) for action in actions])
        best_bids = np.array([action.get("best_bid", 0) for action in actions])
        best_asks = np.array([action.get("best_ask", 0) for action in actions])
        timestamps = np.array([action.get("current_time", 0) for action in actions])
        holdings = np.array([action.get("holdings", 0) for action in actions])
        pnls = np.array([action.get("pnl", 0) for action in actions])
        rewards = np.array([action.get("reward", 0) for action in actions])
        
        # Get the parent order size
        parent_size = actions[0].get("parent_size", 1)
        
        # Calculate mid prices and spreads
        mid_prices = (best_asks + best_bids) / 2
        spreads = best_asks - best_bids
        
        # Create metrics dictionary
        metrics = {}
        
        # Performance metrics
        metrics["total_pnl"] = pnls[-1] if len(pnls) > 0 else 0
        metrics["normalized_pnl"] = metrics["total_pnl"] / parent_size if parent_size != 0 else 0
        metrics["total_reward"] = np.sum(rewards)
        metrics["mean_reward"] = np.mean(rewards)
        
        # Order execution metrics
        initial_holdings = holdings[0] if len(holdings) > 0 else 0
        final_holdings = holdings[-1] if len(holdings) > 0 else 0
        target_execution = abs(initial_holdings)  # Initial position to execute
        actual_execution = abs(initial_holdings - final_holdings)
        
        metrics["execution_percentage"] = 100 * (actual_execution / target_execution) if target_execution != 0 else 100
        metrics["remaining_inventory"] = final_holdings
        metrics["remaining_inventory_percentage"] = 100 * final_holdings / parent_size if parent_size != 0 else 0
        
        # Trading activity metrics
        if len(holdings) > 1:
            holdings_changes = np.diff(holdings)
            buys = holdings_changes[holdings_changes < 0]  # Negative changes represent buying
            sells = holdings_changes[holdings_changes > 0]  # Positive changes represent selling
            
            metrics["total_volume_traded"] = np.sum(np.abs(holdings_changes))
            metrics["num_buys"] = len(buys)
            metrics["num_sells"] = len(sells)
            metrics["num_trades"] = len(buys) + len(sells)
            metrics["avg_trade_size"] = np.mean(np.abs(holdings_changes[holdings_changes != 0])) if np.any(holdings_changes != 0) else 0
            
            # Calculate trade timing
            if len(timestamps) > 1 and len(holdings_changes) > 0:
                trade_times = timestamps[1:][holdings_changes != 0]
                if len(trade_times) > 0:
                    first_trade_time = trade_times[0]
                    last_trade_time = trade_times[-1]
                    metrics["time_to_first_trade"] = first_trade_time - timestamps[0]
                    metrics["execution_duration"] = last_trade_time - first_trade_time if len(trade_times) > 1 else 0
        else:
            metrics["total_volume_traded"] = 0
            metrics["num_buys"] = 0
            metrics["num_sells"] = 0
            metrics["num_trades"] = 0
            metrics["avg_trade_size"] = 0
        
        # Risk metrics
        if len(rewards) > 1:
            metrics["reward_volatility"] = np.std(rewards)
            metrics["sharpe_ratio"] = np.mean(rewards) / (np.std(rewards) + 1e-8)
            
            # Calculate maximum drawdown
            cumulative_pnl = np.cumsum(rewards)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            metrics["max_drawdown"] = np.max(drawdown) if len(drawdown) > 0 else 0
        else:
            metrics["reward_volatility"] = 0
            metrics["sharpe_ratio"] = 0
            metrics["max_drawdown"] = 0
        
        # Market impact and execution quality metrics
        metrics["avg_spread"] = np.mean(spreads)
        metrics["spread_percentage"] = 100 * metrics["avg_spread"] / np.mean(mid_prices) if np.mean(mid_prices) > 0 else 0
        
        if len(last_transactions) > 1:
            metrics["price_change"] = last_transactions[-1] - last_transactions[0]
            metrics["price_change_percentage"] = 100 * metrics["price_change"] / last_transactions[0] if last_transactions[0] > 0 else 0
            
            # VWAP calculation for execution quality
            if len(holdings) > 1:
                # Calculate volume-weighted average execution price
                abs_changes = np.abs(np.diff(holdings))
                trade_indices = np.where(abs_changes > 0)[0]
                
                if len(trade_indices) > 0:
                    trade_prices = mid_prices[trade_indices]
                    trade_volumes = abs_changes[trade_indices]
                    
                    if np.sum(trade_volumes) > 0:
                        vwap = np.sum(trade_prices * trade_volumes) / np.sum(trade_volumes)
                        metrics["execution_vwap"] = vwap
                        
                        # Market VWAP (simple average of mid prices as a benchmark)
                        market_vwap = np.mean(mid_prices)
                        metrics["market_vwap"] = market_vwap
                        
                        # VWAP slippage (for buy orders, lower is better)
                        # For sell orders, the interpretation would be opposite
                        metrics["vwap_slippage"] = 100 * (vwap - market_vwap) / market_vwap if market_vwap > 0 else 0
        
        # Implementation shortfall
        if len(mid_prices) > 0:
            arrival_price = mid_prices[0]
            metrics["arrival_price"] = arrival_price
            
            # Calculate average execution price from PnL and volume traded
            if metrics.get("total_volume_traded", 0) > 0 and initial_holdings != final_holdings:
                metrics["implementation_shortfall"] = (metrics["total_pnl"] / (initial_holdings - final_holdings) - arrival_price) / arrival_price if arrival_price > 0 else 0
        
        # Print summary
        print("\n===== Trading Performance Metrics =====")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        print("======================================\n")
        
        return metrics

    def test(self):
        data_dict = self.run_episode()
        self.evaluate_metrics(data_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained agent model")
    parser.add_argument('--agent', type=str, required=True, choices=["PPO", "DQN", "RPPO"], help="Agent type")
    args = parser.parse_args()

    tester = AgentTester(model_path=args.model_path, agent_name=args.agent)
    tester.test()


if __name__ == "__main__":
    main()
