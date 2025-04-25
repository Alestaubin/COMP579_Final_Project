import argparse
import torch
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from agents.recurrent_ppo_truncated_bptt.model import ActorCriticModel
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.dqn import DQN
from skrl.agents.torch.ppo import PPO
from agents.TWAP import TWAPAgent
from agents.recurrent_ppo_truncated_bptt.environments.abides_gym import AbidesGym
from agents.baselines.trainer import load_skrl_agent
from skrl.envs.wrappers.torch import GymWrapper
from agents.recurrent_ppo.trainer import Trainer
from agents.recurrent_ppo.config_utils import get_config
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# Example usage:
# uv run tester.py --model_path "src/models/2025-04-23-21-25-41_20.nn" --agent "RPPO"
# uv run tester.py --model_path "../runs/torch/CartPole/25-04-23_18-12-48-540371_PPO/checkpoints/best_agent.pt" --agent "PPO"


class AgentTester:
    def __init__(self, model_path, agent_name):
        self.model_path = model_path
        self.agent_name = agent_name.upper()

        self.background_config='rmsc04' 
        self.starting_cash = 10_000_000
        self.timestep_duration="5S"
        self.order_fixed_size= 20
        self.execution_window= "00:30:00"
        self.parent_order_size= 10_000
        self.debug_mode=True

        self.env = gym.make('markets-execution-v0',
                   background_config='rmsc04', 
                   starting_cash = 10_000_000,
                   timestep_duration="5S",
                   order_fixed_size= 20,
                   execution_window= "00:30:00",
                   parent_order_size= 10_000,
                   debug_mode=True)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

        # TWAP-specific parameters (will leave as the default environment set up)
        self.total_shares = 20000  # number of shares that need to be executed by the agent
        self.execution_window_sec = (
        1800  # number of time steps available to our agent to execute the entire order
        )
        self.time_discretization = 30  # TWAP executes trades at every 30 seconds

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
            self.env = AbidesGym(
                   starting_cash = 10_000_000,
                   timestep_duration="5S",
                   order_fixed_size= 20,
                   execution_window= "00:30:00",
                   parent_order_size= 10_000,
                   debug_mode=True)
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
        elif self.agent_name == "RPPO2":              
            trainer = Trainer(config=get_config(), env=self.env, writer_path=None,save_path=self.model_path)
            self.agent = trainer.agent

    def act(self, state, time_step=None, max_steps=None):
        if self.agent_name == 'RPPO':
            # Forward model
            policy, value, self.recurrent_cell = self.model(
                torch.tensor(np.expand_dims(state, 0)), self.recurrent_cell, self.device, 1
            )
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            return action
        elif self.agent_name == 'TWAP':
            current_time_sec = state['current_time'][-1] / 1e9  # converting ns array to seconds
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
            t += 1
            if self.agent_name == "RPPO2":
                state, _, done, info = self.agent.play(state, testing=True)
            else:
                action = self.act(state, time_step=t, max_steps=max_steps)
                if isinstance(self.env, GymWrapper):
                    state, _, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                else:
                    state, _, done, info = self.env.step(action)

            print(info)

            self.infos.append(info)
            self.states.append(state)
        
        print("Episode finished after {} timesteps".format(t))
    
    
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

    def plot_trading_performance(self, save_path=None):
        """
        Visualizes trading performance metrics from self.infos
        Args:
            save_path (str): Optional path to save the figure
        """
        if not hasattr(self, 'infos') or not self.infos:
            raise ValueError("No trading data available. Run an episode first.")
        
        # Convert timestamps from ns to datetime objects
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=info['current_time']/1e9) 
                    for info in self.infos]
        
        # Extract data series
        holdings = [info['holdings'] for info in self.infos]
        best_bids = [info['best_bid'] for info in self.infos]
        best_asks = [info['best_ask'] for info in self.infos]
        last_trans = [info['last_transaction'] for info in self.infos]
        pnls = [info['pnl'] for info in self.infos]
        rewards = [info['reward'] for info in self.infos]
        
        # Calculate derived metrics
        mid_prices = [(b+a)/2 for b,a in zip(best_bids, best_asks)]
        spreads = [a-b for b,a in zip(best_bids, best_asks)]
        
        # Create figure with subplots
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        
        # Plot 1: Holdings Progress
        axes[0].plot(timestamps, holdings, 'b-', linewidth=2, label='Current Holdings')
        axes[0].axhline(y=self.parent_order_size, color='r', linestyle='--', 
                    label='Target Holdings')
        axes[0].set_ylabel('Shares')
        axes[0].set_title(f'{self.agent_name} - Holdings Progress (Final: {holdings[-1]}/{self.parent_order_size})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Price Movement and Execution
        axes[1].plot(timestamps, mid_prices, 'g-', label='Mid Price')
        axes[1].plot(timestamps, best_bids, 'b--', label='Best Bid')
        axes[1].plot(timestamps, best_asks, 'r--', label='Best Ask')
        axes[1].plot(timestamps, last_trans, 'k:', label='Last Trade')
        axes[1].set_ylabel('Price')
        axes[1].set_title('Market Prices and Execution')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Spread Analysis
        axes[2].plot(timestamps, spreads, 'm-', label='Spread')
        axes[2].axhline(y=np.mean(spreads), color='k', linestyle='--', 
                    label=f'Avg Spread: {np.mean(spreads):.2f}')
        axes[2].set_ylabel('Spread')
        axes[2].set_title('Bid-Ask Spread')
        axes[2].legend()
        axes[2].grid(True)
        
        # Plot 4: PnL and Reward
        axes[3].plot(timestamps, pnls, 'c-', label='Cumulative PnL')
        axes[3].plot(timestamps, rewards, 'y-', label='Step Reward')
        axes[3].set_ylabel('Value')
        axes[3].set_title(f'Final PnL: {pnls[-1]:.2f} | Avg Reward: {np.mean(rewards):.4f}')
        axes[3].legend()
        axes[3].grid(True)
        
        # Plot 5: Execution Rate
        if len(timestamps) > 1:
            time_elapsed = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            completion_pct = [abs(h)/self.parent_order_size*100 for h in holdings]
            axes[4].plot(timestamps, completion_pct, 'purple', label='Completion %')
            axes[4].plot(timestamps, [t/max(time_elapsed)*100 for t in time_elapsed], 
                        'gray', linestyle='--', label='Time Progress')
            axes[4].set_ylabel('Percentage')
            axes[4].set_title('Execution Progress vs Time')
            axes[4].legend()
            axes[4].grid(True)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance plot to {save_path}")
        else:
            plt.show() 


    def test(self):
        # self.run_episode()
        # self.evaluate_metrics()

        # evaluating metrics of the agent and printing plots
        self.run_episode()
        metrics = self.evaluate_metrics(self.infos)
        self.plot_trading_performance()
        
        # Print key metrics
        print("\n=== Execution Summary ===")
        print(f"Completion: {metrics.get('execution_percentage', 0):.1f}%")
        print(f"Total PnL: {metrics.get('total_pnl', 0):.2f}")
        print(f"VWAP Slippage: {metrics.get('vwap_slippage', 0):.4f}%")
        print(f"Avg Spread: {metrics.get('avg_spread', 0):.2f}")
        print(f"Execution Duration: {metrics.get('execution_duration', 0)/1e9:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained agent model")
    parser.add_argument('--agent', type=str, required=True, choices=["PPO", "DQN", "RPPO", "RPPO2"], help="Agent type")
    args = parser.parse_args()

    tester = AgentTester(model_path=args.model_path, agent_name=args.agent)
    tester.test()


if __name__ == '__main__':
  main()
