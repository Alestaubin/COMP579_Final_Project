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
    self.env = gym.make(
      'markets-execution-v0',
      background_config='rmsc04',
      starting_cash=10_000_000,
      timestep_duration='1S',
      order_fixed_size=20,
      execution_window='00:30:00',
      parent_order_size=20_000,
      debug_mode=True,
    )

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
    if self.agent_name == 'PPO':
      self.env = wrap_env(self.env)
      self.agent = load_skrl_agent(PPO, self.model_path, self.env, self.device)
    elif self.agent_name == 'DQN':
      self.env = wrap_env(self.env)
      self.agent = load_skrl_agent(DQN, self.model_path, self.env, self.device)
    elif self.agent_name == 'TWAP':
      self.agent = TWAPAgent(self.total_shares, self.execution_window_sec, self.time_discretization)

    elif self.agent_name == 'RPPO':
      # Load model and config
      state_dict, self.config = pickle.load(open(self.model_path, 'rb'))
      # the RPPO agent requires a custom wrapper
      self.env = AbidesGym(self.env, testing=True)
      self.model = ActorCriticModel(
        self.config, self.env.observation_space, (self.env.action_space.n,)
      )
      self.model.load_state_dict(state_dict)
      self.model.to(self.device)
      self.model.eval()
      # Init recurrent cell
      hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
      if self.config['recurrence']['layer_type'] == 'gru':
        self.recurrent_cell = hxs
      elif self.config['recurrence']['layer_type'] == 'lstm':
        self.recurrent_cell = (hxs, cxs)

    else:
      raise ValueError(f'Unsupported agent: {self.agent_name}')

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
    print('Initial state:', state)
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

  def evaluate_metrics(self):
    last_transactions = np.array([action.get('last_transaction', 0) for action in self.infos])
    best_bids = np.array([action.get('best_bid', 0) for action in self.infos])
    best_asks = np.array([action.get('best_ask', 0) for action in self.infos])
    timestamps = np.array([action.get('current_time', 0) for action in self.infos])
    holdings = np.array([action.get('holdings', 0) for action in self.infos])
    pnls = np.array([action.get('pnl', 0) for action in self.infos])
    rewards = np.array([action.get('reward', 0) for action in self.infos])

    parent_size = self.infos[0].get('parent_size', 1)

    mid_prices = (best_asks + best_bids) / 2
    spreads = best_asks - best_bids

    metrics = {}

    metrics['total_pnl'] = pnls[-1] if len(pnls) > 0 else 0
    metrics['normalized_pnl'] = metrics['total_pnl'] / parent_size if parent_size != 0 else 0

    if len(pnls) > 1:
      # Calculate period returns based on PnL changes
      pnl_changes = np.diff(pnls)

      # Use absolute initial capital to normalize returns
      initial_capital = 10_000_000  # From initialization, or extract from infos
      returns = pnl_changes / initial_capital

      # Calculate Sharpe ratio (assuming daily returns, annualize by sqrt(252))
      # Standard annualization factor for daily returns is sqrt(252)
      sharpe_annualization_factor = np.sqrt(252)  # For daily returns

      if np.std(returns) > 0:
        metrics['sharpe_ratio'] = (np.mean(returns) * sharpe_annualization_factor) / (
          np.std(returns) * sharpe_annualization_factor
        )
      else:
        metrics['sharpe_ratio'] = 0
    else:
      metrics['sharpe_ratio'] = 0

    # Order execution metrics
    initial_holdings = holdings[0] if len(holdings) > 0 else 0
    final_holdings = holdings[-1] if len(holdings) > 0 else 0
    target_execution = abs(initial_holdings)  # Initial position to execute
    actual_execution = abs(initial_holdings - final_holdings)

    is_buying = initial_holdings < 0  
    is_selling = initial_holdings > 0 

    metrics['execution_percentage'] = (
      100 * (actual_execution / target_execution) if target_execution != 0 else 100
    )

    metrics['remaining_inventory'] = final_holdings

    metrics['remaining_inventory_percentage'] = (
      100 * final_holdings / parent_size if parent_size != 0 else 0
    )

    if len(holdings) > 1:
      holdings_changes = np.diff(holdings)

      # For buying: holdings increase (negative changes in the context of execution)
      # For selling: holdings decrease (positive changes in the context of execution)
      buys = -holdings_changes[holdings_changes < 0]
      sells = holdings_changes[holdings_changes > 0]

      metrics['total_volume_traded'] = np.sum(np.abs(holdings_changes))
      metrics['num_buys'] = len(buys)
      metrics['num_sells'] = len(sells)
      metrics['num_trades'] = len(buys) + len(sells)

      metrics['avg_trade_size'] = (
        np.mean(np.abs(holdings_changes[holdings_changes != 0]))
        if np.any(holdings_changes != 0)
        else 0
      )

      if len(timestamps) > 1 and len(holdings_changes) > 0:
        trade_times = timestamps[1:][holdings_changes != 0]

        if len(trade_times) > 0:
          first_trade_time = trade_times[0]
          last_trade_time = trade_times[-1]

          metrics['time_to_first_trade'] = first_trade_time - timestamps[0]

          metrics['execution_duration'] = (
            last_trade_time - first_trade_time if len(trade_times) > 1 else 0
          )
    else:
      metrics['total_volume_traded'] = 0
      metrics['num_buys'] = 0
      metrics['num_sells'] = 0
      metrics['num_trades'] = 0
      metrics['avg_trade_size'] = 0

    if len(rewards) > 1:
      metrics['reward_volatility'] = np.std(rewards)

      if len(pnls) > 0:
        running_max = np.maximum.accumulate(pnls)

        drawdowns = running_max - pnls
        metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0

        max_drawdown_idx = np.argmax(drawdowns)

        if running_max[max_drawdown_idx] > 0:
          metrics['max_drawdown_percentage'] = (
            100 * drawdowns[max_drawdown_idx] / running_max[max_drawdown_idx]
          )
        else:
          metrics['max_drawdown_percentage'] = 0
      else:
        metrics['max_drawdown'] = 0
        metrics['max_drawdown_percentage'] = 0
    else:
      metrics['reward_volatility'] = 0
      metrics['max_drawdown'] = 0
      metrics['max_drawdown_percentage'] = 0

    metrics['avg_spread'] = np.mean(spreads)
    metrics['spread_percentage'] = (
      100 * metrics['avg_spread'] / np.mean(mid_prices) if np.mean(mid_prices) > 0 else 0
    )

    if len(last_transactions) > 1:
      metrics['price_change'] = last_transactions[-1] - last_transactions[0]
      metrics['price_change_percentage'] = (
        100 * metrics['price_change'] / last_transactions[0] if last_transactions[0] > 0 else 0
      )

      if len(holdings) > 1:
        holdings_changes = np.diff(holdings)
        trade_indices = np.where(np.abs(holdings_changes) > 0)[0]

        if len(trade_indices) > 0:
          trade_prices = mid_prices[trade_indices + 1]  # +1 because diff shifts indices
          trade_volumes = np.abs(holdings_changes[trade_indices])

          if np.sum(trade_volumes) > 0:
            # Calculate actual execution VWAP
            execution_vwap = np.sum(trade_prices * trade_volumes) / np.sum(trade_volumes)
            metrics['execution_vwap'] = execution_vwap

            market_volumes = np.ones_like(mid_prices)  # Equal weight if no volume data
            market_vwap = np.sum(mid_prices * market_volumes) / np.sum(market_volumes)
            metrics['market_vwap'] = market_vwap

            if is_buying:
              # For buys: positive slippage means we paid more (worse execution)
              metrics['vwap_slippage'] = (
                100 * (execution_vwap - market_vwap) / market_vwap if market_vwap > 0 else 0
              )
            elif is_selling:
              # For sells: positive slippage means we received less (worse execution)
              metrics['vwap_slippage'] = (
                100 * (market_vwap - execution_vwap) / market_vwap if market_vwap > 0 else 0
              )
            else:
              # If neither buying nor selling (shouldn't happen)
              metrics['vwap_slippage'] = 0

    # Implementation shortfall
    if len(mid_prices) > 0:
      arrival_price = mid_prices[0]
      metrics['arrival_price'] = arrival_price

      if metrics.get('total_volume_traded', 0) > 0 and initial_holdings != final_holdings:
        # Calculate average execution price from trades
        if is_buying and metrics.get('num_buys', 0) > 0:
          # For buys: implementation shortfall = execution price - arrival price (positive is bad)
          avg_execution_price = metrics.get('execution_vwap', 0)
          if avg_execution_price > 0:
            metrics['implementation_shortfall'] = (
              100 * (avg_execution_price - arrival_price) / arrival_price
            )
          else:
            # If we don't have VWAP, try to estimate from PnL and volume
            metrics['implementation_shortfall'] = 0

        elif is_selling and metrics.get('num_sells', 0) > 0:
          # For sells: implementation shortfall = arrival price - execution price (positive is bad)
          avg_execution_price = metrics.get('execution_vwap', 0)
          if avg_execution_price > 0:
            metrics['implementation_shortfall'] = (
              100 * (arrival_price - avg_execution_price) / arrival_price
            )
          else:
            # If we don't have VWAP, try to estimate from PnL and volume
            metrics['implementation_shortfall'] = 0
        else:
          metrics['implementation_shortfall'] = 0
      else:
        metrics['implementation_shortfall'] = 0

      # Calculate opportunity cost for unexecuted portion
      if final_holdings != 0:
        final_price = mid_prices[-1]
        opportunity_cost = 0

        if is_buying and final_holdings < 0:  # Still have some to buy
          # Opportunity cost for buys = current price - arrival price (if price increased)
          opportunity_cost = (final_price - arrival_price) * abs(final_holdings)
        elif is_selling and final_holdings > 0:  # Still have some to sell
          # Opportunity cost for sells = arrival price - current price (if price decreased)
          opportunity_cost = (arrival_price - final_price) * abs(final_holdings)

        metrics['opportunity_cost'] = opportunity_cost

        metrics['opportunity_cost_per_share'] = (
          opportunity_cost / abs(final_holdings) if final_holdings != 0 else 0
        )

    print('\n===== Trading Performance Metrics =====')

    for key, value in sorted(metrics.items()):
      if isinstance(value, (int, float)):
        print(f'{key}: {value:.6f}')
      else:
        print(f'{key}: {value}')

    print('======================================\n')

    return metrics

  def test(self):
    self.run_episode()
    self.evaluate_metrics()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--model_path', type=str, required=True, help='Path to the trained agent model'
  )
  parser.add_argument(
    '--agent', type=str, required=True, choices=['PPO', 'DQN', 'RPPO'], help='Agent type'
  )
  args = parser.parse_args()

  tester = AgentTester(model_path=args.model_path, agent_name=args.agent)
  tester.test()


if __name__ == '__main__':
  main()
