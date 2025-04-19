## Modify the following files
In `.venv/lib/python3.9/site-packages/abides_gym/envs/core_environment.py`, comment out lines 146-148:
```
    # assert self.observation_space.contains(
    #     self.state
    # ), f"INVALID STATE {self.state}"
```
In `.venv/lib/python3.9/site-packages/abides_gym/envs/markets_daily_investor_environment_v0.py`, replace the method `raw_state_to_state` by
```
   def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP for the daily investor v0 environnement
        """
        # 0)  Preliminary
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) Imbalance
        imbalances = [
            markets_agent_utils.get_imbalance(b, a, depth=3)
            for (b, a) in zip(bids, asks)
        ]
        # 3) Returns
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        returns = np.diff(mid_prices)
        padded_returns = np.zeros(self.state_history_length - 1)
        padded_returns[-len(returns) :] = (
            returns if len(returns) > 0 else padded_returns
        )

        # 4) Spread
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]
        spreads = np.array(best_asks) - np.array(best_bids)

        # 5) direction feature
        direction_features = np.array(mid_prices) - np.array(last_transactions)
        #6) Compute State (Holdings, Imbalance, Spread, DirectionFeature + Returns)

        N = 10  # Number of levels to include
        latest_bids = bids[-1] if bids else []
        latest_asks = asks[-1] if asks else []

        bid_prices = [p for (p, v) in latest_bids[:N]]
        bid_volumes = [v for (p, v) in latest_bids[:N]]
        ask_prices = [p for (p, v) in latest_asks[:N]]
        ask_volumes = [v for (p, v) in latest_asks[:N]]

        # Padding if fewer than N levels
        bid_prices += [0] * (N - len(bid_prices))
        bid_volumes += [0] * (N - len(bid_volumes))
        ask_prices += [0] * (N - len(ask_prices))
        ask_volumes += [0] * (N - len(ask_volumes))

        # 7) Compute State
        computed_state = np.array(
            [holdings[-1], imbalances[-1], spreads[-1], direction_features[-1]]
            + padded_returns.tolist()
            + bid_prices + bid_volumes
            + ask_prices + ask_volumes,
            dtype=np.float32,
        )
        return computed_state.reshape(
            self.num_state_features + 4 * N, 1
        )  
```

## oterl

**oterl** (**Optimal Trade Execution Reinforcement Learning**) contains the code 
for the [COMP579](https://www.mcgill.ca/study/2024-2025/courses/comp-579) 
final project, a Reinforcement Learning (RL) course offered at McGill
University.

## Background

Optimal trade execution involves buying or selling a large position within a
fixed time horizon while minimizing trading costs and market impact (Hafsi,
2024). It requires balancing fast execution (to reduce price risk) against slow
execution (to minimize impact). We propose using reinforcement learning (RL) to
learn a strategy that optimally navigates these trade-offs.

RL is a natural approach to this sequential decision-making problem. It allows
an agent to learn an execution policy from experience without a rigid market
model (Kabbani, 2022). Unlike traditional frameworks reliant on fixed
assumptions, RL can adapt to complex market dynamics (Zhang, 2019).

Early applications of RL to high-frequency trading and execution have shown
promising results (Hafsi, 2024). Researchers have explored deep Q-networks (DQN)
and actor-critic methods to address scalability and continuous action spaces
(Xiong, 2025; Huang, 2025). Tools like FinRL (Liu, 2022) and ABIDES (Byrd et
al., 2019; Amrouni et al., 2021) enable realistic simulations, while Qlib (Yang
et al., 2020) supports data-driven experimentation.

We formulate the task as a Markov Decision Process with state variables (e.g.,
remaining inventory, time, market indicators) and actions (quantity to trade).
The reward reflects execution cost, slippage, and residual inventory (Walia,
2025). We will train a deep RL agent (e.g., DQN or Proximal Policy Optimization)
within ABIDES to capture limit order book dynamics. Our GPU-based cluster will
support parallelized training and hyperparameter tuning.

We expect the RL-driven strategy to outperform standard benchmarks such as
Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP). By
analyzing performance under various volatility and liquidity conditions, we aim
to highlight how RL adapts to non-stationary markets (Kruthof, 2025). This work
will provide practical insights into deploying adaptive RL frameworks for
optimal trade execution, ultimately reducing costs and improving overall trading
performance.

## Development

To get started, follow the steps to get
[mise](https://mise.jdx.dev/getting-started.html) (optional) and
[uv](https://docs.astral.sh/uv/) installed on your system.

For **macOS** users, the easiest way is with [homebrew](https://brew.sh/):

```bash
brew install mise uv
```

`mise` makes it easy to manage multiple Python versions on your system. For this
project we pin the Python version to `3.9`. If you have it installed, `mise`
will switch your system Python to the pinned version upon entering the project
directory.

We use `uv` to manage dependencies.

Once these are setup, you can run the entry-point with:

```bash
uv run oterl
```

## Prior Art

Amrouni, B., Sellaouti, A., Koubaa, A., & Ammar, A. (2021). abides-gym: Gym
Environments for Multi-agent Discrete Event Simulation and Application to
Financial Markets. arXiv preprint arXiv:2104.11941.

Byrd, J., Hyett, S., Glickman, M., & Patel, J. (2019). ABIDES: Towards
High-Fidelity Market Simulation for AI Research. arXiv preprint
arXiv:1904.12066.

Hafsi, A. (2024). Optimal Execution with Reinforcement Learning. Journal of
Quantitative Finance, 52(1), 105–121.

Huang, H. (2025). A Deep Reinforcement Learning Framework for Dynamic Portfolio
Optimization: Evidence from China’s Stock Market. Quantitative Finance Letters,
2(1), 30–48.

Kabbani, M. (2022). Deep Reinforcement Learning Approach for Trading Automation
in the Stock Market. IEEE Access, 10, 90903–90911.

Kruthof, R. (2025). Can Deep Reinforcement Learning Beat 1/N? International
Journal of Financial Studies, 13(2), 77–95.

Liu, B. (2022). FinRL: A Deep Reinforcement Learning Library for Automated Stock
Trading in Quantitative Finance. SoftwareX, 18, 101138.

Walia, J. (2025). Predicting Liquidity-Aware Bond Yields Using Causal GANs and
Deep Reinforcement Learning with LLM Evaluation. Journal of Fixed Income
Analytics, 4(2), 50–69.

Xiong, X. (2025). FLAG-Trader: Fusion LLM-Agent with Gradient-based
Reinforcement Learning for Financial Trading. Expert Systems with Applications,
210, 118589.

Yang, Y., Wu, S., & Li, P. (2020). Qlib: An AI-Oriented Quantitative Investment
Platform. arXiv preprint arXiv:2010.11014.

Zhang, J. (2019). Deep Reinforcement Learning for Trading. Proceedings of the
2019 International Conference on Data Science, 112–121.
