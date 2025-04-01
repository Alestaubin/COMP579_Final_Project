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
