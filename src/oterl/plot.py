from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def plot_tensorboard_runs(runs_dir: str | Path, scalars: list[str] | None = None):
  """Aggregate multiple tfevent files and plot the chosen scalar tags."""
  runs_dir = Path(runs_dir)

  scalars = scalars or ['A.Train/Reward', 'B.Loss/TotalLoss']

  ea = event_accumulator.EventAccumulator(str(runs_dir))
  ea.Reload()

  for tag in scalars:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=tag.split('/')[-1])

  plt.xlabel('Step')
  plt.legend()
  plt.title(runs_dir.name)
  plt.show()


def plot_trade_episode(trades_csv: str | Path):
  """
  Draw inventory & price evolution for a **single saved episode**.

  Expect a CSV with cols: ts, mid_px, qty, fill_px, inventory.
  You can dump this once per episode from `Agent.play` using pandas.
  """
  df = pd.read_csv(trades_csv)
  _, ax1 = plt.subplots()

  # price vs time
  ax1.plot(df['ts'], df['mid_px'], linewidth=1)
  ax1.set_ylabel('Mid‑price')

  # inventory on secondary axis
  ax2 = ax1.twinx()
  ax2.step(df['ts'], df['inventory'], where='post', linestyle='--')
  ax2.set_ylabel('Inventory')

  plt.title(str(trades_csv))
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  plot_tensorboard_runs('./src/runs')
