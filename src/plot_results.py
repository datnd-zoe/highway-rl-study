"""Plot mean ± std of collision_rate and mean_speed per agent/config.

Usage:
    python -m src.plot_results
    python -m src.plot_results --csv results/eval_summary.csv --out results/comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần cửa sổ GUI
import matplotlib.pyplot as plt
import pandas as pd

METRICS = [
    ("collision_rate", "Collision rate (lower is better)"),
    ("mean_speed", "Mean speed [m/s]"),
]
# Thứ tự cố định: baseline trước, rồi PPO theo mức phạt tăng dần.
ORDER = ["random", "defensive", "ppo_low", "ppo_mid", "ppo_high"]
# Mỗi nhóm một màu cố định (bộ màu thân thiện với người mù màu).
COLORS = {
    "random": "#8c8c8c",
    "defensive": "#0072B2",
    "ppo_low": "#E69F00",
    "ppo_mid": "#009E73",
    "ppo_high": "#CC79A7",
}


def aggregate(csv_path: Path) -> pd.DataFrame:
    """Aggregate the eval summary into mean/std per group.

    A group is the baseline agent name, or the PPO config name. For PPO the
    statistics are over training seeds (each row is already a 50-episode mean);
    baselines have one row each, so their std is 0.
    """
    df = pd.read_csv(csv_path)
    df["group"] = df.apply(
        lambda r: r["config"] if r["agent"] == "ppo" else r["agent"], axis=1
    )
    cols = [m for m, _ in METRICS]
    # ddof=0 ngầm định bởi np.std; với 1 dòng std = 0 thay vì NaN.
    agg = df.groupby("group")[cols].agg(["mean", lambda s: s.std(ddof=0)])
    agg.columns = [f"{m}_{'mean' if s == 'mean' else 'std'}" for m, s in agg.columns]
    agg["n_runs"] = df.groupby("group").size()
    agg = agg.reindex([g for g in ORDER if g in agg.index])
    return agg


def plot(agg: pd.DataFrame, out_path: Path) -> Path:
    """Draw one bar panel per metric (separate axes, no dual-axis)."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.2))
    for ax, (metric, title) in zip(axes, METRICS):
        means, stds = agg[f"{metric}_mean"], agg[f"{metric}_std"]
        ax.bar(
            agg.index, means, yerr=stds, capsize=4, width=0.6,
            color=[COLORS[g] for g in agg.index],
        )
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=20)
        for x, (m, s) in enumerate(zip(means, stds)):
            ax.text(x, m + s, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 1.1)
    fig.suptitle("Effect of collision_reward (PPO: mean ± std over training seeds)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument("--csv", default="results/eval_summary.csv")
    parser.add_argument("--out", default="results/comparison.png")
    args = parser.parse_args()
    agg = aggregate(Path(args.csv))
    print(agg.round(3).to_string())
    print(f"Saved {plot(agg, Path(args.out))}")


if __name__ == "__main__":
    main()
