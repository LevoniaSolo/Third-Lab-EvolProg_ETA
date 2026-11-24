import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_pareto(front, selected, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step([row["n_features"] for row in front],
            [row["validation_accuracy"] for row in front],
            where="post", color="C0", alpha=0.45)
    for supported, marker, color, label in (
        (True, "o", "C0", "Supported by weighted sum"),
        (False, "x", "C3", "Unsupported by weighted sum"),
    ):
        rows = [row for row in front if row["supported"] == supported]
        if rows:
            ax.scatter([row["n_features"] for row in rows],
                       [row["validation_accuracy"] for row in rows],
                       marker=marker, color=color, label=label, zorder=3)
    labels = {
        "min_features": "Fewest features",
        "compromise": "Compromise",
        "max_quality": "Best validation accuracy",
    }
    for row in selected:
        position = row["n_features"], row["validation_accuracy"]
        if row["role"] == "all_features":
            ax.scatter(*position, marker="s", color="C2", zorder=3, label="All 18 features")
        else:
            offset = (5, 18) if row["role"] == "compromise" else (5, 8)
            ax.annotate(labels[row["role"]], position, xytext=offset,
                        textcoords="offset points", fontsize=9)
    ax.set(xlabel="Selected features", ylabel="Validation accuracy",
           title="Pareto front of evaluated masks")
    ax.set_xticks(sorted({row["n_features"] for row in front + selected}))
    ax.margins(y=0.17)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history, path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    runs = {}
    for row in history:
        runs.setdefault(row["run"], []).append(row)
    generations = [row["generation"] for row in next(iter(runs.values()))]
    for ax, (key, label) in zip(axes, (
        ("hypervolume", "Normalized hypervolume"),
        ("best_validation_accuracy", "Best validation accuracy"),
        ("min_features", "Fewest features"),
    )):
        values = np.array([[row[key] for row in rows] for rows in runs.values()])
        ax.plot(generations, values.T, color="C0", alpha=0.25)
        ax.plot(generations, values.mean(axis=0), color="C0", linewidth=2, label="Mean")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_title("Search progress across independent runs")
    axes[0].legend()
    axes[-1].set_xlabel("Generation")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
