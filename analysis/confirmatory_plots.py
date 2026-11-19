"""Render the descriptive figures from the locked seed summaries."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from confirmatory_data import CAP, CONDITIONS, LEVELS
from confirmatory_stats import noise_ace, seed_values

LABELS = ("Real", "Null-A", "Null-B")
COLORS = ("#235f95", "#a04425", "#4c7359")


def box_strip(jobs, getter, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    draw_strip(ax, jobs, getter, ylabel, title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_strip(ax, jobs, getter, ylabel, title):
    values = [seed_values(jobs, getter, (c,)) for c in CONDITIONS]
    boxes = ax.boxplot(
        values, positions=(1, 2, 3), widths=0.46, showfliers=False, patch_artist=True
    )
    jitter = np.random.default_rng(20261119).uniform(-0.16, 0.16, len(values[0]))
    for i, (box, scores, color) in enumerate(zip(boxes["boxes"], values, COLORS, strict=True), 1):
        box.set(facecolor=color, alpha=0.15, edgecolor=color)
        ax.scatter(i + jitter, scores, s=19, color=color, alpha=0.66, linewidths=0.0, zorder=3)
    ax.set_xticks((1, 2, 3), LABELS)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def curve(jobs, getter, x, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    for condition, label, color in zip(CONDITIONS, LABELS, COLORS, strict=True):
        curves = seed_values(jobs, getter, (condition,))
        q1, q3 = np.quantile(curves, [0.25, 0.75], axis=0)
        ax.fill_between(x, q1, q3, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, curves.mean(axis=0), label=label, color=color, linewidth=1.8)
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 1))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def padded_curve(job):
    values = np.full(CAP, job.terminal_mean, dtype=float)
    values[: len(job.observed)] = job.observed
    return values


def noise_panel(jobs, path):
    fig, (dose, ace) = plt.subplots(
        1, 2, figsize=(11.6, 4.4), layout="constrained", width_ratios=(1.55, 1)
    )
    levels = np.array(LEVELS)
    jitter = np.random.default_rng(20261119).uniform(-0.004, 0.004, (30, len(levels)))
    dose.axhline(0.5, color="#697787", linestyle=":", linewidth=1.2, label="Chance")
    for offset, condition, label, color in zip(
        (-0.013, 0, 0.013), CONDITIONS, LABELS, COLORS, strict=True
    ):
        scores = seed_values(jobs, lambda job: job.noise_scores, (condition,))
        x = levels + offset
        q1, median, q3 = np.quantile(scores, [0.25, 0.5, 0.75], axis=0)
        dose.scatter(
            (x + jitter).ravel(), scores.ravel(), s=10, color=color, alpha=0.36, linewidths=0
        )
        dose.fill_between(x, q1, q3, color=color, alpha=0.16, linewidth=0)
        dose.plot(x, median, label=label, color=color, linewidth=1.8, marker="o", markersize=4)
    dose.set_xticks(levels, ("0", "0.1", "0.2", "0.3"))
    dose.set(xlabel="Fraction of active PNs exchanged", ylabel="Held-out discrimination")
    dose.set_title("A   Frozen model at four noise levels", loc="left", fontweight="bold")
    dose.grid(axis="y", alpha=0.2)
    dose.spines[["top", "right"]].set_visible(False)
    dose.legend(frameon=False, ncol=2)
    draw_strip(ace, jobs, noise_ace, "ACE (normalized noise AUC)", "B   Seed distribution of ACE")
    fig.savefig(path, dpi=180)
    plt.close(fig)
