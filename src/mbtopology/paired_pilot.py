"""Pilot-only pairing of the existing cue schedule and optimizer steps."""

from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from . import gnn_training
from .experiment_io import (
    SessionYield,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
)
from .runtime import require
from .task_inputs import schedule

PILOT_SEEDS = (1337, 2026)


def paired_schedule(config, seed):
    task = config["task"]
    require(seed in PILOT_SEEDS, "paired schedule is restricted to the two approved pilot seeds")
    count, n_odors = task["trials_max"], task["n_odors"]
    pairs = [tuple(p) for p in task["odor_pairs"]]
    plus, minus = set(task["cs_plus"]), set(task["cs_minus"])
    require(
        len(pairs) * 2 == n_odors
        and sorted(cue for pair in pairs for cue in pair) == list(range(n_odors))
        and all(a in plus and b in minus for a, b in pairs)
        and (2 * count) % n_odors == 0,
        "paired pilot requires complete balanced cue blocks and disjoint plus/minus pairs",
    )
    extended = deepcopy(config)
    extended["task"]["trials_max"] = 2 * count
    source = schedule(extended, seed).reshape(-1, n_odors)
    grouped = []
    for block in source:
        order = {int(cue): i for i, cue in enumerate(block)}
        within_pair = [tuple(sorted(pair, key=order.__getitem__)) for pair in pairs]
        grouped.extend(sorted(within_pair, key=lambda pair: order[pair[0]]))
    return np.asarray(grouped, dtype=np.int64)


def mean_pair_loss(model, data, pair, plus):
    losses = []
    for odor in pair:
        value = gnn_training.predict(model, data, int(odor))
        losses.append(F.binary_cross_entropy(value, value.new_tensor(float(odor in plus))))
    return torch.stack(losses).mean()


def train_paired(model, optimizer, data, config, pairs, directory, signature, run, deadline):
    task, readout = config["task"], config["readout"]
    require(len(pairs) == task["trials_max"] and pairs.shape[1] == 2, "wrong pair-trial table")
    directory = Path(directory)
    history, reached, streak, elapsed = [], None, 0, 0.0
    saved = load_training_checkpoint(directory, signature)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, reached, streak, elapsed = (
            saved[key] for key in ("history", "reached_at", "streak", "elapsed_seconds")
        )
    require(0 <= len(history) <= len(pairs), "invalid paired checkpoint cursor")
    started = last_save = monotonic()
    in_trial = False

    def checkpoint():
        save_training_checkpoint(
            directory,
            signature,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "cursor": len(history),
                "reached_at": reached,
                "streak": streak,
                "elapsed_seconds": elapsed + monotonic() - started,
            },
        )
        run.save(completed_pair_trials=len(history), cue_presentations=2 * len(history))

    checkpoint()
    plus = frozenset(task["cs_plus"])
    try:
        while len(history) < len(pairs) and reached is None:
            if monotonic() >= deadline:
                raise SessionYield("paired pilot saved at the session deadline")
            trial = len(history) + 1
            pair = pairs[trial - 1]
            in_trial = True
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = mean_pair_loss(model, data, pair, plus)
            require(bool(torch.isfinite(loss)), "nonfinite paired BCE loss")
            loss.backward()
            require(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                ),
                "nonfinite paired gradient",
            )
            optimizer.step()
            score = gnn_training.probe(model, data, config)
            streak = streak + 1 if score >= readout["criterion"] else 0
            if streak >= readout["consecutive_trials"]:
                reached = trial
            history.append(
                {
                    "trial": trial,
                    "cue_first": int(pair[0]),
                    "cue_second": int(pair[1]),
                    "cue_presentations": trial * 2,
                    "training_loss": float(loss.detach().cpu()),
                    "discrimination": score,
                    "criterion_streak": streak,
                }
            )
            in_trial = False
            if monotonic() - last_save >= config["checkpoint_seconds"]:
                checkpoint()
                last_save = monotonic()
    finally:
        if not in_trial:
            checkpoint()
    return history, reached, elapsed + monotonic() - started


def learning_trend(history):
    values = np.asarray([row["discrimination"] for row in history], dtype=np.float64)
    require(len(values) >= 2 and np.isfinite(values).all(), "invalid discrimination history")
    endpoint_size = min(100, len(values) // 2)
    start = float(values[:endpoint_size].mean())
    end = float(values[-endpoint_size:].mean())
    slope = float(np.polyfit(np.arange(1, len(values) + 1), values, 1)[0])
    return {
        "rising": bool(slope > 0 and end > start),
        "end_window_size": endpoint_size,
        "initial_mean": start,
        "terminal_mean": end,
        "ols_slope_per_pair_trial": slope,
        "maximum": float(values.max()),
        "mean_windows_of_100": [
            float(values[i : i + 100].mean()) for i in range(0, len(values), 100)
        ],
    }
