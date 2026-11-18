"""Endpoints for the early-stopped binary conditioning protocol."""

from math import fsum


def rolling_score(history, window):
    if len(history) < window:
        return None
    return fsum(row["discrimination"] for row in history[-window:]) / window


def add_observation(history, values, window):
    a_plus, a_minus = values
    row = {
        "trial": len(history) + 1,
        "discrimination": (a_plus + 1 - a_minus) / 2,
        "heldout_a_plus": a_plus,
        "heldout_a_minus": a_minus,
    }
    history.append(row)
    if window:
        row["rolling_discrimination"] = rolling_score(history, window)


def reached(history, threshold):
    latest = history[-1].get("rolling_discrimination") if history else None
    return latest is not None and latest >= threshold


def endpoints(history, config):
    threshold, cap = config["threshold"], config["trials"]
    stop = history[-1]
    hit = reached(history, threshold)
    scores = [row["discrimination"] for row in history]
    # Treat each trial as one unit of area. Extend the terminal rolling score after early-stop.
    area = fsum(scores) + (cap - len(scores)) * stop["rolling_discrimination"]
    return {
        "trials": len(history),
        "ttc": len(history) if hit else None,
        "right_censored": not hit,
        "first_single_crossing": next(
            (row["trial"] for row in history if row["discrimination"] >= threshold), None
        ),
        "terminal_rolling_discrimination": stop["rolling_discrimination"],
        "auc_to_cap_normalized": area / cap,
    }
