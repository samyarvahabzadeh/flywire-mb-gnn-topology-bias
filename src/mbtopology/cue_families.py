"""Topology-independent cue families for held-out exploratory trials."""

from math import ceil

import numpy as np

from .runtime import require


def make_cues(cfg, seed, n_pn, n_kc, trials, pairs):
    rng = np.random.default_rng(cfg["layout_seed"])
    n_cues = 2 * len(pairs)
    shared = cfg["pair_shared_pns"]
    specific = cfg["family_specific_pns"]
    variable = cfg["instance_variable_pns"]
    fan_in = cfg["symmetric_pn_afferents_per_kc"]
    split = len(pairs) * shared
    stop = split + n_cues * specific
    require(stop + variable <= n_pn and fan_in <= n_pn, "PN pool too small for cue families")
    shuffled = rng.permutation(n_pn)
    common = shuffled[:split].reshape(len(pairs), shared)
    private = shuffled[split:stop].reshape(n_cues, specific)
    pool = shuffled[stop:]
    cores = np.empty((n_cues, shared + specific), dtype=np.int64)
    for i, pair in enumerate(pairs):
        for family in pair:
            cores[family] = np.concatenate((common[i], private[family]))

    # A fixed artificial encoder, independent of the connectome and of CS labels.
    afferents = np.array([rng.choice(n_pn, fan_in, replace=False) for _ in range(n_kc)])
    tie_breaker = (rng.permutation(n_kc) / n_kc).astype(np.float32) * 0.25
    n_active = int(np.floor(n_kc * cfg["kc_active_fraction"] + 0.5))
    require(0 < n_active < n_kc, "invalid KC activity count")
    pn_splits, kc_splits = [], []
    for stream in (cfg["train_stream"], cfg["holdout_stream"]):
        draw = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        flat_pn = np.empty((trials * n_cues, shared + specific + variable), dtype=np.int16)
        flat_kc = np.empty((trials * n_cues, n_active), dtype=np.int16)
        for i, row in enumerate(flat_pn):
            row[:] = np.sort(
                np.concatenate((cores[i % n_cues], draw.choice(pool, variable, False)))
            )
        for low in range(0, len(flat_pn), 256):
            source = flat_pn[low : low + 256]
            mask = np.zeros((len(source), n_pn), dtype=np.bool_)
            mask[np.arange(len(source))[:, None], source] = True
            score = mask[:, afferents].sum(axis=2, dtype=np.int16) + tie_breaker
            indices = np.argpartition(score, -n_active, axis=1)[:, -n_active:]
            flat_kc[low : low + len(source)] = np.sort(indices, axis=1)
        pn_splits.append(flat_pn.reshape(trials, n_cues, -1))
        kc_splits.append(flat_kc.reshape(trials, n_cues, -1))
    check_disjoint(*pn_splits, "PN")
    check_disjoint(*kc_splits, "KC")
    return pn_splits, kc_splits, len(pool)


def check_disjoint(train, probe, name):
    seen = {row.tobytes() for row in train.reshape(-1, train.shape[-1])}
    require(
        not any(row.tobytes() in seen for row in probe.reshape(-1, probe.shape[-1])),
        f"held-out {name} cue overlaps an exact train instance",
    )


def overlap_statistics(train, probe, pairs):
    def summarize(rows):
        x = np.asarray(rows, dtype=float)
        return {
            "mean_shared": float(x.mean()),
            "sd_shared": float(x.std()),
            "mean_jaccard": float((x / (2 * train.shape[-1] - x)).mean()),
        }

    out = {"same_family": [], "same_pair_different_family": [], "different_pair": []}
    # Every training row is compared with independently drawn held-out instances.
    pair_of = {cue: i for i, pair in enumerate(pairs) for cue in pair}
    for train_block, probe_block in zip(train, probe, strict=True):
        for cue, a in enumerate(train_block):
            twin = next(x for x in pairs[pair_of[cue]] if x != cue)
            other = pairs[(pair_of[cue] + 1) % len(pairs)][0]
            for key, k in zip(out, (cue, twin, other), strict=True):
                shared = np.intersect1d(a, probe_block[k], assume_unique=True).size
                out[key].append(shared)
    return {key: summarize(values) for key, values in out.items()}


def second_order(cfg, pool):
    n = cfg["instance_variable_pns"]
    p = n / pool
    joint = n * (n - 1) / (pool * (pool - 1))
    return {
        "variable_pool_size": pool,
        "variable_pn_activation_probability": p,
        "two_variable_pn_joint_activation_probability": joint,
        "two_variable_pn_covariance": joint - p * p,
        "core_pn_activation_probability": 1.0,
        "core_pn_with_variable_pn_joint_activation_probability": p,
        "same_family_expected_shared": cfg["pair_shared_pns"] + cfg["family_specific_pns"] + n * p,
        "same_pair_different_family_expected_shared": cfg["pair_shared_pns"] + n * p,
        "different_pair_expected_shared": n * p,
    }


def choose_cap(values, first_ttc, threshold, cfg):
    """Return a proposed cap after the first sustained qualifying window."""
    if first_ttc is None:
        return None, None
    window = cfg["stability_window"]
    rounded = cfg["round_up_to_trials"]
    hits = (np.asarray(values) >= threshold).astype(np.int64)
    totals = np.convolve(hits, np.ones(window, dtype=np.int64), mode="valid")
    for stable, count in enumerate(totals, start=window):
        if stable >= first_ttc and count / window >= cfg["required_window_fraction_at_criterion"]:
            cap = rounded * ceil(stable * (1 + cfg["padding_fraction"]) / rounded)
            return (cap if cap <= len(values) else None), stable
    return None, None
