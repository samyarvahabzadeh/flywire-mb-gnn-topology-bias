"""Train/validation/held-out linear probes of the frozen real GraphSAGE body."""

import argparse
import subprocess
from itertools import combinations
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from exploratory_rule_models import mb_embeddings
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.cue_families import check_disjoint
from mbtopology.real_cues import binary_cues, check_projected_disjoint, project_cues
from mbtopology.runtime import atomic_json, now, read_json, require, sha256
from mbtopology.task_inputs import load_source

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_readability_and_plasticity.json"
SPLITS = ("train", "validation", "heldout")


def read_plan():
    import yaml

    plan = read_json(PLAN)
    config = ROOT / plan["binary_config"]
    require(sha256(config) == plan["binary_config_sha256"], "frozen binary config differs")
    return plan, yaml.safe_load(config.read_text(encoding="utf-8"))


def real_projection(source):
    _, nodes, graph = load_source(ROOT, source)
    role = nodes["role"].to_numpy()
    pn, kc = np.flatnonzero(role == "PN"), np.flatnonzero(role == "KC")
    projection = graph[pn][:, kc].tocsr()
    projection.sort_indices()
    return nodes, graph, kc, projection


def statistics(target, logits):
    positive = target == 1
    n, m = positive.sum(), (~positive).sum()
    ranks = rankdata(logits)
    auc = (ranks[positive].sum() - n * (n + 1) / 2) / (n * m)
    predicted = logits >= 0
    return {
        "balanced_accuracy": float(
            (predicted[positive].mean() + (~predicted[~positive]).mean()) / 2
        ),
        "auroc": float(auc),
        "log_loss": float((np.logaddexp(0, logits) - target * logits).mean()),
    }


def fit_linear(x, y, l2):
    def mean_loss_and_gradient(parameters):
        weight, bias = parameters[:-1], parameters[-1]
        z = x @ weight + bias
        error = (expit(z) - y) / y.size
        objective = (np.logaddexp(0, z) - y * z).mean() + l2 * (weight @ weight) / 2
        return objective, np.r_[x.T @ error + l2 * weight, error.sum()]

    fit = minimize(
        mean_loss_and_gradient,
        np.zeros(x.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-10},
    )
    require(fit.success, f"linear optimization failed at L2={l2}: {fit.message}")
    return fit.x


def describe(x, plan):
    std = x.std(axis=0, dtype=np.float64)
    margin = plan["saturation_margin"]
    return {
        "features": x.shape[1],
        "min": float(x.min()),
        "max": float(x.max()),
        "coordinate_sd_q25_median_q75": np.quantile(std, [0.25, 0.5, 0.75]).tolist(),
        "constant_coordinates": int((std <= plan["constant_std"]).sum()),
        "fraction_at_or_below_0_01": float((x <= margin).mean()),
        "fraction_at_or_above_0_99": float((x >= 1 - margin).mean()),
    }


def probe(features, labels, plan):
    train, validation, heldout = (features[key] for key in SPLITS)
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    active = std > plan["constant_std"]

    def transform(x):
        return (x[:, active].astype(np.float64) - mean[active]) / std[active]

    x, val = transform(train), transform(validation)
    candidates = []
    for l2 in plan["l2_grid"]:
        beta = fit_linear(x, labels["train"], l2)
        score = statistics(labels["validation"], val @ beta[:-1] + beta[-1])
        candidates.append((l2, beta, score))
    l2, beta, _ = min(
        candidates, key=lambda r: (-r[2]["balanced_accuracy"], r[2]["log_loss"], -r[0])
    )
    result = {
        key: statistics(labels[key], array @ beta[:-1] + beta[-1])
        for key, array in zip(SPLITS, (x, val, transform(heldout)), strict=True)
    }
    return {
        "selected_l2": l2,
        "retained": int(active.sum()),
        "scores": result,
        "scales": {key: describe(features[key], plan) for key in SPLITS},
        "validation_grid": [{"l2": strength, **sc} for strength, _, sc in candidates],
    }


def make_splits(projection, config, plan):
    family = config["family_design"]
    pairs = config["pairs_per_trial"]
    train, valid, heldout = plan["trials_per_split"]
    main, _, _ = binary_cues(projection, plan["seed"], max(train, heldout), pairs, family)
    extra, _, _ = binary_cues(
        projection,
        plan["seed"],
        valid,
        pairs,
        {**family, "heldout_stream": plan["validation_stream"]},
    )
    patterns = (main[0][:train], extra[1], main[1][:heldout])
    drives = [project_cues(projection, cues)[0] for cues in patterns]
    for a, b in combinations(range(3), 2):
        check_disjoint(patterns[a], patterns[b], "PN")
        check_projected_disjoint(drives[a], drives[b])
    return dict(zip(SPLITS, drives, strict=True))


@torch.no_grad()
def extract(model, inputs, drives, presentations):
    result = {}
    for key, drive in drives.items():
        batches = []
        for t in range(drive.shape[0] // presentations):
            h = mb_embeddings(model, inputs, drive, t, presentations)
            y = torch.sigmoid(model.head(h)).squeeze(-1)
            a = y[:, inputs["avoidance"]].mean(1)
            b = y[:, inputs["approach"]].mean(1)
            batches.append(
                (
                    h.flatten(1).cpu().numpy(),
                    y.cpu().numpy(),
                    torch.stack((a, b), dim=1).cpu().numpy(),
                    ((a - b + 1) / 2).cpu().numpy(),
                )
            )
        hidden, output, means, scalar = map(np.concatenate, zip(*batches, strict=True))
        result[key] = {
            "kc_input": drive.toarray(),
            "mbon_full_output": output,
            "group_means": means,
            "fixed_scalar": scalar[:, None],
            "mbon_full_prehead": hidden,
        }
        print(f"READABILITY_EXTRACTED {key} {len(output)}", flush=True)
    return result


def artifact_manifest(output, plan, started, files):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(
        output / "MANIFEST.json",
        {
            "scope": plan["scope"],
            "git_commit": commit,
            "config_sha256": sha256(PLAN),
            "source_graph_manifest_sha256": sha256(ROOT / plan["source"]),
            "completed_utc": now(),
            "wall_seconds": monotonic() - started,
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = monotonic()
    plan, cfg = read_plan()
    p = plan["readability"]
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    _, _, _, projection = real_projection(plan["source"])
    drives = make_splits(projection, cfg, p)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    inputs = gnn_training.frozen_inputs(
        ROOT, bundle["task"], plan["source"], 0, bundle["nulls"], bundle["models"], "cpu"
    )
    inputs["base"] = models.node_features(inputs["roles"], inputs["kc"], inputs["kc"][:0])
    edges = inputs["graph"].edge_index
    inputs["neighbors"] = edges[:, edges[0] != edges[1]]
    model = make_benchmark_model("GraphSAGE", bundle["models"], p["seed"], aggregation="max").eval()
    values = extract(model, inputs, drives, 2 * cfg["pairs_per_trial"])
    labels = {k: np.tile([1.0, 0.0], v.shape[0] // 2) for k, v in drives.items()}
    elapsed_extract = monotonic() - started
    metrics = {}
    for name in values["train"]:
        metrics[name] = probe({key: values[key][name] for key in SPLITS}, labels, p)
        print(f"READABILITY_PROBED {name}", flush=True)
    elapsed_probe = monotonic() - started - elapsed_extract
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report, features = output / "readability.json", output / "representations.npz"
    atomic_json(
        report,
        {
            "seed": p["seed"],
            "sample_counts": {k: len(v) for k, v in labels.items()},
            "extract_seconds": elapsed_extract,
            "probe_seconds": elapsed_probe,
            "paired_exact_disjoint": True,
            "metrics": metrics,
        },
    )
    np.savez_compressed(
        features,
        **{f"{k}_{name}": value for k in SPLITS for name, value in values[k].items()},
        **{f"{k}_label": y for k, y in labels.items()},
    )
    artifact_manifest(output, plan, started, (report, features))


if __name__ == "__main__":
    main()
