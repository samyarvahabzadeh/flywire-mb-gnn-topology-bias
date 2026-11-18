"""Run the predeclared, single-seed diagnostic ladder. Never launch a sweep."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mbtopology import gnn_training, models, runner
from mbtopology.registration import require_registered
from mbtopology.runtime import Exclusive, Run, atomic_json, fingerprint, provenance, require
from mbtopology.task_inputs import initial_weights, load_source

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
SANITY_LIMIT = 99
SANITY_ACCURACY = 0.90
KC_ACCURACY = 0.75
SNR_RATIO_LIMIT = 3.0
SNR_TRIALS = 100
KEEP_PROBABILITY = 0.90
PROBE_TRAIN_VIEWS = 16
PROBE_TEST_VIEWS = 64
CONTRAST_SCALE = 4.0


def fresh_model(bundle, device):
    return models.make_model("GCN", bundle["models"], SEED, device)


def make_optimizer(model, bundle):
    config = bundle["models"]["optimizer"]
    return torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def synthetic_features(rng, n):
    teacher = np.array([4.0, -3.0, 2.0, 0.0, 0.0])
    teacher /= np.linalg.norm(teacher)
    nuisance = rng.normal(size=(n, len(teacher)))
    nuisance -= (nuisance @ teacher)[:, None] * teacher
    signs = rng.permutation(np.resize([-1.0, 1.0], n))
    margin = 12 * signs * rng.uniform(1, 2, size=n)
    features = margin[:, None] * teacher + 0.3 * nuisance
    return features, features @ teacher > 0


def sanity(bundle, device, directory, progress):
    rng = np.random.default_rng(SEED)
    train_x, train_y = synthetic_features(rng, SANITY_LIMIT)
    test_x, test_y = synthetic_features(rng, 256)
    train = torch.tensor(train_x, dtype=torch.float32, device=device)
    heldout = torch.tensor(test_x, dtype=torch.float32, device=device)
    expected = torch.tensor(test_y, device=device)
    linear = nn.Linear(train.shape[1], 1, device=device)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()

    def predict(model, data, index):
        return torch.sigmoid(model(data["features"][index])).squeeze()

    @torch.no_grad()
    def probe(model, _data, _task):
        model.eval()
        return float(((model(heldout).squeeze(1) > 0) == expected).float().mean())

    prototype = {
        "task": {"trials_max": SANITY_LIMIT, "cs_plus": np.flatnonzero(train_y).tolist()},
        "readout": {"criterion": SANITY_ACCURACY, "consecutive_trials": 1},
    }
    chance = probe(linear, None, prototype)
    with (
        patch.object(gnn_training, "predict", predict),
        patch.object(gnn_training, "probe", probe),
    ):
        outcome = gnn_training.train_loop(
            linear,
            make_optimizer(linear, bundle),
            {"features": train, "schedule": np.arange(SANITY_LIMIT)},
            prototype,
            directory / "sanity",
            fingerprint({"seed": SEED, "sanity": prototype, "bundle": bundle["bundle_sha256"]}),
            progress,
        )
    history = outcome["history"]
    return {
        "pass": outcome["trials_to_criterion"] is not None,
        "baseline_accuracy": chance,
        "trials": outcome["trials_completed"],
        "first_trial_at_0_90": outcome["trials_to_criterion"],
        "held_out_examples": len(test_y),
        "last_accuracy": history[-1]["discrimination"],
        "last_loss": history[-1]["training_loss"],
    }


def node_features(data):
    return [models.node_features(data["roles"], data["kc"], cue) for cue in data["patterns"]]


def rescale_contrast(features, pairs, scale):
    scaled = list(features)
    for a, b in pairs:
        midpoint = (features[a] + features[b]) / 2
        scaled[a] = midpoint + scale * (features[a] - midpoint)
        scaled[b] = midpoint + scale * (features[b] - midpoint)
    return scaled


@torch.no_grad()
def representations(model, graph, features, kc, mbon):
    model.eval()
    baseline = features[0].clone()
    baseline[:, -1] = 0
    home = model.encoder(baseline).relu()
    hidden0 = model.convs[0](home, graph.edge_index, edge_weight=graph.gcn_norm).relu()
    hidden1 = model.convs[1](hidden0, graph.edge_index, edge_weight=graph.gcn_norm).relu()
    cases = {"kc_drive": [], "kc_layer1": [], "kc_layer2": [], "mbon_layer2": []}
    for cue in features:
        x = model.encoder(cue).relu()
        hidden = model.convs[0](x, graph.edge_index, edge_weight=graph.gcn_norm).relu()
        hidden2 = model.convs[1](hidden, graph.edge_index, edge_weight=graph.gcn_norm).relu()
        for name, tensor in (
            ("kc_drive", cue[kc, -1:]),
            ("kc_layer1", hidden[kc] - hidden0[kc]),
            ("kc_layer2", hidden2[kc] - hidden1[kc]),
            ("mbon_layer2", hidden2[mbon] - hidden1[mbon]),
        ):
            cases[name].append(tensor.cpu().numpy().astype(np.float64))
    return cases


def pair_probe(positive, negative):
    """Fit a nearest-centroid linear probe on train views, test independent masks."""
    rng = np.random.default_rng(SEED)
    masks = rng.random((2, PROBE_TRAIN_VIEWS + PROBE_TEST_VIEWS, positive.shape[0]))
    masks = masks < KEEP_PROBABILITY
    centers = [
        tensor * masks[i, :PROBE_TRAIN_VIEWS].mean(axis=0)[:, None]
        for i, tensor in enumerate((positive, negative))
    ]
    direction = centers[0] - centers[1]
    intercept = float(np.sum(direction * ((centers[0] + centers[1]) / 2)))
    predictions = [
        (masks[i, PROBE_TRAIN_VIEWS:] @ (tensor * direction).sum(axis=1)) > intercept
        for i, tensor in enumerate((positive, negative))
    ]
    return {
        "balanced_accuracy": float((predictions[0].mean() + (~predictions[1]).mean()) / 2),
        "clean_cue_distance_l2": float(np.linalg.norm(positive - negative)),
        "clean_margin": float(np.sum(direction * (positive - negative))),
    }


def probe_pairs(activations, pairs):
    return {
        name: {
            label: pair_probe(vectors[a], vectors[b])
            for label, (a, b) in zip(("A", "B"), pairs, strict=True)
        }
        for name, vectors in activations.items()
    }


def no_normalization(bundle, data, graph_path):
    _, nodes, csr = load_source(ROOT, graph_path)
    coo = csr.tocoo(copy=False)
    roles = data["roles"].cpu().numpy()
    root_ids = nodes["root_id"].to_numpy()
    weights = coo.data.astype(np.float32, copy=True)
    plastic = (roles[coo.row] == 0) & (roles[coo.col] == 2)
    init = bundle["task"]["plasticity"]["w_init"]
    weights[plastic] = initial_weights(
        root_ids[coo.row[plastic]], root_ids[coo.col[plastic]], SEED, init["low"], init["high"]
    )
    graph = data["graph"]
    original = torch.tensor(np.vstack((coo.row, coo.col)), device=graph.edge_index.device)
    require(torch.equal(graph.edge_index[:, : csr.nnz], original), "unexpected COO ordering")
    magnitude = torch.cat(
        (
            graph.gcn_norm.new_tensor(weights),
            graph.gcn_norm.new_full(
                (graph.gcn_norm.numel() - len(weights),), bundle["models"]["model_self_loop_weight"]
            ),
        )
    )
    return replace(graph, gcn_norm=magnitude * graph.message_sign)


def input_signal(bundle, data, graph_path, device):
    pairs = bundle["task"]["task"]["odor_pairs"][:2]
    features = node_features(data)
    model = fresh_model(bundle, device)
    graph = data["graph"]
    kc, mbon = data["kc"], data["mbon"]
    base = probe_pairs(representations(model, graph, features, kc, mbon), pairs)
    passed = all(row["balanced_accuracy"] >= KC_ACCURACY for row in base["kc_layer2"].values())
    pn = data["roles"] == 1
    sc, tg = graph.edge_index
    result = {
        "pass": passed,
        "chance": 0.5,
        "pn_with_nonzero_odor_drive_max": max(
            int(torch.count_nonzero(cue[pn, -1])) for cue in features
        ),
        "structural_pn_to_kc_edges": int(torch.sum(pn[sc] & (data["roles"][tg] == 0))),
        "n_kc": int(kc.numel()),
        "n_mbon": int(mbon.numel()),
        "pairs": pairs,
        "frozen_representation_probes": base,
        "ablations": {},
    }
    if not passed:
        for name, ablation_graph, inputs in (
            ("no_gcn_normalization", no_normalization(bundle, data, graph_path), features),
            ("scale_cs_contrast_x4", graph, rescale_contrast(features, pairs, CONTRAST_SCALE)),
        ):
            result["ablations"][name] = probe_pairs(
                representations(model, ablation_graph, inputs, kc, mbon), pairs
            )
    return result


def gradient_snr(bundle, data, device, directory, progress):
    model = fresh_model(bundle, device)
    optimizer = make_optimizer(model, bundle)
    norms = []

    def capture(_optimizer, _args, _kwargs):
        squared = [
            p.grad.detach().double().square().sum()
            for p in model.parameters()
            if p.grad is not None
        ]
        norms.append(float(torch.stack(squared).sum().sqrt()))

    hook = optimizer.register_step_pre_hook(capture)
    try:
        with TemporaryDirectory(prefix="snr-", dir=directory) as attempt:
            outcome = gnn_training.train_loop(
                model,
                optimizer,
                data,
                bundle["task"],
                attempt,
                fingerprint(
                    {"bundle": bundle["bundle_sha256"], "seed": SEED, "trials": SNR_TRIALS}
                ),
                progress,
                trial_limit=SNR_TRIALS,
            )
    finally:
        hook.remove()
    require(len(norms) == outcome["trials_completed"] == SNR_TRIALS, "incomplete SNR sample")
    signal = mean(norms)
    # There is no injected perturbation in the frozen primary training path.
    noise = 0.0
    ratio = noise / signal if signal > 0 else None
    return {
        "pass": ratio is not None and ratio <= SNR_RATIO_LIMIT,
        "trials": SNR_TRIALS,
        "mean_loss_gradient_l2": signal,
        "sample_sd_loss_gradient_l2": stdev(norms),
        "mean_injected_noise_l2": noise,
        "noise_to_signal": ratio,
        "last_probe": outcome["history"][-1]["discrimination"],
        "noise_reduction_run": False,
    }


def easy_task(bundle, data, device, directory, progress):
    task = bundle["task"]
    pairs = task["task"]["odor_pairs"]
    original = node_features(data)
    output = {}
    for name, features in (
        ("easy_contrast_x4", rescale_contrast(original, pairs, CONTRAST_SCALE)),
        ("restore_frozen_contrast", original),
    ):
        model = fresh_model(bundle, device)

        def predict(net, inputs, odor):
            embedding = net(inputs["features"][odor], inputs["graph"])
            return models.valence_readout(
                embedding, inputs["mbon"], inputs["avoidance"], inputs["approach"]
            )

        with patch.object(gnn_training, "predict", predict):
            outcome = gnn_training.train_loop(
                model,
                make_optimizer(model, bundle),
                {**data, "features": features},
                task,
                directory / name,
                fingerprint({"bundle": bundle["bundle_sha256"], "seed": SEED, "variant": name}),
                progress,
            )
        probes = [row["discrimination"] for row in outcome.pop("history")]
        output[name] = {**outcome, "last100_mean": mean(probes[-100:]), "max_probe": max(probes)}
        if outcome["trials_to_criterion"] is None:
            break
    return {"pass": output["easy_contrast_x4"]["trials_to_criterion"] is not None, "runs": output}


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    evidence = require_registered(ROOT)
    require(torch.cuda.is_available(), "diagnostic ladder requires the Molab CUDA environment")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    device = "cuda"
    graph_path = bundle["task"]["source_manifest"]
    prefix = bundle["bundle_sha256"][:12]
    directory = ROOT / "checkpoints" / "diagnostic_ladder" / prefix
    directory.mkdir(parents=True, exist_ok=True)
    output = ROOT / "results" / "diagnostic_ladder" / prefix / "summary.manifest.json"
    require(not output.exists(), "this diagnostic ladder already ran for the current bundle")
    config = {
        "seed": SEED,
        "synthetic": {"trial_limit": SANITY_LIMIT, "minimum_accuracy": SANITY_ACCURACY},
        "kc_probe": {
            "minimum_per_pair_balanced_accuracy": KC_ACCURACY,
            "train_views_per_cue": PROBE_TRAIN_VIEWS,
            "test_views_per_cue": PROBE_TEST_VIEWS,
            "measurement_mask_keep_probability": KEEP_PROBABILITY,
        },
        "snr": {"trial_count": SNR_TRIALS, "noise_to_signal_limit": SNR_RATIO_LIMIT},
        "easy_contrast_scale": CONTRAST_SCALE,
    }
    report = {
        "scope": "EXPLORATORY_DIAGNOSTIC_ONLY",
        "git_commit": provenance(ROOT, config)["git_commit"],
        "bundle_sha256": bundle["bundle_sha256"],
        "config_sha256": fingerprint(config),
        "config": config,
        "registration": evidence,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    with Exclusive(ROOT), Run(ROOT, bundle["experiment"], "diagnostic-ladder") as progress:
        report["sanity"] = sanity(bundle, device, directory, progress)
        if report["sanity"]["pass"]:
            data = gnn_training.frozen_inputs(
                ROOT,
                bundle["task"],
                graph_path,
                SEED,
                bundle["nulls"],
                bundle["models"],
                device,
                pilot=True,
            )
            report["kc_signal"] = input_signal(bundle, data, graph_path, device)
            report["snr"] = gradient_snr(bundle, data, device, directory, progress)
            if not (report["kc_signal"]["pass"] and report["snr"]["pass"]):
                report["easy_task"] = easy_task(bundle, data, device, directory, progress)
            else:
                report["easy_task"] = {"status": "NOT_TRIGGERED"}
        else:
            report["status"] = "STOPPED_AT_SYNTHETIC_SANITY"
        atomic_json(output, report)
    print("LADDER_DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
