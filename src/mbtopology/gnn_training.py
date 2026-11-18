from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import models
from .experiment_io import (
    SessionYield,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
    verify_bundle,
)
from .nulls import signs_from_nt
from .registration import require_registered
from .runtime import fingerprint, read_json, require, sha256
from .task_inputs import initial_weights, load_source, make_patterns, node_identity, schedule


def frozen_inputs(
    root, task_config, graph_manifest, seed, null_config, model_config, device, *, pilot=False
):
    root = Path(root)
    _, nodes, matrix = load_source(root, graph_manifest)
    report = read_json(root / "reports/task_inputs.json")
    require(
        report and report["task_config_sha256"] == fingerprint(task_config),
        "task inputs/config are not frozen together",
    )
    require(
        node_identity(nodes) == report["canonical_node_identity_sha256"],
        "FAIRNESS FAIL: condition changed node identities/annotations",
    )
    for path, key in (
        (task_config["cs_encoding"]["odor_sets"], "odor_sets_sha256"),
        (task_config["readout"]["frozen_membership"], "readout_sha256"),
        (task_config["seeds_file"], "seeds_sha256"),
    ):
        require(sha256(root / path) == report[key], "FAIRNESS FAIL: frozen input hash changed")
    odors = read_json(root / task_config["cs_encoding"]["odor_sets"])
    readout = read_json(root / task_config["readout"]["frozen_membership"])
    require(
        (pilot and seed == 1337 and str(seed) not in odors["seeds"])
        or (not pilot and str(seed) in odors["seeds"]),
        "invalid seed for pilot or frozen input set",
    )
    role_values = nodes["role"].to_list()
    role_map = {"KC": 0, "PN": 1, "MBON": 2, "DAN": 3}
    require(set(role_values) <= set(role_map), "unknown model role")
    role_ids = np.array([role_map[value] for value in role_values], dtype=np.int64)
    kc, mbon = np.flatnonzero(role_ids == 0), np.flatnonzero(role_ids == 2)
    ids = nodes["root_id"].to_numpy()
    require(
        [str(x) for x in ids[kc]] == odors["kc_root_ids"]
        and [str(x) for x in ids[mbon]] == readout["mbon_root_ids"],
        "frozen neuron ordering changed",
    )
    coo = matrix.tocoo(copy=False)
    magnitudes = coo.data.astype(np.float32, copy=True)
    plastic = (role_ids[coo.row] == 0) & (role_ids[coo.col] == 2)
    init = task_config["plasticity"]["w_init"]
    magnitudes[plastic] = initial_weights(
        ids[coo.row[plastic]], ids[coo.col[plastic]], seed, init["low"], init["high"]
    )
    node_signs = signs_from_nt(nodes["nt_type"].to_list(), null_config["nt_signs"])
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    graph = models.prepare_graph(
        edge_index,
        torch.tensor(magnitudes, device=device),
        torch.tensor(node_signs[coo.row], device=device),
        len(nodes),
        model_config["model_self_loop_weight"],
    )
    patterns = odors["seeds"].get(str(seed))
    if pilot:
        encoding = task_config["cs_encoding"]
        patterns = make_patterns(
            len(kc),
            task_config["task"]["n_odors"],
            encoding["kc_sparsity"],
            seed,
            encoding["odor_seed_stream"],
        )
    return {
        "graph": graph,
        "roles": torch.tensor(role_ids, device=device),
        "kc": torch.tensor(kc, device=device),
        "mbon": torch.tensor(mbon, device=device),
        "avoidance": torch.tensor(readout["avoidance_indices"], device=device),
        "approach": torch.tensor(readout["approach_indices"], device=device),
        "patterns": [torch.tensor(x, dtype=torch.long, device=device) for x in patterns],
        "schedule": schedule(task_config, seed),
        "input_report": report,
    }


def predict(model, data, odor):
    features = models.node_features(data["roles"], data["kc"], data["patterns"][odor])
    outputs = model(features, data["graph"])
    value = models.valence_readout(outputs, data["mbon"], data["avoidance"], data["approach"])
    require(bool(torch.isfinite(value)), "nonfinite model readout")
    return value


def probe(model, data, task_config):
    model.eval()
    with torch.no_grad():
        values = np.array(
            [
                float(predict(model, data, odor).cpu())
                for odor in range(task_config["task"]["n_odors"])
            ]
        )
    plus, minus = task_config["task"]["cs_plus"], task_config["task"]["cs_minus"]
    return float(0.5 * (values[plus].mean() + (1 - values[minus]).mean()))


def train_loop(
    model,
    optimizer,
    data,
    task_config,
    checkpoint_dir,
    signature,
    run,
    checkpoint_seconds=900,
    deadline=None,
    trial_limit=None,
    record_trial_times=False,
):
    trials_max = task_config["task"]["trials_max"]
    if trial_limit is not None:
        require(0 < trial_limit <= trials_max, "invalid smoke trial limit")
        trials_max = trial_limit
    cursor, history, streak, reached, elapsed = 0, [], 0, None, 0.0
    saved = load_training_checkpoint(checkpoint_dir, signature)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        cursor, history, streak, reached, elapsed = (
            saved[key] for key in ("cursor", "history", "streak", "reached_at", "elapsed_seconds")
        )
        restore_rng(saved["rng"])
    require(
        cursor == len(history) and cursor <= trials_max,
        "invalid resumed trial cursor",
    )
    started, last_save = time.monotonic(), time.monotonic()

    def checkpoint():
        save_training_checkpoint(
            checkpoint_dir,
            signature,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cursor": cursor,
                "history": history,
                "streak": streak,
                "reached_at": reached,
                "elapsed_seconds": elapsed + time.monotonic() - started,
            },
        )
        if run:
            run.save(completed_trials=cursor)

    checkpoint()
    in_trial = False
    try:
        while cursor < trials_max and reached is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise SessionYield("session safety deadline reached; resume the same command")
            in_trial = True
            trial_started = time.perf_counter()
            odor = int(data["schedule"][cursor])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            value = predict(model, data, odor)
            is_plus = odor in task_config["task"]["cs_plus"]
            target = value.new_tensor(float(is_plus))
            loss = F.binary_cross_entropy(value, target)
            require(bool(torch.isfinite(loss)), "nonfinite model training loss")
            loss.backward()
            require(
                all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ),
                "nonfinite model gradient",
            )
            optimizer.step()
            score = probe(model, data, task_config)
            streak = streak + 1 if score >= task_config["readout"]["criterion"] else 0
            cursor += 1
            if streak >= task_config["readout"]["consecutive_trials"]:
                reached = cursor
            row = {
                "trial": cursor,
                "odor_id": odor,
                "us": int(is_plus),
                "training_loss": float(loss.detach().cpu()),
                "discrimination": score,
                "criterion_streak": streak,
            }
            if record_trial_times:
                if data["roles"].is_cuda:
                    torch.cuda.synchronize(data["roles"].device)
                row["trial_seconds"] = time.perf_counter() - trial_started
            history.append(row)
            in_trial = False
            if time.monotonic() - last_save >= checkpoint_seconds:
                checkpoint()
                last_save = time.monotonic()
    finally:
        if not in_trial:
            checkpoint()
    return {
        "history": history,
        "trials_completed": cursor,
        "trials_to_criterion": reached,
        "right_censored": reached is None,
        "elapsed_seconds": elapsed + time.monotonic() - started,
    }


def run_registered(
    root,
    bundle,
    run_config,
    graph_manifest,
    device,
    progress,
    deadline,
    *,
    smoke_trials=None,
    rate_measurement=False,
):
    task_config, model_config, null_config = bundle["task"], bundle["models"], bundle["nulls"]
    require_registered(root)
    verify_bundle(bundle)
    seed = run_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = models.make_model(run_config["architecture"], model_config, seed, device)
    data = frozen_inputs(root, task_config, graph_manifest, seed, null_config, model_config, device)
    optimizer_config = model_config["optimizer"]
    require(optimizer_config["name"] == "Adam", "unsupported frozen optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    signature = fingerprint(
        {
            "run": run_config,
            "graph_sha256": sha256(Path(root) / graph_manifest),
            "device": device,
            "torch_version": torch.__version__,
            "implementation_sha256": sha256(__file__),
            "smoke_trials": smoke_trials,
            "rate_measurement": rate_measurement,
        }
    )
    scope = "molab_rate" if rate_measurement else ("gnn" if smoke_trials is None else "smoke")
    directory = Path(root) / "checkpoints" / scope / bundle["bundle_sha256"][:12]
    directory /= run_config["run_id"]
    result = train_loop(
        model,
        optimizer,
        data,
        task_config,
        directory,
        signature,
        progress,
        model_config["checkpoint_seconds"],
        deadline,
        trial_limit=smoke_trials,
        record_trial_times=rate_measurement,
    )
    result["parameter_count"] = models.parameter_count(model)
    return result
