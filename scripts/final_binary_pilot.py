"""Real-only release test for the proposed binary GraphSAGE-max protocol."""

import json
import os
from pathlib import Path
from time import monotonic

import polars as pl
import torch
import yaml
from phase4_pilot import load_real_cues, train

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.binary_protocol import endpoints
from mbtopology.experiment_io import cached_result, write_result
from mbtopology.paired_pilot import PILOT_SEEDS
from mbtopology.registration import require_registered
from mbtopology.runtime import (
    Exclusive,
    Run,
    atomic_json,
    fingerprint,
    provenance,
    read_json,
    require,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/final_binary.yaml"


def protocol():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        config["version"] == "binary_real_pn_v1"
        and config["models"] == ["GraphSAGE-max"]
        and tuple(config["pilot_seeds"]) == PILOT_SEEDS
        and config["pilot_pass_before"] == 1500
        and config["pairs_per_trial"] == 4
        and config["threshold"] == 0.75
        and config["rolling_window"] == 20
        and config["trials"] == 2000
        and 0 < config["checkpoint_seconds"] <= 900,
        "invalid final binary protocol",
    )
    return config


def setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    registration = require_registered(ROOT)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "need one Molab CUDA GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    config = protocol()
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    require(
        not set(config["pilot_seeds"]) & set(bundle["seeds"])
        and bundle["models"]["optimizer"]["learning_rate"] == 0.001
        and bundle["models"]["dropout"] == 0,
        "pilot seed collision or changed binary optimizer/model config",
    )
    trace = provenance(ROOT, config)
    common = {
        "legacy_bundle_sha256": bundle["bundle_sha256"],
        "bundle_sha256": fingerprint(
            {
                "prior_task_graph_and_models": bundle["bundle_sha256"],
                "final_protocol": trace["config_sha256"],
                "binary_training_path": sha256(ROOT / "scripts/phase4_pilot.py"),
                "binary_endpoints": sha256(ROOT / "src/mbtopology/binary_protocol.py"),
            }
        ),
        "config_file_sha256": sha256(CONFIG_PATH),
        "pilot_script_sha256": sha256(__file__),
        "source_sha256": trace["source_sha256"],
        "real_graph_manifest_sha256": sha256(ROOT / bundle["task"]["source_manifest"]),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    return config, bundle, trace, common, registration


def inputs_for(bundle, seed, graph):
    inputs = gnn_training.frozen_inputs(
        ROOT, bundle["task"], graph, seed, bundle["nulls"], bundle["models"], "cuda", pilot=True
    )
    inputs["base"] = models.node_features(inputs["roles"], inputs["kc"], inputs["kc"][:0])
    return inputs


def model_for(bundle, seed):
    config = bundle["models"]
    model = make_benchmark_model("GraphSAGE", config, seed, "cuda", aggregation="max")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    return model, optimizer


def main():
    config, bundle, trace, common, registration = setup()
    scope = "FINAL_BINARY_REAL_ONLY_PILOT"
    graph = bundle["task"]["source_manifest"]
    run_id = fingerprint(common)[:12]
    result_dir = ROOT / "results" / "final_binary_pilot" / run_id
    checkpoint_dir = ROOT / "checkpoints" / "final_binary_pilot" / run_id
    deadline = monotonic() + bundle["experiment"]["session_hours"] * 3600
    print("FINAL_BINARY_START", run_id, trace["git_commit"], trace["config_sha256"], flush=True)
    jobs = []
    with Exclusive(ROOT), Run(ROOT, config, "final-binary-pilot") as progress:
        for seed in PILOT_SEEDS:
            job = {
                **common,
                "scope": scope,
                "seed": seed,
                "condition": "real",
                "model": "GraphSAGE-max",
            }
            signature = fingerprint(job)
            path = result_dir / f"real-seed{seed}.parquet"
            if cached_result(path, signature, common["bundle_sha256"]):
                outcome = read_json(path.with_suffix(".manifest.json"))["outcome"]
            else:
                progress.save(seed=seed)
                drives, _, audit = load_real_cues(bundle["task"], config, seed)
                inputs = inputs_for(bundle, seed, graph)
                model, optimizer = model_for(bundle, seed)
                history, elapsed = train(
                    model,
                    optimizer,
                    inputs,
                    drives,
                    config,
                    checkpoint_dir / path.stem,
                    signature,
                    progress,
                    deadline,
                )
                outcome = {"seed": seed, **endpoints(history, config), "seconds": elapsed}
                write_result(
                    ROOT,
                    path,
                    pl.DataFrame(history),
                    {
                        **trace,
                        **job,
                        "run_config_sha256": signature,
                        "outcome": outcome,
                        "cue_protocol": audit,
                        "registration": registration,
                    },
                )
            jobs.append(outcome)
            print("FINAL_BINARY_RESULT", json.dumps(outcome), flush=True)
        passed = all(j["ttc"] is not None and j["ttc"] < config["pilot_pass_before"] for j in jobs)
        atomic_json(
            result_dir / "summary.manifest.json",
            {
                **trace,
                **common,
                "scope": scope,
                "passed": passed,
                "runs": jobs,
            },
        )
    print("FINAL_BINARY_COMPLETE", run_id, passed, flush=True)


if __name__ == "__main__":
    main()
