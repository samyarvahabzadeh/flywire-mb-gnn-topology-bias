"""Noncomparative seed-1337 pilot on the already materialized null-A instance 0."""

import argparse
import os
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PILOT_SEED = 1337
RATES = (0.001, 0.003, 0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    from mbtopology import gnn_training, models, runner
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

    evidence = require_registered(ROOT)
    require(torch.cuda.is_available(), "pilot requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    graph = Path(args.graph_manifest)
    require(not graph.is_absolute() and ".." not in graph.parts, "use a local relative manifest")
    graph_record = read_json(ROOT / graph)
    require(
        graph_record
        and graph_record["status"] == "PASS"
        and graph_record["variant"] == "A"
        and graph_record["instance"] == 0,
        "pilot requires verified null-A instance 0",
    )
    model_cfg, task = bundle["models"], bundle["task"]
    require(model_cfg["optimizer"]["learning_rate"] == RATES[0], "pilot baseline changed")
    provenance_record = provenance(ROOT, {"seed": PILOT_SEED, "learning_rates": RATES})
    prefix = bundle["bundle_sha256"][:12]
    output = ROOT / "results" / "learnability_pilot" / prefix
    data = gnn_training.frozen_inputs(
        ROOT, task, graph, PILOT_SEED, bundle["nulls"], model_cfg, "cuda", pilot=True
    )
    deadline = time.monotonic() + bundle["experiment"]["session_hours"] * 3600
    with Exclusive(ROOT), Run(ROOT, bundle["experiment"], "learnability-pilot") as progress:
        for rate in RATES:
            config = {
                "seed": PILOT_SEED,
                "learning_rate": rate,
                "graph_manifest_sha256": sha256(ROOT / graph),
                "bundle_sha256": bundle["bundle_sha256"],
                "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
            }
            signature = fingerprint(config)
            report_path = output / f"adam-{rate}.manifest.json"
            previous = read_json(report_path)
            if previous is None:
                model = models.make_model("GCN", model_cfg, PILOT_SEED, "cuda")
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=rate, weight_decay=model_cfg["optimizer"]["weight_decay"]
                )
                outcome = gnn_training.train_loop(
                    model,
                    optimizer,
                    data,
                    task,
                    ROOT / "checkpoints" / "learnability_pilot" / prefix / f"adam-{rate}",
                    signature,
                    progress,
                    deadline=deadline,
                )
                scores = [row["discrimination"] for row in outcome.pop("history")]
                previous = {
                    **config,
                    **outcome,
                    "scope": "EXPLORATORY_SINGLE_GRAPH_PILOT_1337",
                    "git_commit": provenance_record["git_commit"],
                    "config_sha256": signature,
                    "last100_mean": mean(scores[-100:]),
                    "max_discrimination": max(scores),
                    "registration": evidence,
                }
                atomic_json(report_path, previous)
            require(previous["config_sha256"] == signature, "pilot manifest/config mismatch")
            import json

            print("PILOT_RESULT", json.dumps(previous), flush=True)
    print("PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
