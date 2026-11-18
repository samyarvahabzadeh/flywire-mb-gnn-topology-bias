"""Run the four authorized real-graph paired-cue pilots on one Molab GPU."""

import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]


def audit_noise(data, seed, config):
    import torch

    from mbtopology.models import node_features
    from mbtopology.robustness_pilot import perturb_cue
    from mbtopology.runtime import require

    require(config["scope"] == "PROPOSED_EXPLORATORY_ONLY", "robustness pilot scope changed")
    rows = []
    n_kc = data["kc"].numel()
    for level in config["levels"]:
        norms = []
        for cue, active in enumerate(data["patterns"]):
            injected = perturb_cue(active.cpu().numpy(), n_kc, level, seed=seed, cue=cue)
            clean = node_features(data["roles"], data["kc"], active)
            noisy = node_features(data["roles"], data["kc"], active.new_tensor(injected))
            norm = float(torch.linalg.vector_norm(noisy - clean))
            require((norm > 0) == (level > 0), "robustness noise unexpectedly zero or nonzero")
            norms.append(norm)
        rows.append(
            {"level": level, "mean_feature_delta_l2": sum(norms) / len(norms), "min_l2": min(norms)}
        )
    return rows


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import polars as pl
    import torch
    import yaml

    from mbtopology import gnn_training, runner
    from mbtopology.benchmark_models import ARCHITECTURES, make_benchmark_model
    from mbtopology.experiment_io import cached_result, write_result
    from mbtopology.paired_pilot import PILOT_SEEDS, learning_trend, paired_schedule, train_paired
    from mbtopology.registration import require_registered
    from mbtopology.runtime import (
        Exclusive,
        Run,
        atomic_json,
        fingerprint,
        provenance,
        require,
        sha256,
    )

    evidence = require_registered(ROOT)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "attach a Molab GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    task, model_cfg = bundle["task"], bundle["models"]
    require(
        task["task"]["trials_max"] == 2000
        and task["readout"]["criterion"] == 0.8
        and task["readout"]["consecutive_trials"] == 20
        and ARCHITECTURES == ("GCN", "GraphSAGE")
        and not set(PILOT_SEEDS) & set(bundle["seeds"]),
        "paired pilot constraints differ from the authorized run",
    )
    graph = task["source_manifest"]
    robustness = yaml.safe_load((ROOT / "configs/robustness_pilot.yaml").read_text())
    prefix = bundle["bundle_sha256"][:12]
    output = ROOT / "results" / "paired_pilot" / prefix
    code_hash = sha256(__file__)
    deadline = monotonic() + bundle["experiment"]["session_hours"] * 3600
    all_reports, noise_audit = [], {}
    with Exclusive(ROOT), Run(ROOT, bundle["experiment"], "paired-pilot") as progress:
        for seed in PILOT_SEEDS:
            data = gnn_training.frozen_inputs(
                ROOT, task, graph, seed, bundle["nulls"], model_cfg, "cuda", pilot=True
            )
            pairs = paired_schedule(task, seed)
            noise_audit[str(seed)] = audit_noise(data, seed, robustness)
            for arch in ARCHITECTURES:
                spec = {
                    "seed": seed,
                    "architecture": arch,
                    "condition": "real",
                    "training_step": "one_adam_step_on_mean_bce_of_two_opposite_cues",
                    "pair_table_sha256": fingerprint(pairs.tolist()),
                    "schedule_scope": "same_seeded_balanced_cue_blocks_grouped_by_pair",
                    "bundle_sha256": bundle["bundle_sha256"],
                    "pilot_script_sha256": code_hash,
                    "graph_sha256": sha256(ROOT / graph),
                    "gpu": torch.cuda.get_device_name(),
                    "cuda": torch.version.cuda,
                    "torch": str(torch.__version__),
                }
                signature = fingerprint(spec)
                stem = f"{arch}-real-seed{seed}"
                artifact = output / f"{stem}.parquet"
                if cached_result(artifact, signature, bundle["bundle_sha256"]):
                    manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
                    all_reports.append(manifest["paired_pilot_report"])
                    continue
                progress.save(pilot=stem)
                print("PAIRED_START", stem, flush=True)
                model = make_benchmark_model(arch, model_cfg, seed, "cuda")
                opt = model_cfg["optimizer"]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"]
                )
                history, criterion, elapsed = train_paired(
                    model,
                    optimizer,
                    data,
                    task,
                    pairs,
                    ROOT / "checkpoints" / "paired_pilot" / prefix / stem,
                    signature,
                    progress,
                    deadline,
                )
                trend = learning_trend(history)
                count = len(history)
                require(criterion == count or count == 2000, "paired job did not finish")
                counts = Counter(int(cue) for pair in pairs[:count] for cue in pair)
                report = {
                    **spec,
                    "config_sha256": signature,
                    "trials": count,
                    "cue_presentations": 2 * count,
                    "presentation_counts": dict(sorted(counts.items())),
                    "criterion_trial": criterion,
                    "passed": criterion is not None and trend["rising"],
                    "elapsed_seconds": elapsed,
                    **trend,
                }
                trace = provenance(ROOT, spec)
                write_result(
                    ROOT,
                    artifact,
                    pl.DataFrame(history),
                    {
                        **trace,
                        **spec,
                        "scope": "EXPLORATORY_PAIRED_PILOT_ONLY",
                        "run_config_sha256": signature,
                        "paired_pilot_report": report,
                        "registration": evidence,
                    },
                )
                all_reports.append(report)
                print("PAIRED_RESULT", json.dumps(report), flush=True)

        require(len(all_reports) == 4, "incomplete paired pilot matrix")
        trace = provenance(ROOT, {"kind": "paired_pilot", "runs": all_reports})
        summary = {
            "scope": "EXPLORATORY_PAIRED_PILOT_ONLY",
            "status": "COMPLETE",
            "all_four_passed": all(report["passed"] for report in all_reports),
            "runs": all_reports,
            "noise_audit": noise_audit,
            "robustness_config_file_sha256": sha256(ROOT / "configs/robustness_pilot.yaml"),
            "git_commit": trace["git_commit"],
            "config_sha256": trace["config_sha256"],
            "bundle_sha256": bundle["bundle_sha256"],
        }
        atomic_json(output / "summary.manifest.json", summary)
    print("PAIRED_COMPLETE", prefix, summary["all_four_passed"], flush=True)


if __name__ == "__main__":
    main()
