"""Run the fixed-horizon exploratory GraphSAGE layer intervention."""

import argparse
import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from exploratory_decoder_topology import gather_layers, graph_inputs, graphs, reduced_embeddings
from exploratory_readability import ROOT, read_plan
from phase4_pilot import binary_step, load_real_cues
from torch import nn

from mbtopology import runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.binary_protocol import add_observation
from mbtopology.experiment_io import load_training_checkpoint, save_training_checkpoint
from mbtopology.runtime import atomic_json, fingerprint, now, read_json, require, sha256

PLAN = ROOT / "configs/exploratory_layer_swap.json"


class HybridSAGE(nn.Module):
    def __init__(self, body, inputs):
        super().__init__()
        self.body = body
        self.inputs = inputs

    def forward(self, x, _graph):
        nodes, local_mbon, left, right = self.inputs["layers"]
        h = torch.relu(self.body.encoder(x))
        h = torch.relu(self.body.convs[0]((h, h[:, nodes]), left))
        mbon = torch.relu(self.body.convs[1]((h, h[:, local_mbon]), right))
        score = torch.sigmoid(self.body.head(mbon)).squeeze(-1)
        return score.new_zeros((len(x), x.shape[1])).index_copy(1, self.inputs["mbon"], score)


@dataclass
class Job:
    seed: int
    label: str
    model: HybridSAGE
    optimizer: torch.optim.Optimizer
    inputs: dict
    drives: list
    directory: Path
    signature: str
    stream: torch.cuda.Stream | None = None
    history: list = field(default_factory=list)

    def restore(self):
        saved = load_training_checkpoint(self.directory, self.signature)
        if saved is not None:
            require(
                saved["cursor"] == len(saved["history"]), "layer-swap checkpoint cursor mismatch"
            )
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.history = saved["history"]

    def save(self):
        save_training_checkpoint(
            self.directory,
            self.signature,
            {
                "cursor": len(self.history),
                "history": self.history,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
        )


def conditions(data):
    real = data["real"]
    result = {"RR": real}
    for variant in ("A", "B"):
        null = data[f"{variant}-00"]
        for code, first, final in (("NR", null, real), ("RN", real, null), ("NN", null, null)):
            layers = gather_layers(
                first["neighbors"],
                real["mbon"],
                len(real["roles"]),
                final_neighbors=final["neighbors"],
            )
            result[f"{variant}-{code}"] = {**first, "layers": layers}
    return result


def verify(model, inputs, first, final, drive, presentations):
    body = model.body
    h = reduced_embeddings(body, inputs, drive, 0, presentations)
    # Rebuild the first input block directly to compare full and reduced gradients.
    row, col = drive[:presentations].nonzero()
    device = inputs["base"].device
    x = inputs["base"].expand(presentations, -1, -1).clone()
    x[torch.as_tensor(row, device=device), inputs["kc"][torch.as_tensor(col, device=device)], 4] = (
        torch.as_tensor(drive[:presentations].data, device=device)
    )
    reference = torch.relu(body.encoder(x))
    reference = torch.relu(body.convs[0](reference, first["neighbors"]))
    reference = torch.relu(body.convs[1](reference, final["neighbors"]))[:, inputs["mbon"]]
    torch.testing.assert_close(h, reference, atol=1e-6, rtol=1e-6)
    reduced = model(x, inputs["graph"])[:, inputs["mbon"]]
    full = torch.sigmoid(body.head(reference)).squeeze(-1)
    torch.testing.assert_close(reduced, full, atol=1e-6, rtol=1e-6)
    rg = torch.autograd.grad(reduced.mean(), tuple(body.parameters()))
    fg = torch.autograd.grad(full.mean(), tuple(body.parameters()))
    for a, b in zip(rg, fg, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)


def train_wave(jobs, config, every, device):
    labels = torch.tensor([1.0, 0.0] * config["pairs_per_trial"], device=device)
    default = torch.cuda.current_stream() if device == "cuda" else None
    started = monotonic()
    for job in jobs:
        job.restore()
        if job.stream is not None:
            job.stream.wait_stream(default)
    require(all(len(j.history) <= config["trials"] for j in jobs), "checkpoint above fixed cap")
    last_save = monotonic()
    for t in range(min(len(j.history) for j in jobs), config["trials"]):
        active = [j for j in jobs if len(j.history) == t]
        queued = []
        for job in active:
            with torch.cuda.stream(job.stream) if job.stream is not None else nullcontext():
                queued.append(
                    binary_step(job.model, job.optimizer, job.inputs, job.drives, t, labels)
                )
        if default is not None:
            for job in active:
                default.wait_stream(job.stream)
        rows = torch.stack(queued).cpu().tolist()
        require(all(v[2] for v in rows), f"nonfinite loss, gradients or held-out on trial {t + 1}")
        for job, (positive, negative, _) in zip(active, rows, strict=True):
            add_observation(job.history, (positive, negative), config["rolling_window"])
        if (t + 1) % every == 0 or monotonic() - last_save >= 900:
            for job in jobs:
                job.save()
            print(f"LAYER_CHECKPOINT {t + 1} trials; {monotonic() - started:.1f}s", flush=True)
            last_save = monotonic()
    for job in jobs:
        job.save()
    return monotonic() - started


def summarize(output, trace, config, seeds, details):
    table = pl.DataFrame(trace)
    target = config["threshold"]
    endpoints = table.group_by("seed", "condition", maintain_order=True).agg(
        pl.len().alias("trials"),
        pl.col("discrimination").mean().alias("auc"),
        pl.col("trial").filter(pl.col("rolling_discrimination") >= target).min().alias("ttc"),
    )
    endpoints = endpoints.with_columns(pl.col("ttc").is_null().alias("right_censored"))
    require(
        endpoints.height == 7 * len(seeds)
        and endpoints["trials"].unique().to_list() == [config["trials"]],
        "layer-swap trace inventory mismatch",
    )
    values = {(j["seed"], j["condition"]): j["auc"] for j in endpoints.iter_rows(named=True)}
    rows = []
    for seed in seeds:
        for variant in ("A", "B"):
            rr = values[seed, "RR"]
            nn, nr, rn = (values[seed, f"{variant}-{k}"] for k in ("NN", "NR", "RN"))
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "both": nn - rr,
                    "first": nr - rr,
                    "second": rn - rr,
                    "interaction": nn - nr - rn + rr,
                }
            )
    contrasts = pl.DataFrame(rows)
    plot = output / "layer-swap.png"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", sharey=True)
    interval = 32
    for variant, ax in zip(("A", "B"), axes, strict=True):
        for code in ("RR", "NR", "RN", "NN"):
            label = code if code == "RR" else f"{variant}-{code}"
            filtered = table.filter(pl.col("condition") == label)
            y = (
                np.stack(
                    [
                        filtered.filter(pl.col("seed") == s)
                        .sort("trial")["discrimination"]
                        .to_numpy()
                        for s in seeds
                    ]
                )
                .reshape(len(seeds), -1, interval)
                .mean(2)
            )
            x = interval * np.arange(1, y.shape[1] + 1)
            ax.plot(x, np.median(y, axis=0), label=code)
            ax.fill_between(x, *np.quantile(y, (0.25, 0.75), axis=0), alpha=0.12)
        ax.axhline(0.75, color="#555555", linestyle=":", linewidth=1)
        ax.set(title=f"Null {variant}-00", xlabel="Trial")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(ylabel="Held-out discrimination")
    axes[1].legend(title="Layer 1 / layer 2", frameon=False)
    fig.savefig(plot, dpi=185, facecolor="white")
    plt.close(fig)
    lines = [
        "# Exploratory GraphSAGE layer intervention",
        "",
        "R denotes the real graph and N denotes one fixed null graph. The characters indicate the first and last GraphSAGE layer. The last layer is measured through MBONs only. The real PN→KC projection, initialization, cue stream, optimizer, and legacy fixed readout are paired within seed.",
        "",
        "The primary response is mean held-out discrimination across all 768 trials. TTC is the first 20-trial rolling mean ≥0.75; runs without that event are right-censored at 768. All 35 jobs train for the same budget without early stopping.",
        "",
        "| Null | Seed | NN−RR | NR−RR | RN−RR | Two-layer interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        effects = " | ".join(
            f"{100 * r[k]:+.3f}" for k in ("both", "first", "second", "interaction")
        )
        lines.append(f"| {r['variant']} | {r['seed']} | {effects} |")
    lines += [
        "",
        "Differences are percentage points. The interaction is NN−NR−RN+RR. The plotted lines and bands are paired-seed medians and interquartile ranges within 32-trial blocks.",
        "",
        "| Seed | Layer pair | AUC | TTC |",
        "|---:|---|---:|---:|",
    ]
    for r in endpoints.iter_rows(named=True):
        event = r["ttc"] if r["ttc"] is not None else ">768"
        lines.append(f"| {r['seed']} | {r['condition']} | {r['auc']:.5f} | {event} |")
    lines += [
        "",
        "![Layer-swap learning curves](layer-swap.png)",
        "",
        "This intervention concerns routing within two GraphSAGE layers. The null sample is one fixed wiring per variant and the five seeds replicate model initialization/cue instances. The results do not identify a specific biological connection class or estimate effects across the full null ensemble.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    table.write_parquet(output / "traces.parquet")
    endpoints.write_parquet(output / "endpoints.parquet")
    contrasts.write_parquet(output / "contrasts.parquet")
    atomic_json(output / "wall-times.json", details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(args.device == "cpu" or torch.cuda.is_available(), "requested CUDA GPU unavailable")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    start = monotonic()
    plan, (_, config) = read_json(PLAN), read_plan()
    config = {**config, "trials": plan["trials"]}
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    require(
        len(set(plan["seeds"])) == 5
        and not set(plan["seeds"]) & (set(bundle["seeds"]) | set(config["pilot_seeds"]))
        and bundle["models"]["message_passing_depth"] == 2
        and config["trials"] == 768
        and config["pairs_per_trial"] == 4
        and config["rolling_window"] == 20
        and config["threshold"] == 0.75,
        "layer-swap design differs from freeze",
    )
    paths = graphs(bundle, plan)
    data = {g: graph_inputs(bundle, path, args.device) for g, path in paths.items()}
    arms = conditions(data)
    args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / "MANIFEST.json").exists(), "layer-swap output is already complete")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    signature = {
        "scope": plan["scope"],
        "git_commit": git_commit,
        "device": args.device,
        "config_sha256": sha256(PLAN),
        "torch": str(torch.__version__),
        "graph_manifests_sha256": {g: sha256(ROOT / p) for g, p in paths.items()},
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
    }
    cues = {s: load_real_cues(bundle["task"], config, s)[0] for s in plan["seeds"]}
    jobs_to_run = iter((s, name) for s in plan["seeds"] for name in arms)
    traces, times, timing_files = [], [], []
    while wave := list(islice(jobs_to_run, plan["wave_width"])):
        jobs = []
        for seed, name in wave:
            inputs = arms[name]
            model = HybridSAGE(
                make_benchmark_model(
                    "GraphSAGE", bundle["models"], seed, args.device, aggregation="max"
                ),
                inputs,
            )
            if seed == plan["seeds"][0]:
                variant, code = name.split("-") if name != "RR" else ("A", "RR")
                first, final = [data["real"] if c == "R" else data[f"{variant}-00"] for c in code]
                verify(model, inputs, first, final, cues[seed][0], 2 * config["pairs_per_trial"])
            optim = bundle["models"]["optimizer"]
            optimizer = torch.optim.Adam(
                model.parameters(), lr=optim["learning_rate"], weight_decay=optim["weight_decay"]
            )
            jobs.append(
                Job(
                    seed,
                    name,
                    model,
                    optimizer,
                    inputs,
                    cues[seed],
                    args.output / "checkpoints" / f"{seed}-{name}",
                    fingerprint({"run": signature, "seed": seed, "condition": name}),
                    torch.cuda.Stream() if args.device == "cuda" else None,
                )
            )
        timing_path = args.output / f"wave-{len(times) + 1:02d}.json"
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        seconds = train_wave(jobs, config, plan["checkpoint_every_trials"], args.device)
        timing = read_json(timing_path)
        if timing is None:
            timing = {
                "jobs": wave,
                "this_attempt_seconds": seconds,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
            }
            atomic_json(timing_path, timing)
        timing_files.append(timing_path)
        times.append(timing)
        for job in jobs:
            traces.extend({"seed": job.seed, "condition": job.label, **r} for r in job.history)
        print(
            f"LAYER_WAVE_COMPLETE {len(times)} {len(traces) // config['trials']}/35 jobs",
            flush=True,
        )
    summarize(args.output, traces, config, plan["seeds"], times)
    files = timing_files + [
        args.output / name
        for name in (
            "REPORT.md",
            "layer-swap.png",
            "endpoints.parquet",
            "traces.parquet",
            "contrasts.parquet",
            "wall-times.json",
        )
    ]
    atomic_json(
        args.output / "MANIFEST.json",
        {
            **signature,
            "jobs": len(traces) // config["trials"],
            "trials_per_job": config["trials"],
            "seed_list": plan["seeds"],
            "attempt_wall_seconds": monotonic() - start,
            "completed_utc": now(),
            "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
        },
    )
    print(f"LAYER_COMPLETE 35 jobs {monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
