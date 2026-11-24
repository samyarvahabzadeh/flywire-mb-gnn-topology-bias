"""Compare two matched rewiring controls at each GraphSAGE layer."""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import torch
from exploratory_decoder_topology import graph_inputs
from exploratory_layer_bank import run_panel
from exploratory_layer_swap import conditions
from exploratory_readability import ROOT, read_plan
from scipy import sparse

from mbtopology import runner
from mbtopology.nulls import DirectedSwapChain, signs_from_nt
from mbtopology.runtime import atomic_file, atomic_json, now, read_json, require, sha256
from mbtopology.sweep_archive import ArchiveStore
from mbtopology.task_inputs import load_source

PLAN = ROOT / "configs/exploratory_role_layers.json"


class RoleSwap(DirectedSwapChain):
    """Swap targets within a fixed (source role, target role, source sign) stratum."""

    def __init__(self, matrix, signs, roles, seed):
        super().__init__(matrix, signs, "B", seed)
        self.roles = np.asarray(roles)
        keys = np.column_stack((self.roles[self.pre], self.roles[self.post], self.signs))
        _, self.strata = np.unique(keys, axis=0, return_inverse=True)
        self.strata = self.strata.reshape(-1)
        self.groups = [np.flatnonzero(self.strata == k) for k in range(self.strata.max() + 1)]
        self.signature = hashlib.sha256(self.signature.encode() + self.roles.tobytes()).hexdigest()

    def step(self):
        self.attempts += 1
        i = int(self.rng.integers(len(self.pre)))
        group = self.groups[self.strata[i]]
        j = group[self.rng.integers(len(group))]
        a, b, c, d = int(self.pre[i]), int(self.post[i]), int(self.pre[j]), int(self.post[j])
        if a == c or b == d or a == d or c == b:
            return False
        first, second = a * self.n + d, c * self.n + b
        if first in self.edges or second in self.edges:
            return False
        self.edges.difference_update((a * self.n + b, c * self.n + d))
        self.edges.update((first, second))
        self.post[i], self.post[j] = d, b
        self.accepted += 1
        return True

    def validate(self):
        cert = super().validate()
        n_roles = int(self.roles.max()) + 1
        source_group = self.roles[self.pre] * 3 + self.signs + 1
        before = self.original_post * (3 * n_roles) + source_group
        after = self.post * (3 * n_roles) + source_group
        require(np.array_equal(np.sort(before), np.sort(after)), "incoming role/sign counts differ")
        require(
            np.array_equal(self.roles[self.original_post], self.roles[self.post]),
            "outgoing target role per signed edge slot differs",
        )
        return {**cert, "variant": "S", "signed_role_degrees_per_neuron": True}


def make_graphs(bundle, plan):
    real = bundle["task"]["source_manifest"]
    source, nodes, matrix = load_source(ROOT, real)
    names, roles = np.unique(nodes["role"].to_numpy().astype(str), return_inverse=True)
    require(names.tolist() == plan["roles"], "role inventory differs")
    edge_signs = signs_from_nt(nodes["nt_type"].to_list(), bundle["nulls"]["nt_signs"])[
        matrix.tocoo().row
    ]
    paths, certificates = {"real": real}, {}
    for donor in plan["donors"]:
        for kind in ("B", "S"):
            label = f"{kind}-{donor}"
            folder = ROOT / "data" / plan["scope"] / label
            path = folder / "MANIFEST.json"
            if not path.exists():
                chain = (
                    RoleSwap(matrix, edge_signs, roles, donor)
                    if kind == "S"
                    else DirectedSwapChain(matrix, edge_signs, "B", donor)
                )
                target = plan["accepted_swaps_per_edge"] * matrix.nnz
                checkpoint = ROOT / "checkpoints" / plan["scope"] / label
                chain.restore(checkpoint)
                cert = chain.run(
                    target,
                    target * bundle["nulls"]["max_attempts_per_accepted_swap"],
                    checkpoint,
                    900,
                )
                csr = folder / "null_csr.npz"
                generated = chain.as_csr()
                atomic_file(csr, lambda p, graph=generated: sparse.save_npz(p, graph))
                atomic_json(
                    path,
                    {
                        "status": "PASS",
                        "config_sha256": sha256(PLAN),
                        "source_manifest": real,
                        "source_sha256": sha256(ROOT / real),
                        "instance": donor,
                        "variant": kind,
                        "report": cert,
                        "artifacts": {
                            "nodes": source["artifacts"]["nodes"],
                            "csr": {
                                "path": csr.relative_to(ROOT).as_posix(),
                                "sha256": sha256(csr),
                                "bytes": csr.stat().st_size,
                            },
                        },
                    },
                )
            paths[label] = path.relative_to(ROOT).as_posix()
            spec, _, _ = load_source(ROOT, paths[label])
            cert = spec["report"]
            require(
                spec["config_sha256"] == sha256(PLAN)
                and spec["source_sha256"] == sha256(ROOT / real)
                and spec["instance"] == donor
                and cert["variant"] == kind
                and cert["exact_signed_degree"]
                and cert["accepted_swaps"] == plan["accepted_swaps_per_edge"] * matrix.nnz
                and (kind == "B" or cert["signed_role_degrees_per_neuron"])
                and 1 - cert["edge_overlap_fraction"] >= plan["minimum_fraction_edges_replaced"],
                f"invalid or insufficiently rewired graph {label}",
            )
            certificates[label] = cert
            print(
                f"ROLE_GRAPH {label} replaced={1 - cert['edge_overlap_fraction']:.4f}", flush=True
            )
    require(
        len({read_json(ROOT / p)["artifacts"]["csr"]["sha256"] for p in paths.values()}) == 7,
        "duplicate sampled graphs",
    )
    return paths, certificates


def summarize(output, plan, run, certs):
    frames = [pl.read_parquet(output / f"panel-{i:02d}/traces.parquet") for i in range(1, 6)]
    traces = pl.concat(frames)
    require(
        traces.height == 65 * plan["trials"] and traces["discrimination"].is_finite().all(),
        "invalid final trial inventory",
    )
    means = {
        (s, c): v
        for s, c, v in traces.group_by("seed", "condition")
        .agg(pl.col("discrimination").mean())
        .iter_rows()
    }
    rows = []
    for s in plan["seeds"]:
        for donor in plan["donors"]:
            row = {"seed": s, "donor": donor, "rr": means[s, "RR"]}
            for kind in ("B", "S"):
                for pair in ("NR", "RN"):
                    row[f"{kind}_{pair}"] = means[s, f"{kind}-{donor}-{pair}"]
                row[f"{kind}_gap_pp"] = 100 * (row[f"{kind}_NR"] - row[f"{kind}_RN"])
            row["attenuation_pp"] = row["B_gap_pp"] - row["S_gap_pp"]
            rows.append(row)
    paired = pl.DataFrame(rows)
    medians = (
        paired.group_by("donor")
        .agg(pl.col("B_gap_pp", "S_gap_pp", "attenuation_pp").median())
        .sort("donor")
    )
    for name, frame in (("traces", traces), ("paired", paired), ("donor_medians", medians)):
        atomic_file(output / f"{name}.parquet", frame.write_parquet)
    files = [output / f"{name}.parquet" for name in ("traces", "paired", "donor_medians")]
    manifest = output / "MANIFEST.json"
    atomic_json(
        manifest,
        {
            **run,
            "completed": now(),
            "jobs": 65,
            "certificates": certs,
            "files": {p.name: sha256(p) for p in files},
        },
    )
    print("ROLE_RESULT", medians.to_dicts(), flush=True)
    return [*files, manifest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graphs-only", action="store_true")
    parser.add_argument("--drive-prefix")
    args = parser.parse_args()
    plan = read_json(PLAN)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    paths, certs = make_graphs(bundle, plan)
    if args.graphs_only:
        return
    require(torch.cuda.is_available(), "CUDA unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    _, cfg = read_plan()
    cfg = {**cfg, "trials": plan["trials"]}
    require(
        not set(plan["seeds"]) & (set(bundle["seeds"]) | set(cfg["pilot_seeds"])), "seed overlap"
    )
    data = {key: graph_inputs(bundle, path, "cuda") for key, path in paths.items()}
    arms = {k: v for k, v in conditions(data, retain_null_id=True).items() if not k.endswith("-NN")}
    require(len(arms) == 13 and len(plan["seeds"]) == 5, "role study inventory differs")
    run = {
        "scope": plan["scope"],
        "config_sha256": sha256(PLAN),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "graphs": {k: sha256(ROOT / p) for k, p in paths.items()},
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = None
    if args.drive_prefix:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(
            use_listings_cache=False,
            skip_instance_cache=True,
            auth_kwargs={"use_local_webserver": False},
        )
        store = ArchiveStore(fs, args.drive_prefix, sha256(PLAN))
        store.record("run.json", run)
    state = plan, cfg, data, arms, bundle, run, output
    for number, seed in enumerate(plan["seeds"], 1):
        panel = run_panel(number, [seed], state, float("inf"), store)
        if store:
            store.verify(f"panel-{number:02d}")
        print("ROLE_PANEL_COMPLETE", number, panel["jobs"], flush=True)
    files = summarize(output, plan, run, certs)
    if store:
        store.commit("final", output, files, {"jobs": 65})
    print("ROLE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
