"""Build the Molab payload from the audited real graph and frozen input controls."""

import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from exploratory_readability import ROOT, read_plan, real_projection
from scipy import sparse

from mbtopology.runtime import atomic_json, read_json, require, sha256

out = ROOT / "dist/differential-molab"
out.mkdir(parents=True, exist_ok=True)
old, _ = read_plan()
nodes, _, kc, projection = real_projection(old["source"])
sparse.save_npz(out / "projection.npz", projection)
mroot = ROOT / "results/exploratory/memory_generalization_v1"
catalog = mroot / "input-masks.json"
require(
    sha256(catalog) == read_json(mroot / "MANIFEST.json")["files"][catalog.name]["sha256"],
    "bad source masks",
)
raw = read_json(catalog)["masks"]
kc_index = {str(nodes["root_id"][int(k)]): i for i, k in enumerate(kc)}
outputs = old["plasticity"]["mbon_ids"]
arrays = {}
for key, value in raw.items():
    arrays[key + "_pre"] = np.asarray([kc_index[s] for s in value["presynaptic_roots"]])
    arrays[key + "_post"] = np.asarray([outputs.index(s) for s in value["postsynaptic_roots"]])
    assert len(arrays[key + "_pre"]) == 1638
np.savez_compressed(out / "masks.npz", **arrays)
shutil.copyfile(ROOT / "configs/exploratory_differential.json", out / "plan.json")
shutil.copyfile(ROOT / "scripts/molab_differential.py", out / "run.py")
files = ["projection.npz", "masks.npz", "plan.json", "run.py"]
atomic_json(
    out / "bundle.json",
    {
        "files": {p: sha256(out / p) for p in files},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_masks_sha256": sha256(catalog),
        "real_graph_manifest_sha256": sha256(ROOT / old["source"]),
    },
)
archive = out.with_suffix(".zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for name in [*files, "bundle.json"]:
        z.write(out / name, name)
print(archive, sha256(archive))
