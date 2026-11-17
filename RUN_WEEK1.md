# Week-one execution

Current checkpoint (2026-11-17): the researcher confirmed `connections_princeton` and
authorized auxiliary downloads plus immediate full-source sparse graph construction.
All four products are now downloaded and converted. `data/graphs/source_full` contains
the unsigned full graph (139255 nodes, 3732460 directed pairs). The active config approval
is limited to lossless ingestion and this full source graph; MB roles, hemisphere,
sign mapping and biological acceptance bounds are still pending researcher decisions.

```powershell
.venv\Scripts\python src/week1.py profile
.venv\Scripts\python src/week1.py full-graph
.venv\Scripts\python scripts/annotation_review.py
```

`profile` reports raw row counts and exact unique pre/post IDs before pair aggregation.
`full-graph` preserves the source neuron census and all raw annotation columns without
choosing preferred MB labels or assigning E/I signs. Its original columns are retained
as PyG attributes, including `primary_type` and `additional_type(s)` separately.
The full-source self-loop policy retains source information; no loops were observed.
`graph` is the separate MB stage and still refuses missing semantic choices.

The frozen README, PREREGISTRATION and EXPERIMENT_SPEC are unchanged. No training code
or model metrics are included. `DECISIONS.md` distinguishes implemented instructions,
pending human semantic choices, and synthetic engineering fixtures.

## Windows: all large files on D

From the repository, run `powershell -File scripts/setup.ps1` (or `pwsh`). Python,
uv cache, temporary files and `.venv` are placed under `D:\icss\acc2`.
The lock uses CPU PyTorch; week-one data processing does not need CUDA.

```powershell
.venv\Scripts\python src/week1.py env
.venv\Scripts\python src/week1.py toy
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/validate.py
```

Put the FlyWire **Codex** API token in `.local/codex_token.txt`, or set
`FLYWIRE_CODEX_TOKEN` in the process environment. The CAVE token is a different credential.
Token files, raw data, checkpoints, logs and environment files are gitignored.
Authenticated URLs and server error bodies are never printed or persisted.

```powershell
.venv\Scripts\python src/week1.py catalogue
.venv\Scripts\python src/week1.py download --product consolidated_cell_types
.venv\Scripts\python src/week1.py download --product connections
.venv\Scripts\python src/week1.py inspect
```

Stop after `catalogue` if the actual connection product has a different name. Report the
catalogue to the researcher, record their choice in DECISIONS, and update only the
`remote_product` field under the stable `connections` key. Never silently substitute it.
Resource redirects must identify `/fafb/783/`; otherwise the downloader fails.
The portal's current annotations may differ from original release-time archives.

After `inspect`, send `reports/schema_inventory.json` (actual headers + raw examples)
to the researcher. Stop for schema review before converting. The researcher must set
`approval.reviewed_by`, `reviewed_at`, `decision_ids`, `inventory_sha256` and each product's
ordered `schema.<product>.header`. Put actual approved choices in DECISIONS before use.

```powershell
.venv\Scripts\python src/week1.py convert
.venv\Scripts\python src/week1.py values
```

Parquet conversion streams gzip through Arrow record batches and writes each batch with
Polars. Every CSV field remains text, including IDs and blanks. No float inference can
damage 64-bit IDs. `data/MANIFEST.json` records raw and parquet shard SHA-256, sizes and
timestamps. `values` writes unparsed label frequencies for human selection.

Ask the researcher to complete `node_universe`, `node_attributes`, `edge_columns`, exact
`roles` lists, `nt_signs`, graph aggregation/loop policy and sanity acceptance bounds.
If more annotation products are needed, record that choice, download and inspect them,
and reapprove the new inventory. Never infer PN/DAN membership from broad NT classes.

```yaml
# Structure only; replace placeholders with actual human-approved values.
node_universe: {product: YOUR_PRODUCT_KEY, id_column: YOUR_ID_COLUMN}
node_attributes:
  cell_type: {product: YOUR_PRODUCT_KEY, id_column: YOUR_ID_COLUMN, value_column: YOUR_TYPE_COLUMN}
  nt_type: {product: YOUR_PRODUCT_KEY, id_column: YOUR_ID_COLUMN, value_column: YOUR_NT_COLUMN}
```

```powershell
.venv\Scripts\python src/week1.py graph
.venv\Scripts\python src/week1.py report
```

Adjacency uses CSR rows=presynaptic, columns=postsynaptic; PyG `edge_index` follows the
same direction. Weights are integer synapse counts; signs are separate edge attributes
from the approved presynaptic NT mapping. Nodes are ordered by exact UInt64 root ID;
MB local index and whole-brain index are saved separately. The induced MB graph retains
all selected nodes, including isolated ones. A failed count/convergence check writes
`reports/sanity.json`, prints `FAIL`, and exits nonzero; it does not publish PASS outputs.
`data/graphs/MANIFEST.json` identifies completed graph artifacts and code/config hashes.

## Resume and 12-hour sessions

Run one writer per data root. Each stage writes atomic progress checkpoints and a
background checkpoint at most every 900 seconds; downloads save durable byte progress,
conversion commits each parquet shard, graph stages reuse verified artifacts, and CSR
construction periodically saves its sparse matrix. A heartbeat records progress already
committed; it cannot capture the internals of a running native sort or file serializer.
An interrupted native operation replays from the last completed stage. No whole-brain
dense matrix is allocated. gzip conversion resumes by replaying the compressed stream
and skipping verified committed batches (gzip is not randomly seekable).

Rerun the same command/config/root after interruption. Downloads use HTTP Range and
If-Range when the source supplies a stable validator; otherwise they restart safely.
Input/config/code changes fail resume rather than mix artifacts. After a hard session
death, inspect `checkpoints/pipeline.lock`, verify its process is gone, then remove that
one lock file. Do not delete raw data or checkpoints to make a scientific check pass.
Keep the repository/data root on persistent molab storage, not an ephemeral session path.

## molab / marimo

Use Python 3.11+ and run from a persistent project directory:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR"
uv sync --python 3.11 --extra dev --frozen
uv run python src/week1.py env
uv run python src/week1.py toy
```

The same `.py` entry point can be called from a marimo cell with
`subprocess.run([sys.executable, "src/week1.py", "inspect"], check=True)` after installing
the project into that interpreter. There is no notebook-only state.

For PyG reload, use `torch_geometric.io.fs.torch_load` or an explicitly trusted local
`torch.load(path, weights_only=False)`. Never unpickle an untrusted artifact.

## Google Drive

Connected account identity was verified as `s.rasad.793@gmail.com`. Use the connected
Drive as a backup destination for reviewable artifacts; inspect any quota failure before
deciding which specific shared folder to leave. The source bundle excludes `.local`,
tokens, environments, caches, temporary files and checkpoints.
Local D space is checked before download and conversion; scientific files are never
deleted automatically to recover space.
