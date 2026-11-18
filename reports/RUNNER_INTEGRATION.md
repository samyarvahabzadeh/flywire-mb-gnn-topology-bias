# Follow-up runner integration audit — 2026-11-18

Full suite: **41 passed**. All new integration executions were temporary, explicitly
synthetic unit fixtures; no FlyWire model learning, actual OSF registration or real/null
scientific comparison occurred.

The integration tests exercise GCN and GAT through the real runner execute/results path,
including two toy trials, graph/config/registration hash checks, result parquet plus JSON
manifest, right-censoring metadata and verified skipping of a completed fixture.

An audit identified a gap in direct programmatic use: a caller could mutate a bundle
object after hashing and retain the previous hash string. Planning, runner execution and
the GNN entry point now recompute the bundle-content hash before proceeding. A test changes
the learning rate while preserving the old hash and verifies rejection before execution.

Scientific defaults were not changed. The implementation change intentionally produces
a new plan/code bundle identity. Production results and generated real null graphs are
still absent. Git pushes are now explicitly authorized at each 45-minute work checkpoint.
