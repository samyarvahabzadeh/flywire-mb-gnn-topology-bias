import json

import pytest
from fsspec.implementations.local import LocalFileSystem

from mbtopology.runtime import Fail
from mbtopology.sweep_archive import ArchiveStore


def test_external_receipt_and_resume_do_not_require_scratch_state(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    original = scratch / "results.parquet"
    original.write_bytes(b"engineering fixture")
    checkpoint = scratch / "checkpoint.json"
    checkpoint.write_text('{"cursor": 17}', encoding="utf-8")
    fs = LocalFileSystem()
    store = ArchiveStore(fs, (tmp_path / "external").as_posix(), "locked-plan")
    receipt = store.commit("batch-0000", scratch, [original, checkpoint], {"jobs": ["job01"]})

    new_session = tmp_path / "replacement"
    session_store = ArchiveStore(fs, store.prefix, "locked-plan")
    assert "batch-0000.receipt.json" in session_store.list_names()
    recovered, documents = session_store.verify(
        "batch-0000", restore_root=new_session, json_suffix="checkpoint.json"
    )
    assert recovered == receipt
    assert documents == {"checkpoint.json": {"cursor": 17}}
    assert (new_session / original.name).read_bytes() == original.read_bytes()

    fs.pipe_file(session_store.path(receipt["archive"]), b"corrupted")
    with pytest.raises(Fail, match="archive hash mismatch"):
        session_store.verify("batch-0000")


def test_immutable_plan_and_receipt_refuse_mismatched_bytes(tmp_path):
    store = ArchiveStore(LocalFileSystem(), tmp_path.as_posix(), "plan1")
    store.record("frozen-plan.json", {"plan_sha256": "plan1"})
    store.record("frozen-plan.json", {"plan_sha256": "plan1"})
    with pytest.raises(Fail, match="immutable storage object differs"):
        store.record("frozen-plan.json", {"plan_sha256": "plan2"})
    assert json.loads(store.read("frozen-plan.json")) == {"plan_sha256": "plan1"}
