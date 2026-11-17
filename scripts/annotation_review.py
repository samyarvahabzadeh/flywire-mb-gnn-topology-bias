"""Reproduce raw annotation frequencies without assigning mushroom-body roles."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mbtopology.runtime import atomic_json


def main():
    root = Path(__file__).resolve().parents[1]
    nodes = pl.read_parquet(root / "data/graphs/source_full/nodes.parquet")
    report = {
        "scope": "Raw annotation frequencies only; no MB membership, hemisphere or sign mapping applied",
        "class_by_side": nodes.group_by("class", "side").len().sort("class", "side").to_dicts(),
        "class_by_primary_type_and_sub_class": nodes.group_by("class", "primary_type", "sub_class")
        .len()
        .sort("class", "primary_type", "sub_class")
        .to_dicts(),
        "nt_type": nodes.group_by("nt_type").len().sort("nt_type").to_dicts(),
    }
    atomic_json(root / "reports/annotation_review.json", report)
    print(
        json.dumps(
            {"status": "raw annotations reported", "report": "reports/annotation_review.json"}
        )
    )


if __name__ == "__main__":
    main()
