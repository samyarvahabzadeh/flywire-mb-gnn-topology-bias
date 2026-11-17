from __future__ import annotations

import csv
import gzip
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import requests

from .runtime import (
    Fail,
    atomic_file,
    atomic_json,
    disk_guard,
    fingerprint,
    now,
    read_json,
    require,
    sha256,
)

BASE = "https://codex.flywire.ai"


def clean_url(url):
    pieces = urlsplit(url)
    return urlunsplit((pieces.scheme, pieces.netloc, pieces.path, "", ""))


def read_token(token_file):
    token = os.environ.get("FLYWIRE_CODEX_TOKEN", "").strip()
    if not token and Path(token_file).is_file():
        token = Path(token_file).read_text(encoding="utf-8-sig").strip()
    require(
        token and not any(c.isspace() for c in token),
        "Codex token missing/invalid; use FLYWIRE_CODEX_TOKEN or .local/codex_token.txt",
    )
    return token


class ProductLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.products = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "href" and value:
                self.products.update(parse_qs(urlsplit(value).query).get("data_product", []))


def request(url, **kwargs):
    try:
        response = requests.get(url, timeout=(30, 60), **kwargs)
        require(
            response.status_code in (200, 206),
            f"Codex HTTP {response.status_code}; response body and authenticated URL withheld",
        )
        return response
    except requests.RequestException as exc:
        raise Fail(f"FAIL: network request failed ({type(exc).__name__}); URL withheld") from None


def catalogue(root, config, token):
    with request(
        BASE + "/api/download",
        params={"dataset": "fafb", "data_version": "783", "api_token": token},
    ) as response:
        parser = ProductLinks()
        parser.feed(response.text)
    report = {
        "created_at": now(),
        "dataset": "fafb",
        "snapshot_requested": "783",
        "products": sorted(parser.products),
        "status": "metadata_only",
        "endpoint": BASE + "/api/download",
    }
    atomic_json(Path(root) / "reports" / "catalogue.json", report)
    require(
        parser.products, "no product links found; catalogue structure/authentication needs review"
    )
    return report


def manifest_update(root, key, value):
    path = Path(root) / "data" / "MANIFEST.json"
    manifest = read_json(
        path, {"manifest_version": 1, "dataset": "fafb", "snapshot": "783", "artifacts": {}}
    )
    require(
        manifest["dataset"] == "fafb" and manifest["snapshot"] == "783",
        "manifest dataset/snapshot conflict",
    )
    manifest["artifacts"][key] = value
    manifest["updated_at"] = now()
    atomic_json(path, manifest)


def record_file(root, path, **extra):
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256(path),
        "recorded_at": now(),
        **extra,
    }


def finalize_download(root, product, path, partial, resume_path, info):
    require(partial.stat().st_size == info["total_bytes"], "download length mismatch")
    with partial.open("rb") as stream:
        require(stream.read(2) == b"\x1f\x8b", "download is not gzip CSV")
    item = record_file(
        root, partial, **info, checksum_scope="local bytes; no publisher SHA supplied"
    )
    item["path"] = path.relative_to(root).as_posix()
    atomic_json(resume_path, {**info, "bytes": info["total_bytes"], "prepared_artifact": item})
    os.replace(partial, path)
    manifest_update(root, f"raw/{product}", item)
    return item


def download(root, config, product, token, run):
    root = Path(root)
    require(re.fullmatch(r"[A-Za-z0-9_]+", product), "unsafe product key")
    remote = config["products"][product]["remote_product"]
    require(re.fullmatch(r"[A-Za-z0-9_]+", remote), "unsafe remote product")
    path = root / "data" / "raw" / f"{product}.csv.gz"
    run.save(requested_product=product, remote_product=remote)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "data" / "MANIFEST.json", {"artifacts": {}})
    recorded = manifest["artifacts"].get(f"raw/{product}")
    partial = path.with_suffix(path.suffix + ".partial")
    resume_path = root / "checkpoints" / f"download_{product}_resume.json"
    previous = read_json(resume_path, {})
    require(not previous or previous["remote_product"] == remote, "download config changed")
    if path.exists():
        if not recorded and previous.get("prepared_artifact"):
            recorded = previous["prepared_artifact"]
            require(recorded["sha256"] == sha256(path), "prepared raw file checksum mismatch")
            manifest_update(root, f"raw/{product}", recorded)
        require(
            recorded
            and recorded["remote_product"] == remote
            and recorded["sha256"] == sha256(path),
            "existing raw file lacks matching provenance/hash; do not overwrite it",
        )
        return recorded
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and offset == previous.get("bytes") == previous.get("total_bytes"):
        info = {
            k: previous[k]
            for k in (
                "remote_product",
                "etag",
                "last_modified",
                "total_bytes",
                "resolved_resource",
                "dataset",
                "snapshot",
            )
        }
        return finalize_download(root, product, path, partial, resume_path, info)
    headers = {"Accept-Encoding": "identity"}
    validator = previous.get("etag") or previous.get("last_modified")
    if offset and validator:
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    else:
        offset = 0
    params = {"dataset": "fafb", "data_version": "783", "data_product": remote, "api_token": token}
    with request(
        BASE + "/api/download_resource", params=params, headers=headers, stream=True
    ) as response:
        final_url = clean_url(response.url)
        require(
            "/fafb/783/" in urlsplit(final_url).path,
            "cannot verify resolved resource belongs to fafb/783; inspect endpoint metadata",
        )
        require(
            response.headers.get("Content-Encoding", "identity") == "identity",
            "encoded HTTP response would invalidate byte-range resume",
        )
        if response.status_code == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            require(match and int(match[1]) == offset, "invalid byte range on resumed download")
            total = int(match[3])
            require(
                int(match[2]) == total - 1, "partial range response does not reach resource end"
            )
            require(
                not previous.get("total_bytes") or previous["total_bytes"] == total,
                "resource length changed during resume",
            )
            if previous.get("etag") and response.headers.get("ETag"):
                require(
                    previous["etag"] == response.headers["ETag"],
                    "resource ETag changed during resume",
                )
        else:
            offset = 0
            total = int(response.headers.get("Content-Length", 0))
        require(total > 0, "resource Content-Length/total is required for reliable resume")
        disk_guard(root, config["minimum_free_bytes"], total - offset)
        info = {
            "remote_product": remote,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "total_bytes": total,
            "resolved_resource": final_url,
            "dataset": "fafb",
            "snapshot": "783",
        }
        with partial.open("ab" if offset else "wb") as stream:
            try:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    disk_guard(root, config["minimum_free_bytes"], len(chunk))
                    stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                    offset += len(chunk)
                    atomic_json(resume_path, {**info, "bytes": offset, "checkpoint_at": now()})
                    run.save(product=product, bytes=offset, total_bytes=total)
            except requests.RequestException as exc:
                raise Fail(
                    f"FAIL: interrupted download ({type(exc).__name__}); partial preserved"
                ) from None
        require(offset == total, "download length mismatch; partial file preserved")
    return finalize_download(root, product, path, partial, resume_path, info)


def inspect_raw(root, config):
    root = Path(root)
    manifest = read_json(root / "data" / "MANIFEST.json", {"artifacts": {}})
    products = {}
    missing = []
    for product in config["products"]:
        item = manifest["artifacts"].get(f"raw/{product}")
        if not item:
            missing.append(product)
            continue
        path = root / item["path"]
        require(sha256(path) == item["sha256"], f"raw/{product} checksum mismatch")
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            require(header and len(header) == len(set(header)), "empty or duplicate CSV column")
            examples = []
            for _, row in zip(range(5), reader):
                require(len(row) == len(header), "CSV example width differs from header")
                examples.append(row)
        products[product] = {
            "header": header,
            "raw_examples": examples,
            "raw_sha256": item["sha256"],
            "bytes": item["bytes"],
            "remote_product": item["remote_product"],
            "resolved_resource": item.get("resolved_resource"),
            "raw_format": "gzip CSV; untyped text fields",
        }
    require(products, "no downloaded raw products to inspect")
    inventory = {
        "dataset": "fafb",
        "snapshot": "783",
        "products": products,
        "missing_products": missing,
        "status": "INCOMPLETE" if missing else "COMPLETE",
    }
    atomic_json(root / "reports" / "schema_inventory.json", inventory)
    return inventory


def approved(root, config):
    inventory_path = Path(root) / "reports" / "schema_inventory.json"
    require(inventory_path.exists(), "inspect raw schema before proceeding")
    approval = config["approval"]
    require(
        approval["reviewed_by"] and approval["reviewed_at"] and approval["decision_ids"],
        "human semantic review has not been recorded",
    )
    decisions = (Path(root) / "DECISIONS.md").read_text(encoding="utf-8")
    require(
        all(identifier in decisions for identifier in approval["decision_ids"]),
        "approval decision IDs are missing from DECISIONS.md",
    )
    require(
        approval["inventory_sha256"] == sha256(inventory_path),
        "schema inventory changed or has not been approved",
    )
    inventory = read_json(inventory_path)
    require(set(inventory["products"]) == set(config["products"]), "product inventory differs")
    for product, item in inventory["products"].items():
        require(
            config["schema"].get(product, {}).get("header") == item["header"],
            f"schema mismatch for {product}; report actual columns to human first",
        )
        require(
            config["products"][product]["remote_product"] == item["remote_product"],
            "remote product changed since inspection",
        )
    return inventory


def convert_product(root, config, product, inventory, run):
    root = Path(root)
    source = root / "data" / "raw" / f"{product}.csv.gz"
    source_hash = sha256(source)
    require(
        source_hash == inventory["products"][product]["raw_sha256"],
        "raw data changed after human schema review",
    )
    header = inventory["products"][product]["header"]
    output = root / "data" / "parquet" / product
    output.mkdir(parents=True, exist_ok=True)
    state_path = root / "checkpoints" / f"convert_{product}_resume.json"
    signature = fingerprint(
        {
            "raw_sha256": source_hash,
            "header": header,
            "block_size": config["arrow_block_bytes"],
            "arrow_version": pa.__version__,
            "polars_version": pl.__version__,
        }
    )
    state = read_json(state_path, {"signature": signature, "shards": []})
    require(state["signature"] == signature, "conversion inputs/settings changed")
    for item in state["shards"]:
        require(sha256(root / item["path"]) == item["sha256"], "saved parquet shard is corrupt")
    with pa.input_stream(str(source), compression="gzip") as stream:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=config["arrow_block_bytes"]),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types={h: pa.string() for h in header}, strings_can_be_null=False
            ),
        )
        require(reader.schema.names == header, "actual Arrow header differs from inventory")
        count = 0
        for index, batch in enumerate(reader):
            count += 1
            if index < len(state["shards"]):
                require(
                    state["shards"][index]["rows"] == batch.num_rows, "CSV replay partition changed"
                )
                continue
            disk_guard(root, config["minimum_free_bytes"], batch.nbytes)
            target = output / f"part-{index:06d}.parquet"
            frame = pl.from_arrow(batch)
            atomic_file(target, lambda p, frame=frame: frame.write_parquet(p, compression="zstd"))
            state["shards"].append(record_file(root, target, rows=batch.num_rows))
            atomic_json(state_path, state)
            run.save(product=product, completed_shards=len(state["shards"]))
        require(count == len(state["shards"]), "CSV replay lost previously converted shards")
    require(state["shards"], "empty CSV product")
    result = {
        "raw_sha256": source_hash,
        "shards": state["shards"],
        "rows": sum(p["rows"] for p in state["shards"]),
        "bytes": sum(p["bytes"] for p in state["shards"]),
        "sha256": fingerprint(state["shards"]),
        "checksum_scope": "ordered shard records",
        "created_at": now(),
        "conversion_signature": signature,
    }
    manifest_update(root, f"parquet/{product}", result)
    return result


def scan_product(root, product, verify=True):
    manifest = read_json(Path(root) / "data" / "MANIFEST.json", {"artifacts": {}})
    item = manifest["artifacts"].get(f"parquet/{product}")
    require(item, f"parquet/{product} missing; convert first")
    paths = [Path(root) / shard["path"] for shard in item["shards"]]
    if verify:
        for path, shard in zip(paths, item["shards"]):
            require(sha256(path) == shard["sha256"], f"parquet checksum mismatch: {path.name}")
    return pl.scan_parquet(paths)


def value_inventory(root, config):
    result = {}
    for product in config["products"]:
        frame = scan_product(root, product)
        columns = {}
        for column in frame.collect_schema().names():
            counts = frame.group_by(column).len().sort("len", descending=True)
            columns[column] = {
                "unique_values": counts.select(pl.len()).collect().item(),
                "top_100": counts.head(100).collect().to_dicts(),
            }
        result[product] = columns
    atomic_json(Path(root) / "reports" / "value_inventory.json", result)
    return {"status": "PASS", "report": "reports/value_inventory.json"}


def connection_profile(root, config, run):
    root = Path(root)
    inventory = read_json(root / "reports" / "schema_inventory.json")
    item = inventory["products"]["connections"]
    path = root / "data" / "raw" / "connections.csv.gz"
    require(sha256(path) == item["raw_sha256"], "raw connection checksum mismatch")
    header = item["header"]
    require(
        {"pre_root_id", "post_root_id", "syn_count"} <= set(header),
        "actual connection columns differ from researcher-approved names; report them first",
    )
    signature = fingerprint(
        {
            "raw_sha256": item["raw_sha256"],
            "block": config["arrow_block_bytes"],
            "arrow": pa.__version__,
        }
    )
    checkpoint = root / "checkpoints" / "connection_profile_resume.json"
    state = read_json(checkpoint, {})
    require(not state or state["signature"] == signature, "connection profile input changed")
    pre, post = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    rows, above, completed = 0, 0, 0
    if state:
        saved = root / "checkpoints" / state["file"]
        require(sha256(saved) == state["sha256"], "connection-profile checkpoint corrupt")
        with np.load(saved, allow_pickle=False) as values:
            pre, post = values["pre"], values["post"]
        rows, above, completed = state["rows"], state["above"], state["completed_batches"]
    last_save = time.monotonic()
    with pa.input_stream(str(path), compression="gzip") as stream:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=config["arrow_block_bytes"]),
            convert_options=pacsv.ConvertOptions(
                column_types={h: pa.string() for h in header}, strings_can_be_null=False
            ),
        )
        for index, batch in enumerate(reader):
            if index < completed:
                continue
            frame = pl.from_arrow(batch).select(
                pl.col("pre_root_id").cast(pl.UInt64, strict=True),
                pl.col("post_root_id").cast(pl.UInt64, strict=True),
                pl.col("syn_count").cast(pl.Int64, strict=True),
            )
            require(
                frame.null_count().sum_horizontal().item() == 0 and frame["syn_count"].min() > 0,
                "invalid connection endpoints/counts",
            )
            rows += len(frame)
            above += int((frame["syn_count"] >= 5).sum())
            pre = np.union1d(pre, frame["pre_root_id"].unique().to_numpy())
            post = np.union1d(post, frame["post_root_id"].unique().to_numpy())
            run.save(
                profile_rows=rows,
                rows_syn_count_ge_5=above,
                unique_pre=len(pre),
                unique_post=len(post),
            )
            if time.monotonic() - last_save >= config["checkpoint_seconds"]:
                target = root / "checkpoints" / f"connection_profile_{index + 1:06d}.npz"
                atomic_file(
                    target, lambda p, pre=pre, post=post: np.savez_compressed(p, pre=pre, post=post)
                )
                atomic_json(
                    checkpoint,
                    {
                        "signature": signature,
                        "file": target.name,
                        "sha256": sha256(target),
                        "rows": rows,
                        "above": above,
                        "completed_batches": index + 1,
                    },
                )
                last_save = time.monotonic()
    require(rows > 0 and above > 0, "no nonempty >=5-synapse connection population")
    reference = 2613129
    report = {
        "dataset": "fafb",
        "snapshot": "783",
        "remote_product": item["remote_product"],
        "raw_sha256": item["raw_sha256"],
        "header": header,
        "rows": rows,
        "unique_pre_root_id": len(pre),
        "unique_post_root_id": len(post),
        "unique_endpoint_union": len(np.union1d(pre, post)),
        "rows_syn_count_ge_5": above,
        "v630_reference": reference,
        "difference_from_v630": above - reference,
        "difference_percent": 100 * (above - reference) / reference,
        "status": "OBSERVED_NO_EXACT_V783_TOLERANCE_SUPPLIED",
        "scope": "raw rows before pair aggregation; not a model metric",
    }
    atomic_json(root / "reports" / "connection_sanity.json", report)
    return report
