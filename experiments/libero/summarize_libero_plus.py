"""Validate and aggregate RIFT LIBERO-Plus task receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

from experiments.libero.libero_plus_catalog import (
    TOTAL_TASKS,
    canonical_sha256,
    file_sha256,
    load_catalog_file,
    write_json_atomic,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in rows)
    denominator = len(rows)
    return {
        "successes": successes,
        "denominator": denominator,
        "success_rate": None if denominator == 0 else successes / denominator,
        "success_rate_percent": (
            None if denominator == 0 else 100.0 * successes / denominator
        ),
    }


def _group(
    rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {key: _metric(groups[key]) for key in sorted(groups)}


def summarize(run_dir: Path, catalog_path: Path, *, allow_partial: bool) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    catalog = load_catalog_file(catalog_path.resolve())
    manifest = _load_json(run_dir / "run_manifest.json")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if canonical_sha256(manifest_body) != manifest.get("manifest_sha256"):
        raise RuntimeError("run manifest digest mismatch")
    if manifest.get("schema") != "rift.libero-plus-run-manifest.v1":
        raise RuntimeError("unsupported run manifest schema")
    if manifest.get("catalog_sha256") != catalog.get("catalog_sha256"):
        raise RuntimeError("run manifest and catalog hashes differ")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("run manifest has no protocol mapping")
    selected_ids = protocol.get("selected_global_task_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise RuntimeError("run manifest has no selected task IDs")
    if any(type(value) is not int for value in selected_ids):
        raise RuntimeError("selected task IDs must be integers")
    if selected_ids != sorted(set(selected_ids)):
        raise RuntimeError("selected task IDs must be sorted and unique")
    if protocol.get("selected_task_count") != len(selected_ids):
        raise RuntimeError("manifest selected task count is inconsistent")

    catalog_by_id = {int(task["global_task_id"]): task for task in catalog["tasks"]}
    if len(catalog_by_id) != TOTAL_TASKS:
        raise RuntimeError("catalog global task IDs are not unique and complete")
    unknown_ids = sorted(set(selected_ids) - set(catalog_by_id))
    if unknown_ids:
        raise RuntimeError(f"selected task IDs are absent from catalog: {unknown_ids[:10]}")
    selected_tasks = [catalog_by_id[task_id] for task_id in selected_ids]
    full_selection = selected_ids == list(range(TOTAL_TASKS))
    if not full_selection and not allow_partial:
        raise RuntimeError("selected task subset requires --allow-partial")

    expected_paths: set[Path] = set()
    completed_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    receipt_hashes: list[dict[str, str]] = []
    for task in selected_tasks:
        path = (
            run_dir
            / "receipts"
            / task["suite"]
            / f"{int(task['runtime_task_id']):04d}.json"
        )
        expected_paths.add(path.resolve())
        if not path.is_file():
            missing.append(f"{task['suite']}:{task['runtime_task_id']}")
            continue
        receipt = _load_json(path)
        expected = {
            "schema": "rift.libero-plus-task-result.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "catalog_sha256": catalog["catalog_sha256"],
            "suite": task["suite"],
            "runtime_task_id": task["runtime_task_id"],
            "classification_id": task["classification_id"],
            "global_task_id": task["global_task_id"],
            "task_name": task["name"],
            "category": task["category"],
            "difficulty_level": task["difficulty_level"],
            "prompt_mode": "pinned_task_language",
            "logical_bddl_file": task["bddl_file"],
            "resolved_bddl_path": task["resolved_bddl_path"],
            "resolved_init_path": task["resolved_init_path"],
            "prompt": task["task_language"],
            "seed": manifest["seed"],
            "initial_state_index": 0,
        }
        for key, expected_value in expected.items():
            if receipt.get(key) != expected_value:
                raise RuntimeError(
                    f"{path}: {key}={receipt.get(key)!r}, expected {expected_value!r}"
                )
        if type(receipt.get("success")) is not bool:
            raise RuntimeError(f"{path}: success must be boolean")
        completed_rows.append({**task, "success": receipt["success"]})
        receipt_hashes.append(
            {"path": str(path.relative_to(run_dir)), "sha256": file_sha256(path)}
        )

    receipt_root = run_dir / "receipts"
    actual_paths = (
        {path.resolve() for path in receipt_root.rglob("*.json")}
        if receipt_root.exists()
        else set()
    )
    extras = sorted(str(path) for path in actual_paths - expected_paths)
    if extras:
        raise RuntimeError(f"unexpected receipt files: {extras[:10]}")
    if missing:
        raise RuntimeError(
            f"missing selected LIBERO-Plus receipts: {len(completed_rows)}/"
            f"{len(selected_tasks)}; first missing={missing[:10]}"
        )

    difficulty_name = lambda row: (
        "unclassified"
        if row["difficulty_level"] is None
        else f"level_{int(row['difficulty_level'])}"
    )
    summary = {
        "schema": "rift.libero-plus-summary.v1",
        "complete": full_selection,
        "manifest_sha256": manifest["manifest_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "completed_tasks": len(completed_rows),
        "expected_tasks": len(selected_tasks),
        "official_task_count": TOTAL_TASKS,
        "selected_global_task_ids": selected_ids,
        "overall_micro": _metric(completed_rows),
        "by_suite": _group(completed_rows, lambda row: row["suite"]),
        "by_category": _group(completed_rows, lambda row: row["category"]),
        "by_difficulty": _group(completed_rows, difficulty_name),
        "by_category_and_difficulty": _group(
            completed_rows,
            lambda row: f"{row['category']}::{difficulty_name(row)}",
        ),
        "receipt_set_sha256": canonical_sha256(
            sorted(receipt_hashes, key=lambda item: item["path"])
        ),
    }
    output_name = "summary.json" if full_selection else "partial_summary.json"
    write_json_atomic(run_dir / output_name, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = summarize(args.run_dir, args.catalog, allow_partial=args.allow_partial)
    metric = result["overall_micro"]
    print(
        f"complete={result['complete']} "
        f"tasks={result['completed_tasks']}/{result['expected_tasks']} "
        f"success={metric['successes']}/{metric['denominator']} "
        f"sr={metric['success_rate_percent']}"
    )


if __name__ == "__main__":
    main()
