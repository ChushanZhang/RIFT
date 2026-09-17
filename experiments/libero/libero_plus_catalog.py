"""Build and validate the official 10,030-task LIBERO-Plus catalog."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CATALOG_SCHEMA = "rift.libero-plus-catalog.v1"
SUITE_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
SUITE_ORDER = tuple(SUITE_COUNTS)
CATEGORY_COUNTS = {
    "Camera Viewpoints": 1599,
    "Robot Initial States": 1550,
    "Language Instructions": 1537,
    "Light Conditions": 1142,
    "Background Textures": 1076,
    "Sensor Noise": 1601,
    "Objects Layout": 1525,
}
TOTAL_TASKS = sum(SUITE_COUNTS.values())


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path | str, value: Any, *, exclusive: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, target)
            temporary.unlink()
        else:
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_revision(repo: Path | str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def bootstrap_libero_plus(root: Path | str) -> Path:
    plus_root = Path(root).expanduser().resolve()
    package_init = plus_root / "libero" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"LIBERO-Plus package not found below {plus_root}; expected {package_init}."
        )

    imported = sys.modules.get("libero")
    if imported is not None:
        imported_file = Path(str(imported.__file__)).resolve()
        if not imported_file.is_relative_to(plus_root):
            raise RuntimeError(
                f"libero is already imported from {imported_file}, not {plus_root}."
            )

    root_text = str(plus_root)
    while root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    libero = importlib.import_module("libero")
    imported_file = Path(str(libero.__file__)).resolve()
    if not imported_file.is_relative_to(plus_root):
        raise RuntimeError(f"failed to bind LIBERO-Plus; imported {imported_file}")
    return plus_root


def validate_catalog_document(catalog: dict[str, Any]) -> dict[str, Any]:
    required = ("schema", "identity", "suite_counts", "category_counts", "tasks")
    try:
        digest_payload = {key: catalog[key] for key in required}
    except KeyError as exc:
        raise ValueError(f"catalog is missing required key {exc.args[0]!r}") from exc
    if catalog["schema"] != CATALOG_SCHEMA:
        raise ValueError(f"unsupported catalog schema: {catalog['schema']!r}")
    actual_digest = canonical_sha256(digest_payload)
    if actual_digest != catalog.get("catalog_sha256"):
        raise ValueError("catalog digest mismatch")
    if catalog.get("suite_counts") != SUITE_COUNTS:
        raise ValueError("catalog suite counts do not match LIBERO-Plus")
    if catalog.get("category_counts") != CATEGORY_COUNTS:
        raise ValueError("catalog category counts do not match LIBERO-Plus")
    if catalog.get("total_tasks") != TOTAL_TASKS or len(catalog["tasks"]) != TOTAL_TASKS:
        raise ValueError("catalog is not the complete 10,030-task protocol")
    return catalog


def load_catalog_file(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        catalog = json.load(stream)
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a JSON object")
    return validate_catalog_document(catalog)


def load_classification(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if set(raw) != set(SUITE_ORDER):
        raise ValueError("task classification contains unexpected suites")

    category_counts: dict[str, int] = {}
    validated: dict[str, list[dict[str, Any]]] = {}
    for suite in SUITE_ORDER:
        rows = raw[suite]
        if not isinstance(rows, list) or len(rows) != SUITE_COUNTS[suite]:
            raise ValueError(f"{suite}: unexpected classification count")
        seen_names: set[str] = set()
        checked_rows: list[dict[str, Any]] = []
        for expected_id, row in enumerate(rows, start=1):
            if int(row.get("id", -1)) != expected_id:
                raise ValueError(f"{suite}: non-contiguous classification IDs")
            name = str(row.get("name", ""))
            if not name or name in seen_names:
                raise ValueError(f"{suite}: empty or duplicate task name {name!r}")
            seen_names.add(name)
            category = str(row.get("category", ""))
            if category not in CATEGORY_COUNTS:
                raise ValueError(f"{suite}/{name}: unknown category {category!r}")
            difficulty = row.get("difficulty_level")
            if difficulty is not None and int(difficulty) not in range(1, 6):
                raise ValueError(f"{suite}/{name}: invalid difficulty {difficulty!r}")
            category_counts[category] = category_counts.get(category, 0) + 1
            checked_rows.append(
                {
                    "id": expected_id,
                    "name": name,
                    "category": category,
                    "difficulty_level": None if difficulty is None else int(difficulty),
                }
            )
        validated[suite] = checked_rows
    if category_counts != CATEGORY_COUNTS:
        raise ValueError("task classification category counts do not match LIBERO-Plus")
    return validated


def resolve_bddl_path(task: Any, bddl_root: Path) -> Path:
    candidate = bddl_root / task.problem_folder / task.bddl_file
    candidate_text = str(candidate)
    if "_view_" in candidate_text and "_initstate_" in candidate_text:
        return Path(candidate_text.split("_view_", 1)[0] + ".bddl")
    return candidate


def resolve_init_path(task: Any, init_root: Path) -> Path:
    filename = str(task.init_states_file)
    suite = str(task.problem_folder)
    suffix = "." + filename.rsplit(".", 1)[-1]
    if "_language_" in filename:
        return init_root / suite / (filename.split("_language_", 1)[0] + suffix)
    if "_view_" in filename:
        return init_root / suite / (filename.split("_view_", 1)[0] + suffix)
    if re.search(r"_table_\d+", filename):
        return init_root / suite / re.sub(r"_table_\d+", "", filename)
    if re.search(r"_tb_\d+", filename):
        return init_root / suite / re.sub(r"_tb_\d+", "", filename)
    if "_light_" in filename:
        return init_root / suite / (filename.split("_light_", 1)[0] + suffix)
    if "_add_" in filename or "_level" in filename:
        return init_root / "libero_newobj" / suite / filename
    return init_root / suite / filename


def build_catalog(
    *,
    libero_plus_root: Path | str,
    expected_revision: str | None = None,
    validate_resources: bool = True,
    assets_archive: Path | str | None = None,
) -> dict[str, Any]:
    plus_root = bootstrap_libero_plus(libero_plus_root)
    from libero.libero import benchmark, get_libero_path

    benchmark_file = Path(benchmark.__file__).resolve()
    if not benchmark_file.is_relative_to(plus_root):
        raise RuntimeError(f"imported benchmark outside {plus_root}: {benchmark_file}")
    revision = git_revision(plus_root)
    if expected_revision is not None and revision != expected_revision:
        raise RuntimeError(
            f"LIBERO-Plus revision {revision!r} != requested {expected_revision!r}"
        )

    classification_path = benchmark_file.with_name("task_classification.json")
    classification = load_classification(classification_path)
    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    init_root = Path(get_libero_path("init_states")).resolve()
    assets_root = Path(get_libero_path("assets")).resolve()
    benchmark_dict = benchmark.get_benchmark_dict()
    tasks: list[dict[str, Any]] = []
    missing: list[str] = []

    for suite_index, suite_name in enumerate(SUITE_ORDER):
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark_dict[suite_name]()
        if int(suite.n_tasks) != SUITE_COUNTS[suite_name]:
            raise RuntimeError(f"{suite_name}: unexpected runtime task count")
        rows_by_name = {row["name"]: row for row in classification[suite_name]}
        runtime_names = [suite.get_task(index).name for index in range(suite.n_tasks)]
        if set(runtime_names) != set(rows_by_name):
            raise RuntimeError(f"{suite_name}: runtime and classification tasks differ")

        for task_id, task_name in enumerate(runtime_names):
            task = suite.get_task(task_id)
            row = rows_by_name[task_name]
            if row["id"] != task_id + 1:
                raise RuntimeError(f"{suite_name}/{task_name}: classification ID mismatch")
            bddl_path = resolve_bddl_path(task, bddl_root)
            init_path = resolve_init_path(task, init_root)
            if validate_resources:
                if not bddl_path.is_file():
                    missing.append(f"BDDL:{bddl_path}")
                if not init_path.is_file():
                    missing.append(f"INIT:{init_path}")
            tasks.append(
                {
                    "global_task_id": len(tasks),
                    "suite_index": suite_index,
                    "suite": suite_name,
                    "runtime_task_id": task_id,
                    "classification_id": row["id"],
                    "name": task_name,
                    "category": row["category"],
                    "difficulty_level": row["difficulty_level"],
                    "task_language": str(task.language),
                    "bddl_file": str(task.bddl_file),
                    "resolved_bddl_path": str(bddl_path),
                    "init_states_file": str(task.init_states_file),
                    "resolved_init_path": str(init_path),
                }
            )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} LIBERO-Plus resources are missing; first entries:\n{preview}"
        )

    identity: dict[str, Any] = {
        "libero_plus_root": str(plus_root),
        "libero_plus_revision": revision,
        "benchmark_file": str(benchmark_file),
        "benchmark_sha256": file_sha256(benchmark_file),
        "classification_file": str(classification_path),
        "classification_sha256": file_sha256(classification_path),
        "bddl_root": str(bddl_root),
        "init_root": str(init_root),
        "assets_root": str(assets_root),
    }
    config_env = os.environ.get("LIBERO_CONFIG_PATH")
    if config_env:
        config_path = Path(config_env).expanduser().resolve()
        if config_path.is_dir():
            config_path = config_path / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"invalid LIBERO_CONFIG_PATH: {config_path}")
        identity["libero_config_file"] = str(config_path)
        identity["libero_config_sha256"] = file_sha256(config_path)
    if assets_archive is not None:
        archive_path = Path(assets_archive).expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"assets archive not found: {archive_path}")
        archive_stat = archive_path.stat()
        identity["assets_archive"] = {
            "path": str(archive_path),
            "sha256": file_sha256(archive_path),
            "size": archive_stat.st_size,
            "mtime_ns": archive_stat.st_mtime_ns,
        }

    digest_payload = {
        "schema": CATALOG_SCHEMA,
        "identity": identity,
        "suite_counts": SUITE_COUNTS,
        "category_counts": CATEGORY_COUNTS,
        "tasks": tasks,
    }
    return {
        **digest_payload,
        "total_tasks": TOTAL_TASKS,
        "catalog_sha256": canonical_sha256(digest_payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--libero-plus-revision")
    parser.add_argument("--assets-archive", type=Path)
    parser.add_argument("--skip-resource-validation", action="store_true")
    args = parser.parse_args()
    catalog = build_catalog(
        libero_plus_root=args.libero_plus_root,
        expected_revision=args.libero_plus_revision,
        validate_resources=not args.skip_resource_validation,
        assets_archive=args.assets_archive,
    )
    write_json_atomic(args.output, catalog)
    print(f"validated {catalog['total_tasks']} LIBERO-Plus tasks")


if __name__ == "__main__":
    main()
