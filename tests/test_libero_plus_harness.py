from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

from experiments.libero.eval_libero_single import run_single_episode
from experiments.libero.libero_plus_catalog import (
    CATALOG_SCHEMA,
    CATEGORY_COUNTS,
    SUITE_COUNTS,
    TOTAL_TASKS,
    bootstrap_libero_plus,
    canonical_sha256,
    resolve_bddl_path,
    resolve_init_path,
    write_json_atomic,
)
from experiments.libero.summarize_libero_plus import summarize


class LiberoPlusHarnessTest(unittest.TestCase):
    def test_bootstrap_libero_plus_uses_nested_checkout_package(self) -> None:
        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "libero" or name.startswith("libero.")
        }
        original_path = sys.path[:]
        try:
            for name in saved_modules:
                sys.modules.pop(name, None)
            with tempfile.TemporaryDirectory() as directory:
                checkout = Path(directory) / "LIBERO-plus"
                package_init = checkout / "libero/libero/__init__.py"
                package_init.parent.mkdir(parents=True)
                package_init.write_text("", encoding="utf-8")

                sys.path.insert(0, str(checkout))
                namespace = importlib.import_module("libero")
                self.assertIsNone(namespace.__file__)

                self.assertEqual(bootstrap_libero_plus(checkout), checkout.resolve())
                package = importlib.import_module("libero.libero")
                self.assertTrue(
                    Path(package.__file__).resolve().is_relative_to(checkout.resolve())
                )
        finally:
            sys.path[:] = original_path
            for name in list(sys.modules):
                if name == "libero" or name.startswith("libero."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_resource_resolution_matches_libero_plus(self) -> None:
        root = Path("/tmp/libero-plus-test")
        task = SimpleNamespace(
            problem_folder="libero_spatial",
            bddl_file="pick_bowl_view_1_2_100_0_0_initstate_3_noise_2.bddl",
            init_states_file="pick_bowl_view_1_2_100_0_0_initstate_3_noise_2.pruned_init",
        )
        self.assertEqual(
            resolve_bddl_path(task, root / "bddl"),
            root / "bddl/libero_spatial/pick_bowl.bddl",
        )
        self.assertEqual(
            resolve_init_path(task, root / "init"),
            root / "init/libero_spatial/pick_bowl.pruned_init",
        )
        task.init_states_file = "pick_bowl_table_17.pruned_init"
        self.assertEqual(resolve_init_path(task, root / "init").name, "pick_bowl.pruned_init")
        task.init_states_file = "pick_bowl_light_9.pruned_init"
        self.assertEqual(resolve_init_path(task, root / "init").name, "pick_bowl.pruned_init")

    def test_episode_worker_exposes_persistent_eval_controls(self) -> None:
        parameters = inspect.signature(run_single_episode).parameters
        self.assertIn("collect_rollout_images", parameters)
        self.assertIn("show_progress", parameters)
        self.assertTrue(parameters["collect_rollout_images"].default)
        self.assertTrue(parameters["show_progress"].default)

    def test_atomic_exclusive_writer_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json_atomic(path, {"value": 1}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write_json_atomic(path, {"value": 2}, exclusive=True)
            self.assertEqual(json.loads(path.read_text())["value"], 1)

    def test_partial_summary_uses_rift_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            catalog_path = root / "catalog.json"
            task = {
                "global_task_id": 0,
                "suite": "libero_goal",
                "runtime_task_id": 0,
                "classification_id": 1,
                "name": "task_a",
                "category": "Light Conditions",
                "difficulty_level": None,
                "task_language": "task a",
                "bddl_file": "task_a.bddl",
                "resolved_bddl_path": "/bddl/task_a.bddl",
                "resolved_init_path": "/init/task_a.pruned_init",
            }
            tasks = [task]
            for task_id in range(1, TOTAL_TASKS):
                tasks.append(
                    {
                        "global_task_id": task_id,
                        "suite": "libero_goal",
                        "runtime_task_id": task_id,
                        "classification_id": task_id + 1,
                        "name": f"missing_{task_id}",
                        "category": "Camera Viewpoints",
                        "difficulty_level": 1,
                        "task_language": f"missing {task_id}",
                        "bddl_file": f"missing_{task_id}.bddl",
                        "resolved_bddl_path": f"/bddl/missing_{task_id}.bddl",
                        "resolved_init_path": f"/init/missing_{task_id}.pruned_init",
                    }
                )
            catalog_body = {
                "schema": CATALOG_SCHEMA,
                "identity": {},
                "suite_counts": SUITE_COUNTS,
                "category_counts": CATEGORY_COUNTS,
                "tasks": tasks,
            }
            catalog_sha = canonical_sha256(catalog_body)
            write_json_atomic(
                catalog_path,
                {
                    **catalog_body,
                    "total_tasks": TOTAL_TASKS,
                    "catalog_sha256": catalog_sha,
                },
            )
            manifest_body = {
                "schema": "rift.libero-plus-run-manifest.v1",
                "catalog_sha256": catalog_sha,
                "seed": 42,
                "protocol": {
                    "selected_task_count": 1,
                    "selected_global_task_ids": [0],
                },
            }
            manifest_sha = canonical_sha256(manifest_body)
            write_json_atomic(
                run_dir / "run_manifest.json",
                {**manifest_body, "manifest_sha256": manifest_sha},
            )
            write_json_atomic(
                run_dir / "receipts/libero_goal/0000.json",
                {
                    "schema": "rift.libero-plus-task-result.v1",
                    "manifest_sha256": manifest_sha,
                    "catalog_sha256": catalog_sha,
                    "suite": "libero_goal",
                    "runtime_task_id": 0,
                    "classification_id": 1,
                    "global_task_id": 0,
                    "task_name": "task_a",
                    "category": "Light Conditions",
                    "difficulty_level": None,
                    "prompt_mode": "pinned_task_language",
                    "logical_bddl_file": "task_a.bddl",
                    "resolved_bddl_path": "/bddl/task_a.bddl",
                    "resolved_init_path": "/init/task_a.pruned_init",
                    "prompt": "task a",
                    "seed": 42,
                    "initial_state_index": 0,
                    "success": True,
                },
            )

            result = summarize(run_dir, catalog_path, allow_partial=True)
            self.assertEqual(result["schema"], "rift.libero-plus-summary.v1")
            self.assertFalse(result["complete"])
            self.assertEqual(result["overall_micro"]["success_rate"], 1.0)
            self.assertEqual(result["by_difficulty"]["unclassified"]["denominator"], 1)
            self.assertTrue((run_dir / "partial_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
