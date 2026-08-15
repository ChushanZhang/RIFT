from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RIFT_SOURCES = (
    REPO_ROOT / "rift/model_factory.py",
    REPO_ROOT / "rift/models/wan22/rift_model.py",
)

REMOVED_TRAINING_MODULES = (
    "rift/models/wan22/fastwam_joint.py",
    "rift/models/wan22/fastwam_idm.py",
    "rift/models/wan22/wan22.py",
    "rift/datasets/lerobot/transforms/misc.py",
    "rift/datasets/lerobot/transforms/relative_action.py",
    "rift/datasets/lerobot/transforms/rotation.py",
    "rift/datasets/lerobot/utils/rotation.py",
    "rift/utils/config_resolvers.py",
    "rift/utils/video_io.py",
    "rift/utils/video_metrics.py",
)


class RIFTSourceContractTest(unittest.TestCase):
    def test_python_package_uses_repository_root_layout(self) -> None:
        self.assertTrue((REPO_ROOT / "rift/__init__.py").is_file())
        self.assertFalse((REPO_ROOT / "fastwam").exists())
        self.assertFalse((REPO_ROOT / "src").exists())

        packaging = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("[tool.setuptools.package-dir]", packaging)
        self.assertIn('name = "rift-wam"', packaging)
        self.assertIn('where = ["."]', packaging)
        self.assertIn('include = ["rift*"]', packaging)

    def test_public_source_exposes_only_the_canonical_method(self) -> None:
        forbidden_identifiers = {
            "anticip_" + "fut_" + "commit",
            "anticip_" + "fut_" + "commit_replace",
            "fut_" + "commit",
            "fut_" + "commit_replace",
            "anticip_" + "native_video_" + "pass",
            "native_video_" + "pass",
        }

        for source_path in RIFT_SOURCES:
            with self.subTest(source=source_path.relative_to(REPO_ROOT)):
                source = source_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(source_path))
                identifiers = {
                    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                }
                identifiers.update(
                    node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
                )
                identifiers.update(
                    node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                )
                self.assertEqual(identifiers & forbidden_identifiers, set())

    def test_full_state_resume_restores_curriculum_position(self) -> None:
        trainer_source = (REPO_ROOT / "rift/trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'unwrapped_model._anticip_step = int(self.global_step)',
            trainer_source,
        )

    def test_training_runtime_excludes_benchmark_orchestration(self) -> None:
        runtime_path = REPO_ROOT / "rift/runtime.py"
        runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        runtime_functions = {
            node.name
            for node in runtime_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("run_training", runtime_functions)
        self.assertNotIn("run_inference", runtime_functions)
        self.assertNotIn("build_datasets", runtime_functions)

        trainer_path = REPO_ROOT / "rift/trainer.py"
        trainer_tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
        trainer_methods = {
            node.name
            for node in ast.walk(trainer_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("evaluate", trainer_methods)

        trainer_class = next(
            node
            for node in trainer_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Wan22Trainer"
        )
        trainer_init = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_args = {arg.arg for arg in trainer_init.args.args}
        self.assertNotIn("val_dataset", init_args)

        for relative_path in REMOVED_TRAINING_MODULES:
            with self.subTest(path=relative_path):
                self.assertFalse((REPO_ROOT / relative_path).exists())

    def test_training_rejects_weights_only_initialization(self) -> None:
        trainer_path = REPO_ROOT / "rift/trainer.py"
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
        resume_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_resume_training_state"
        )
        calls = {
            node.func.attr
            for node in ast.walk(resume_fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        raised_errors = {
            node.exc.func.id
            for node in ast.walk(resume_fn)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
        }
        self.assertIn("load_training_state", calls)
        self.assertIn("is_dir", calls)
        self.assertNotIn("load_checkpoint", calls)
        self.assertIn("ValueError", raised_errors)

    def test_shared_training_entrypoint_matches_upstream_layout(self) -> None:
        launcher = REPO_ROOT / "scripts/train_zero2.sh"
        self.assertTrue((REPO_ROOT / "scripts/train.py").is_file())
        self.assertTrue(launcher.is_file())
        self.assertEqual(
            {path.name for path in (REPO_ROOT / "scripts").glob("train*.sh")},
            {"train_zero2.sh"},
        )

        launcher_source = launcher.read_text(encoding="utf-8")
        self.assertIn("scripts/train.py", launcher_source)
        self.assertIn('"${EXTRA_ARGS[@]}"', launcher_source)


if __name__ == "__main__":
    unittest.main()
