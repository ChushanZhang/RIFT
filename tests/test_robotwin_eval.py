from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.checkpoint_utils import load_rift_checkpoint_exact
from experiments.robotwin.eval_robotwin_single import (
    _ensure_policy_symlink,
    _resolve_robotwin_root,
)
from experiments.robotwin.rift_policy import deploy_policy
from experiments.robotwin.run_robotwin_manager import (
    _load_all_tasks,
    _parse_success_rate,
)
from rift.models.wan22.rift_model import RIFTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class RoboTwinEvalConfigTest(unittest.TestCase):
    def test_sim_config_resolves_to_canonical_policy(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="sim_robotwin")
            OmegaConf.resolve(cfg)

        self.assertEqual(cfg.model._target_, "rift.model_factory.create_rift")
        self.assertEqual(cfg.model.anticip_future_tokens, 240)
        self.assertEqual(cfg.data.train.processor.num_output_cameras, 3)
        self.assertEqual(list(cfg.data.train.video_size), [384, 320])
        self.assertEqual(cfg.EVALUATION.policy_name, "rift_policy")
        self.assertIsNone(cfg.EVALUATION.robotwin_root)
        self.assertTrue(cfg.model.load_text_encoder)
        self.assertTrue(cfg.model.skip_dit_load_from_pretrain)

    def test_policy_adapter_matches_model_and_robotwin_abi(self) -> None:
        model_parameters = inspect.signature(RIFTModel.infer_action).parameters
        self.assertLessEqual(
            {
                "prompt",
                "input_image",
                "action_horizon",
                "num_video_frames",
                "proprio",
                "num_inference_steps",
            },
            set(model_parameters),
        )
        self.assertTrue(callable(deploy_policy.get_model))
        self.assertTrue(callable(deploy_policy.eval))
        self.assertTrue(callable(deploy_policy.reset_model))


class RoboTwinExternalCheckoutTest(unittest.TestCase):
    def test_checkpoint_contract_rejects_partial_policy_state(self) -> None:
        class DummyMoT:
            @staticmethod
            def state_dict() -> dict[str, object]:
                return {"anticip.head_fm.weight": object()}

        class DummyModel:
            mot = DummyMoT()
            proprio_encoder = None

            @staticmethod
            def load_checkpoint(_path: str) -> dict:
                return {"mot": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weights.pt"
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                load_rift_checkpoint_exact(DummyModel(), checkpoint)

    def test_external_root_is_required_and_policy_link_is_scoped(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ROBOTWIN_ROOT"):
                _resolve_robotwin_root(None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "RoboTwin"
            policy_root = root / "policy"
            policy_root.mkdir(parents=True)
            source = Path(tmpdir) / "rift_policy"
            source.mkdir()
            target = _ensure_policy_symlink(root, source)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())
            self.assertFalse((REPO_ROOT / "third_party" / "RoboTwin").exists())

    def test_external_task_list_and_result_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_config = root / "task_config"
            task_config.mkdir()
            (task_config / "_eval_step_limit.yml").write_text(
                "task_a: 100\ntask_b: 200\ntask_a: 300\n", encoding="utf-8"
            )
            self.assertEqual(_load_all_tasks(root), ["task_a", "task_b"])

            result = root / "_result.txt"
            result.write_text("episode 1\n0.25\n0.75\n", encoding="utf-8")
            self.assertEqual(_parse_success_rate(result), 0.75)


if __name__ == "__main__":
    unittest.main()
