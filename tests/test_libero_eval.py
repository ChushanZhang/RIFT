from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from experiments.libero.action_ensembler import ActionEnsembler
from experiments.libero.eval_libero_single import (
    _center_crop_resize,
    _get_max_steps,
    _get_task_initial_states,
    _load_model_checkpoint,
    _num_video_frames,
)
from experiments.libero.run_libero_manager import _is_blocked_override
from experiments.libero.summarize_results import summarize_results
from rift.models.wan22.rift_model import RIFTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class LiberoEvalConfigTest(unittest.TestCase):
    def test_sim_config_resolves_to_canonical_policy(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="sim_libero")
            OmegaConf.resolve(cfg)

        self.assertEqual(cfg.model._target_, "rift.model_factory.create_rift")
        self.assertEqual(cfg.model.anticip_future_tokens, 196)
        self.assertEqual(cfg.data.train.processor.num_output_cameras, 2)
        self.assertEqual(list(cfg.data.train.video_size), [224, 448])
        self.assertEqual(_num_video_frames(cfg), 9)
        self.assertTrue(cfg.model.load_text_encoder)
        self.assertTrue(cfg.model.skip_dit_load_from_pretrain)
        self.assertIsNone(cfg.model.action_dit_pretrained_path)
        self.assertEqual(cfg.EVALUATION.num_inference_steps, 10)

    def test_eval_call_matches_model_inference_abi(self) -> None:
        parameters = inspect.signature(RIFTModel.infer_action).parameters
        expected = {
            "prompt",
            "input_image",
            "action_horizon",
            "num_video_frames",
            "proprio",
            "negative_prompt",
            "text_cfg_scale",
            "num_inference_steps",
            "sigma_shift",
            "seed",
            "rand_device",
            "tiled",
        }
        self.assertLessEqual(expected, set(parameters))


class LiberoEvalHelpersTest(unittest.TestCase):
    def test_action_ensembler_averages_overlapping_chunks(self) -> None:
        ensembler = ActionEnsembler()
        ensembler.add_actions(np.asarray([[1.0, 3.0], [2.0, 4.0]]), 0)
        ensembler.add_actions(np.asarray([[4.0, 8.0]]), 1)
        np.testing.assert_allclose(ensembler.get_action(0), [1.0, 3.0])
        np.testing.assert_allclose(ensembler.get_action(1), [3.0, 6.0])

    def test_image_geometry_and_suite_horizons(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        resized = _center_crop_resize(image, width=224, height=224)
        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(_get_max_steps("libero_spatial"), 400)
        self.assertEqual(_get_max_steps("libero_10"), 700)

    def test_checkpoint_loader_uses_public_model_method(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.loaded = None
                self.mot = torch.nn.Linear(2, 2)
                self.proprio_encoder = None

            def load_checkpoint(self, path: str) -> dict:
                self.loaded = path
                return {"mot": self.mot.state_dict()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weights.pt"
            checkpoint.touch()
            model = DummyModel()
            _load_model_checkpoint(model, checkpoint)
            self.assertEqual(model.loaded, str(checkpoint))

    def test_official_init_states_use_explicit_pickle_compatibility(self) -> None:
        source = inspect.getsource(_get_task_initial_states)
        self.assertIn("weights_only=False", source)
        self.assertIn('map_location="cpu"', source)

    def test_manager_does_not_forward_worker_identity(self) -> None:
        self.assertTrue(_is_blocked_override("ckpt=/tmp/model.pt"))
        self.assertTrue(_is_blocked_override("EVALUATION.task_id=2"))
        self.assertTrue(_is_blocked_override("MULTIRUN.num_gpus=4"))
        self.assertFalse(_is_blocked_override("EVALUATION.save_video=false"))

    def test_result_summarizer_writes_official_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            suite_dir = output_dir / "libero_spatial"
            suite_dir.mkdir()
            for task_id, successes in ((0, 3), (1, 1)):
                payload = {
                    "task_suite": "libero_spatial",
                    "task_id": task_id,
                    "successes": successes,
                    "total_episodes": 4,
                    "duration": 2.0 + task_id,
                    "task_description": f"task {task_id}",
                }
                (suite_dir / f"gpu0_task{task_id}_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            summary = summarize_results(output_dir)
            self.assertEqual(summary["overall"]["total_trials"], 8)
            self.assertEqual(summary["overall"]["total_successes"], 4)
            self.assertEqual(summary["overall"]["success_rate"], 0.5)
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "task_success_rates.csv").is_file())


if __name__ == "__main__":
    unittest.main()
