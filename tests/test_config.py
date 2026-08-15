from __future__ import annotations

import unittest
from inspect import signature
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT))

from rift.model_factory import create_rift  # noqa: E402


class RIFTConfigTest(unittest.TestCase):
    def test_canonical_tasks_compose_and_resolve(self) -> None:
        tasks = {
            "libero_rift_2cam224_1e-4": {
                "future_tokens": 196,
                "action_dim": 7,
                "proprio_dim": 8,
                "anneal_total_steps": 21700,
                "batch_size": 16,
                "num_workers": 8,
                "num_epochs": 10,
                "max_steps": None,
                "num_cameras": 2,
                "video_size": [224, 448],
            },
            "robotwin_rift_3cam_384_1e-4": {
                "future_tokens": 240,
                "action_dim": 14,
                "proprio_dim": 14,
                "anneal_total_steps": 46966,
                "batch_size": 16,
                "num_workers": 6,
                "num_epochs": 1,
                "max_steps": 46966,
                "num_cameras": 3,
                "video_size": [384, 320],
            },
        }

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            for task, expected in tasks.items():
                with self.subTest(task=task):
                    cfg = compose(config_name="train", overrides=[f"task={task}"])
                    OmegaConf.resolve(cfg)
                    model = cfg.model

                    self.assertIsNone(cfg.resume)
                    self.assertNotIn("eval_every", cfg)
                    self.assertNotIn("eval_num_inference_steps", cfg)

                    self.assertEqual(
                        model._target_, "rift.model_factory.create_rift"
                    )
                    self.assertEqual(model.anticip_latent_t, 3)
                    self.assertEqual(
                        model.anticip_future_tokens, expected["future_tokens"]
                    )
                    self.assertEqual(
                        model.action_dit_config.action_dim, expected["action_dim"]
                    )
                    self.assertEqual(model.proprio_dim, expected["proprio_dim"])
                    self.assertEqual(
                        model.anticip_anneal_total_steps,
                        expected["anneal_total_steps"],
                    )
                    self.assertEqual(cfg.batch_size, expected["batch_size"])
                    self.assertEqual(cfg.num_workers, expected["num_workers"])
                    self.assertEqual(cfg.num_epochs, expected["num_epochs"])
                    self.assertEqual(cfg.max_steps, expected["max_steps"])
                    self.assertEqual(cfg.gradient_accumulation_steps, 1)
                    self.assertEqual(cfg.learning_rate, 1.0e-4)
                    self.assertEqual(cfg.lr_scheduler_type, "cosine")
                    self.assertEqual(
                        len(cfg.data.train.shape_meta.images),
                        expected["num_cameras"],
                    )
                    self.assertEqual(
                        list(cfg.data.train.video_size), expected["video_size"]
                    )
                    self.assertEqual(cfg.data.train.num_frames, 33)
                    self.assertEqual(cfg.data.train.action_video_freq_ratio, 4)

                    self.assertEqual(model.anticip_fm_lambda, 1.0)
                    self.assertEqual(model.anticip_fm_width, 512)
                    self.assertEqual(model.anticip_fm_blocks, 2)
                    self.assertEqual(model.anticip_fm_lambda_final, 0.2)

                    self.assertEqual(model.anticip_delta_mix, 0.8)
                    self.assertEqual(model.anticip_lambda_final, 0.2)
                    self.assertEqual(model.anticip_cond_noise_p, 0.3)
                    self.assertEqual(model.anticip_cond_noise_sigma, 0.06)
                    self.assertTrue(model.anticip_cond_noise_action_free)

                    factory_args = set(signature(create_rift).parameters)
                    config_args = set(model) - {"_target_"}
                    self.assertEqual(
                        factory_args,
                        config_args | {"model_dtype", "device"},
                    )


if __name__ == "__main__":
    unittest.main()
