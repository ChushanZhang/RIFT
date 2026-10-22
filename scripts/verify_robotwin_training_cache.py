"""Audit cached/raw training samples after VAE and text publication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--text-dir', type=Path, required=True)
    parser.add_argument('--text-run-dir', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    if json.loads((args.text_run_dir / 'text_validation.json').read_text())['status'] != 'passed':
        raise RuntimeError('Text cache has not passed validation')
    if json.loads((args.run_dir / 'validation.json').read_text())['status'] != 'passed':
        raise RuntimeError('VAE cache has not passed validation')
    os.environ.update(ROBOTWIN_RUN_DIR=str(args.run_dir.resolve()),
                      ROBOTWIN_CACHE_DIR=str(args.cache_dir.resolve()),
                      ROBOTWIN_TEXT_DIR=str(args.text_dir.resolve()), CUDA_VISIBLE_DEVICES='')
    with tempfile.TemporaryDirectory(prefix='reader-check-', dir=args.run_dir) as temporary:
        os.environ.update(HF_HOME=temporary + '/hf', HF_DATASETS_CACHE=temporary + '/datasets',
                          HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from precompute_robocoin_vae import save_json
        from rift.utils.vae_latent_cache import VaeCacheCollator, tensor_digest
        torch.set_num_threads(2)
        reports = []
        for task, clean in [('robotwin_rift_3cam_384_cached', False),
                            ('robotwin_clean2random_rift_3cam_384_cached', True)]:
            with initialize_config_dir(config_dir=str(code / 'configs'), version_base=None):
                cfg = compose(config_name='train', overrides=['task=' + task])
            node = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
            node.dataset_dirs = [str((code / str(path)).absolute()) for path in node.dataset_dirs]
            node.pretrained_norm_stats = str((code / str(node.pretrained_norm_stats)).absolute())
            with patch('rift.utils.misc.get_work_dir', return_value=temporary):
                dataset = instantiate(node)
            child = dataset.lerobot_dataset.multi_dataset._datasets[0]
            if clean:
                assert len(dataset) == 549787
                assert list(child.episodes) == [i for i in range(27500) if i % 550 < 50]
            samples = []
            for slot in (0, len(child.episodes) // 2, len(child.episodes) - 1):
                idx = int(child.episode_data_index['from'][slot])
                if slot == len(child.episodes) - 1:
                    idx = int(child.episode_data_index['to'][slot]) - 1
                cached = dataset[idx]
                assert 'video_latents' in cached and 'video' not in cached
                assert cached['video_latents'].shape == (48, 3, 24, 20)
                batch = VaeCacheCollator()([cached])
                assert batch['video_latents'].shape == (1, 48, 3, 24, 20)
                reader = dataset._latent_reader
                dataset._latent_reader = None
                dataset.lerobot_dataset._set_return_images(True)
                try:
                    raw = dataset[idx]
                finally:
                    dataset._latent_reader = reader
                    dataset.lerobot_dataset._set_return_images(False)
                assert raw['video'].shape == (3, 9, 384, 320)
                assert tensor_digest(raw['video']) == cached['vae_cache_keys']
                for key in ('action', 'proprio', 'action_is_pad', 'image_is_pad',
                            'proprio_is_pad', 'context', 'context_mask'):
                    assert torch.equal(cached[key], raw[key]), (task, key)
                samples.append(dict(episode_index=int(child.episodes[slot]),
                                    nonvisual_bitwise_equal=True))
            reports.append(dict(task=task, frames=len(dataset), episodes=len(child.episodes), samples=samples))
            del dataset
        report = dict(status='passed', checks=reports,
                      completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        save_json(args.run_dir / 'training_reader_validation.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
