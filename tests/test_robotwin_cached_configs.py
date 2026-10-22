"""Both RoboTwin presets must consume the selected cache and shared text run."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize('task,clean', [('robotwin_rift_3cam_384_cached', False),
                                     ('robotwin_clean2random_rift_3cam_384_cached', True)])
def test_cache_environment_and_clean_membership(monkeypatch, task, clean):
    monkeypatch.setenv('ROBOTWIN_CACHE_DIR', '/tmp/robotwin/cache')
    monkeypatch.setenv('ROBOTWIN_RUN_DIR', '/tmp/robotwin/run')
    monkeypatch.setenv('ROBOTWIN_TEXT_DIR', '/tmp/robotwin/text')
    code = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(code / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=' + task])
    train = OmegaConf.to_container(cfg.data.train, resolve=True)
    assert train['vae_cache_dir'] == '/tmp/robotwin/cache'
    assert train['vae_cache_index_dir'] == '/tmp/robotwin/run'
    assert train['text_embedding_cache_dir'] == '/tmp/robotwin/text'
    assert OmegaConf.to_container(cfg, resolve=True)['vae_cache_dir'] == '/tmp/robotwin/cache'
    if clean:
        excluded = set(train['exclude_episodes']['robotwin2.0'])
        assert len(excluded) == 25000
        assert [i for i in range(27500) if i not in excluded] == [i for i in range(27500) if i % 550 < 50]
