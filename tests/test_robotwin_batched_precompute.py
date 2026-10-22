"""Check the batching boundary, tail handling and incompatible resume guards."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from precompute_robotwin_cache import (
    check_settings, encode_batch, encoding_settings, make_encoder, numerical_report,
    prefetched_batches,
)
from robotwin_vae_inputs import make_window
from rift.utils.vae_latent_cache import tensor_digest


class RecordingVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = [torch.tensor(0.), torch.tensor(1.)]
        self.model = SimpleNamespace(encode=self.encode_batch)
        self.batch_sizes = []
        self.eval().requires_grad_(False)

    def encode_batch(self, videos, scale):
        self.batch_sizes.append(len(videos))
        return videos * scale[1] - scale[0]


def options(**changes):
    return SimpleNamespace(**dict(dict(batch_size=4, encode_mode='batched',
                                      world_size=2, root=Path('/tmp/robotwin')), **changes))


def test_whole_batch_reaches_model_once_and_tail_is_padded():
    vae = RecordingVae()
    encoder = make_encoder(vae, options())
    videos = torch.randn(3, 3, 9, 8, 8)
    actual = encode_batch(encoder, videos, 4)
    assert vae.batch_sizes == [4]
    torch.testing.assert_close(actual, videos, rtol=0, atol=0)
    assert actual.shape[0] == 3
    assert not actual.requires_grad


def test_full_batch_is_not_split_into_single_windows():
    vae = RecordingVae()
    encoder = make_encoder(vae, options())
    videos = torch.randn(4, 3, 9, 8, 8)
    encode_batch(encoder, videos, 4)
    assert vae.batch_sizes == [4]


@pytest.mark.parametrize('change', [dict(batch_size=8), dict(encode_mode='cudagraphs'),
                                  dict(world_size=3), dict(root=Path('/tmp/other'))])
def test_incompatible_resume_is_rejected(change):
    args = options()
    settings = dict(world_size=2, source_root='/tmp/robotwin', encoding=encoding_settings(args))
    check_settings(args, settings)
    with pytest.raises(RuntimeError, match='separate run'):
        check_settings(options(**change), settings)


def test_old_sequential_run_cannot_be_silently_resumed_as_batched():
    with pytest.raises(RuntimeError, match='encoding settings changed'):
        check_settings(options(), dict(world_size=2, source_root='/tmp/robotwin'))


def test_numerical_audit_accepts_small_bf16_error_and_rejects_bad_outputs():
    reference = torch.ones(4, 48, 3, 2, 2, dtype=torch.bfloat16)
    report = numerical_report(reference + 0.003, reference)
    assert report['normalized_rmse'] <= report['nrmse_limit']
    for bad in (reference + 1, reference * float('nan')):
        with pytest.raises(RuntimeError, match='numerical check failed'):
            numerical_report(bad, reference)


def test_invalid_tail_batch_is_rejected():
    encoder = make_encoder(RecordingVae(), options())
    with pytest.raises(ValueError, match='1..batch_size'):
        encode_batch(encoder, torch.zeros(5, 3, 9, 8, 8), 4)


def test_prefetch_preserves_original_float32_keys_values_and_tail_order():
    canvas = torch.rand(7, 3, 384, 320)
    expected = [make_window(canvas, frame) for frame in range(len(canvas))]
    batches = list(prefetched_batches(canvas, 4, pin_memory=False))
    assert [len(batch[0]) for batch in batches] == [4, 3]
    actual = torch.cat([batch[0] for batch in batches])
    torch.testing.assert_close(actual, torch.stack(expected).to(torch.bfloat16), rtol=0, atol=0)
    assert [key for batch in batches for key in batch[1]] == [tensor_digest(window) for window in expected]
