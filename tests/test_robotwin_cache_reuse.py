"""Prove legacy reuse is zero-copy, preserves newer outputs and reads correctly."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from precompute_robotwin_cache import PREPROCESSING, sha, verify_done, worker_episodes
from reuse_robotwin_cache import balance_assignments, reuse, source_record
from rift.utils.robotwin_cache_reuse import load_catalog, payload_namespace
from rift.utils.robotwin_episode_cache import LATENT_SHAPE, pack_episode
from rift.utils.robotwin_indexed_vae_cache import RobotwinIndexedVaeLatentReader
from rift.utils.vae_latent_cache import identity_hash, tensor_digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def runs(tmp_path):
    torch.set_num_threads(1)
    cache, old, new, raw = (tmp_path / name for name in ('cache', 'old', 'new', 'raw'))
    raw.mkdir()
    common = dict(format=1, weights_sha256='a' * 64, scale=['b' * 64, 'c' * 64], input_dtype='torch.bfloat16')
    old_identity = dict(common, vae_class='legacy.WanVideoVAE38', encode_source_sha256='old')
    new_identity = dict(common, precompute_encoding=dict(mode='batched', batch_size=16))
    old_ns, new_ns = identity_hash(old_identity), identity_hash(new_identity)
    for ns, identity in [(old_ns, old_identity), (new_ns, new_identity)]:
        write_json(cache / ns / 'identity.json', identity)
    episodes, fingerprints = [], {}
    for episode_index in range(4):
        path = raw / f'{episode_index}.parquet'
        path.write_bytes(b'unchanged-source')
        stat = path.stat()
        fingerprints[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        episodes.append(dict(dataset='robotwin2.0', episode_index=episode_index, length=episode_index + 2,
                             parquet_path=str(path), video_paths={}))
    for run, ns, identity in [(old, old_ns, old_identity), (new, new_ns, new_identity)]:
        write_json(run / 'episodes.json', episodes)
        write_json(run / 'input_fingerprints.json', fingerprints)
        settings = dict(PREPROCESSING, source_root=str(raw), world_size=2,
                        episodes_sha256=sha(run / 'episodes.json'),
                        fingerprints_sha256=sha(run / 'input_fingerprints.json'))
        if run == new:
            settings['encoding'] = dict(mode='batched', batch_size=16)
        write_json(run / 'settings.json', settings)
        write_json(run / 'smoke.json', dict(status='passed', namespace=ns, identity=identity,
                                           preprocessing=PREPROCESSING))

    def complete(run, ns, episode, value):
        count = episode['length']
        keys = [hashlib.sha256(f"{episode['episode_index']}/{i}".encode()).hexdigest() for i in range(count)]
        index = run / 'indices/robotwin2.0' / f"episode_{episode['episode_index']:06d}.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(''.join(json.dumps(dict(start_frame=i, key=key)) + '\n' for i, key in enumerate(keys)))
        latents = torch.full((count, *LATENT_SHAPE), value, dtype=torch.bfloat16)
        path = cache / ns / 'episodes/robotwin2.0' / index.with_suffix('.safetensors').name
        digest = pack_episode(path, latents, namespace=ns, dataset='robotwin2.0', episode_index=episode['episode_index'],
                              index_sha256=sha(index), window_keys=keys,
                              window_sha256=[tensor_digest(t) for t in latents])
        write_json(index.with_suffix('.done.json'), dict(namespace=ns, dataset='robotwin2.0',
            episode_index=episode['episode_index'], windows=count, index_sha256=sha(index), packed_sha256=digest,
            all_window_checksums_verified=True))
        return path, latents, keys

    old0 = complete(old, old_ns, episodes[0], 1.)
    old1 = complete(old, old_ns, episodes[1], 2.)
    new0 = complete(new, new_ns, episodes[0], 3.)
    complete(new, new_ns, episodes[2], 4.)
    args = SimpleNamespace(run_dir=new, cache_dir=cache, source_run=old, workers=2)
    return SimpleNamespace(args=args, old=old, new=new, cache=cache, old_ns=old_ns, new_ns=new_ns,
                           old1=old1, old0=old0, new0=new0, episodes=episodes, fingerprints=fingerprints,
                           new_identity=new_identity)


def test_reuse_preserves_newer_files_and_references_original_payload(runs):
    legacy_bytes = runs.old1[0].read_bytes()
    newer_bytes = runs.new0[0].read_bytes()
    reuse(runs.args)
    assert runs.old1[0].read_bytes() == legacy_bytes
    assert runs.new0[0].read_bytes() == newer_bytes
    imported_index = runs.new / 'indices/robotwin2.0/episode_000001.jsonl'
    source_index = runs.old / 'indices/robotwin2.0/episode_000001.jsonl'
    assert imported_index.stat().st_ino == source_index.stat().st_ino
    assert not (runs.cache / runs.new_ns / 'episodes/robotwin2.0/episode_000001.safetensors').exists()
    fresh_args = SimpleNamespace(run_dir=runs.new, cache_dir=runs.cache)
    done, keys = verify_done(fresh_args, runs.episodes[1], runs.new_ns, runs.fingerprints)
    assert done['payload_namespace'] == runs.old_ns
    assert done['encoding_origin'] == 'legacy_sequential'
    assert list(keys) == runs.old1[2]
    report = json.loads((runs.new / 'reuse_report.json').read_text())
    assert report['imported_windows'] == 3
    assert report['completed_windows'] == 9
    assert report['remaining_windows'] == 5

    # The actual training reader must open the old file with its original header identity.
    reader = RobotwinIndexedVaeLatentReader.__new__(RobotwinIndexedVaeLatentReader)
    reader.namespace, reader.cache_root = runs.new_ns, runs.cache
    reader._reuse_catalog = load_catalog(runs.new, runs.cache, runs.new_ns, PREPROCESSING)
    reader._episodes = {1: dict(done)}
    reader._headers = None
    reader.key = lambda dataset, episode, frame: keys[frame]
    assert torch.equal(reader.load(0, 1, 2), runs.old1[1][2])

    assignments = []
    for rank in range(2):
        args = SimpleNamespace(run_dir=runs.new, world_size=2, rank=rank)
        assignments.extend(worker_episodes(args, runs.episodes))
    assert sorted(e['episode_index'] for e in assignments) == [0, 1, 2, 3]
    # A second import validates the same references and never duplicates computation.
    reuse(runs.args)
    assert json.loads((runs.new / 'reuse_report.json').read_text())['imported_windows'] == 0


def test_corrupted_legacy_payload_is_rejected_before_import(runs):
    path = runs.old1[0]
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(RuntimeError, match='Corrupt packed episode'):
        reuse(runs.args)
    assert not (runs.new / 'indices/robotwin2.0/episode_000001.done.json').exists()


def test_different_weights_are_not_compatible(runs):
    changed = dict(runs.new_identity, weights_sha256='d' * 64)
    settings = json.loads((runs.new / 'settings.json').read_text())
    with pytest.raises(RuntimeError, match='weights, scales or dtype'):
        source_record(runs.old, runs.cache, settings, changed)


def test_changed_catalog_and_unapproved_namespace_are_rejected(runs):
    reuse(runs.args)
    with pytest.raises(RuntimeError, match='Unapproved'):
        payload_namespace(dict(payload_namespace='f' * 64, encoding_origin='legacy_sequential'),
                          runs.new_ns, {'sources': {}})
    with (runs.new / 'reuse_sources.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(RuntimeError, match='checksum changed'):
        load_catalog(runs.new, runs.cache, runs.new_ns, PREPROCESSING)


def test_missing_work_is_balanced_separately_and_covered_once():
    episodes = [dict(dataset='robotwin2.0', episode_index=i, length=100 - i) for i in range(60)]
    completed = {('robotwin2.0', i) for i in range(30)}
    assignments = balance_assignments(episodes, completed, 4)
    flat = [tuple(key) for rank in assignments for key in rank]
    assert len(flat) == len(set(flat)) == 60
    lengths = {(e['dataset'], e['episode_index']):e['length'] for e in episodes}
    missing_loads = [sum(lengths[tuple(key)] for key in rank if tuple(key) not in completed)
                     for rank in assignments]
    assert max(missing_loads) - min(missing_loads) <= 70
