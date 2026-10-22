"""Validate and reference complete legacy episodes, then balance missing work.

The source safetensors stay byte-for-byte unchanged in their original namespace.
Only index hardlinks and small completion/provenance records are added.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from precompute_robotwin_cache import (
    PREPROCESSING, assigned_episodes, index_path, packed_path, save_json, sha, verify_done,
)
from rift.utils.robotwin_cache_reuse import CATALOG, FORMAT, load_catalog
from rift.utils.vae_latent_cache import identity_hash, shared_cache_identity


def source_record(source_run, cache_dir, settings, destination_identity):
    source_settings = json.loads((source_run / 'settings.json').read_text())
    smoke = json.loads((source_run / 'smoke.json').read_text())
    if smoke['status'] != 'passed' or smoke['preprocessing'] != PREPROCESSING:
        raise RuntimeError('Legacy source smoke or preprocessing differs')
    # This importer authorizes the known sequential encoder, not arbitrary modes.
    if source_settings.get('encoding') is not None:
        raise RuntimeError('Expected an original sequential source run')
    namespace = smoke['namespace']
    identity = json.loads((cache_dir / namespace / 'identity.json').read_text())
    if identity_hash(identity) != namespace or identity != smoke['identity']:
        raise RuntimeError('Legacy source identity differs')
    if shared_cache_identity(identity) != shared_cache_identity(destination_identity):
        raise RuntimeError('Legacy VAE weights, scales or dtype differ')
    for field, filename in [('episodes_sha256', 'episodes.json'),
                            ('fingerprints_sha256', 'input_fingerprints.json')]:
        if source_settings[field] != settings[field] or sha(source_run / filename) != settings[field]:
            raise RuntimeError('Legacy source input inventory differs')
    if source_settings['source_root'] != settings['source_root']:
        raise RuntimeError('Legacy dataset source differs')
    return namespace, dict(identity=identity, preprocessing=PREPROCESSING,
        source_run=str(source_run), source_root=source_settings['source_root'],
        episodes_sha256=source_settings['episodes_sha256'],
        fingerprints_sha256=source_settings['fingerprints_sha256'],
        encoding_origin='legacy_sequential')


def balance_assignments(episodes, completed, world_size):
    """Balance missing windows independently of already completed episode sizes."""
    cached = [e for e in episodes if (e['dataset'], e['episode_index']) in completed]
    missing = [e for e in episodes if (e['dataset'], e['episode_index']) not in completed]
    cached_ranks = assigned_episodes(cached, world_size)
    missing_ranks = assigned_episodes(missing, world_size)
    return [[(e['dataset'], e['episode_index']) for e in cached_ranks[r] + missing_ranks[r]]
            for r in range(world_size)]


def reuse(args):
    settings = json.loads((args.run_dir / 'settings.json').read_text())
    smoke = json.loads((args.run_dir / 'smoke.json').read_text())
    namespace, identity = smoke['namespace'], smoke['identity']
    if smoke['status'] != 'passed':
        raise RuntimeError('Destination smoke has not passed')
    stored_identity = json.loads((args.cache_dir / namespace / 'identity.json').read_text())
    if identity_hash(stored_identity) != namespace or stored_identity != identity:
        raise RuntimeError('Destination cache identity differs')
    for field, filename in [('episodes_sha256', 'episodes.json'),
                            ('fingerprints_sha256', 'input_fingerprints.json')]:
        if sha(args.run_dir / filename) != settings[field]:
            raise RuntimeError('Destination inventory changed')
    source_namespace, record = source_record(args.source_run, args.cache_dir, settings, identity)
    catalog = load_catalog(args.run_dir, args.cache_dir, namespace, PREPROCESSING, unpublished=True)
    previous = catalog['sources'].get(source_namespace)
    if previous is not None and previous != record:
        raise RuntimeError('Legacy source provenance changed')
    catalog['sources'][source_namespace] = record
    args._reuse_catalog = catalog
    source_args = SimpleNamespace(run_dir=args.source_run, cache_dir=args.cache_dir)
    episodes = json.loads((args.run_dir / 'episodes.json').read_text())
    fingerprints = json.loads((args.run_dir / 'input_fingerprints.json').read_text())
    existing, candidates = [], []
    for episode in episodes:
        if index_path(args, episode).with_suffix('.done.json').exists():
            existing.append(episode)
        elif index_path(source_args, episode).with_suffix('.done.json').exists():
            candidates.append(episode)
    progress = dict(status='verifying', existing_episodes=len(existing), candidate_episodes=len(candidates),
                    verified_episodes=0, verified_windows=0, imported_episodes=0, imported_windows=0)
    progress_path = args.run_dir / 'reuse_progress.json'
    save_json(progress_path, progress)

    def validate(episode, original):
        source = source_args if original else args
        ns = source_namespace if original else namespace
        done, _ = verify_done(source, episode, ns, fingerprints, read_payload=True)
        return episode, done, original

    validated = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(validate, e, False) for e in existing]
        pending.extend(pool.submit(validate, e, True) for e in candidates)
        for future in as_completed(pending):
            item = future.result()
            validated.append(item)
            progress['verified_episodes'] += 1
            progress['verified_windows'] += item[0]['length']
            if progress['verified_episodes'] % 100 == 0:
                save_json(progress_path, progress)
                print('REUSE_VERIFY', json.dumps(progress), flush=True)
    # No completion records are published before every selected payload passes.
    save_json(args.run_dir / CATALOG, catalog)
    imported = []
    completed = set()
    for episode, done, original in validated:
        completed.add((episode['dataset'], episode['episode_index']))
        if not original:
            continue
        source_index = index_path(source_args, episode)
        target_index = index_path(args, episode)
        target_index.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_index, target_index)
        except FileExistsError:
            if sha(target_index) != done['index_sha256']:
                raise RuntimeError('Existing destination index differs')
        copied = dict(done, namespace=namespace, payload_namespace=source_namespace,
                      encoding_origin='legacy_sequential', reuse_source_run=str(args.source_run),
                      source_done_sha256=sha(source_index.with_suffix('.done.json')))
        save_json(target_index.with_suffix('.done.json'), copied)
        progress['imported_episodes'] += 1
        progress['imported_windows'] += episode['length']
        imported.append(dict(dataset=episode['dataset'], episode_index=episode['episode_index'],
                             windows=episode['length'], payload_namespace=source_namespace,
                             packed_sha256=done['packed_sha256']))
    assignments = balance_assignments(episodes, completed, settings['world_size'])
    save_json(args.run_dir / 'assignments.json', dict(world_size=settings['world_size'],
        episodes_sha256=settings['episodes_sha256'], assignments=assignments))
    settings.update(reuse_sources_sha256=sha(args.run_dir / CATALOG),
                    assignments_sha256=sha(args.run_dir / 'assignments.json'))
    save_json(args.run_dir / 'settings.json', settings)
    progress.update(status='passed', completed_windows=sum(e['length'] for e, _, _ in validated),
                    remaining_windows=sum(e['length'] for e in episodes if (e['dataset'], e['episode_index']) not in completed),
                    destination_namespace=namespace, source_namespace=source_namespace)
    save_json(args.run_dir / 'reuse_report.json', dict(progress, imported=imported))
    save_json(progress_path, progress)
    print('REUSE_COMPLETE', json.dumps(progress), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.source_run.resolve() == args.run_dir.resolve():
        parser.error('Use a distinct source run and a positive verification worker count')
    args.run_dir, args.cache_dir, args.source_run = (p.resolve() for p in (args.run_dir, args.cache_dir, args.source_run))
    torch.set_num_threads(1)
    settings = json.loads((args.run_dir / 'settings.json').read_text())
    # Hold every worker lock through validation, references and assignment publication.
    with ExitStack() as stack:
        for rank in range(settings['world_size']):
            lock = stack.enter_context((args.run_dir / f'worker_rank{rank}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reuse(args)


if __name__ == '__main__':
    main()
