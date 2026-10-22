"""Authorize unchanged RoboTwin payloads from other encoder execution modes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .vae_latent_cache import identity_hash, shared_cache_identity

CATALOG = 'reuse_sources.json'
FORMAT = 'robotwin_episode_reuse_v1'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(run_dir, cache_dir, namespace, preprocessing, *, unpublished=False):
    """Validate source weights, preprocessing and immutable input inventories."""
    run_dir, cache_dir = Path(run_dir), Path(cache_dir)
    path = run_dir / CATALOG
    if not path.exists():
        return {'format': FORMAT, 'destination_namespace': namespace, 'sources': {}}
    settings = json.loads((run_dir / 'settings.json').read_text())
    if not unpublished and settings.get('reuse_sources_sha256') != file_sha(path):
        raise RuntimeError('Reuse catalog has not been committed or its checksum changed')
    catalog = json.loads(path.read_text())
    if catalog.get('format') != FORMAT or catalog.get('destination_namespace') != namespace:
        raise RuntimeError('Reuse catalog destination or format differs')
    identity = json.loads((cache_dir / namespace / 'identity.json').read_text())
    if identity_hash(identity) != namespace:
        raise RuntimeError('Destination cache identity differs')
    for source_namespace, source in catalog['sources'].items():
        if re.fullmatch('[0-9a-f]{64}', source_namespace) is None:
            raise RuntimeError('Invalid source namespace')
        stored = json.loads((cache_dir / source_namespace / 'identity.json').read_text())
        if identity_hash(stored) != source_namespace or stored != source['identity']:
            raise RuntimeError('Reuse source identity differs')
        if shared_cache_identity(stored) != shared_cache_identity(identity):
            raise RuntimeError('Reuse source VAE weights, scales or dtype differ')
        if source['preprocessing'] != preprocessing:
            raise RuntimeError('Reuse source preprocessing differs')
        for field in ('source_root', 'episodes_sha256', 'fingerprints_sha256'):
            if source[field] != settings[field]:
                raise RuntimeError(f'Reuse source input inventory differs: {field}')
    return catalog


def payload_namespace(done, namespace, catalog):
    """A foreign header is accepted only with explicit, validated provenance."""
    payload = done.get('payload_namespace', namespace)
    if payload != namespace:
        if payload not in catalog['sources']:
            raise RuntimeError('Unapproved reused payload namespace')
        if done.get('encoding_origin') != 'legacy_sequential':
            raise RuntimeError('Reused payload encoding origin is missing')
    return payload
