"""Generate and audit RoboTwin window latents as episode safetensors.

No per-window .pt files are written. A validation manifest is published only
once every original episode, input fingerprint and packed payload is checked.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precompute_robocoin_vae import save_json, input_fingerprints, collate_windows, assigned_episodes
from rift.models.wan22.helpers.vae_encoding import OnlineVaeEncoder
from rift.utils.vae_latent_cache import frozen_vae_identity, identity_hash
from robotwin_vae_inputs import discover_episodes, prepare_episode, make_window, validate_episode_inputs
from rift.utils.vae_latent_cache import tensor_digest
from rift.utils.robotwin_episode_cache import pack_episode, load_window, validate_header, LATENT_SHAPE

CONTRACT = 'robotwin_3cam384_v1'
PREPROCESSING = dict(contract=CONTRACT, num_frames=33, video_offsets=list(range(0,33,4)),
    per_camera_resize=[240,320], canvas=[384,320], normalization=dict(mean=0.5,std=0.5),
    window_policy='each frame, repeat final frame at episode boundary')


def encoding_settings(args):
    return dict(implementation='OnlineVaeEncoder', mode=args.encode_mode,
                batch_size=args.batch_size, tail_policy='repeat_last_to_full_batch')


def check_settings(args, settings):
    if (args.world_size != settings['world_size']
            or args.root.resolve() != Path(settings['source_root'])
            or settings.get('encoding') != encoding_settings(args)):
        raise RuntimeError('Run source, worker count or encoding settings changed; use a separate run directory')


def load_cache(args):
    from types import SimpleNamespace
    from rift.models.wan22.helpers.loader import _load_registered_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    vae = _load_registered_model(str(args.vae_weights), 'wan_video_vae', torch.bfloat16, str(device))
    vae.eval().requires_grad_(False)
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Use native BF16 inputs and weights without autocast')
    # Batched BF16 convolutions need not be bitwise equal to sequential ones.
    # Keep the execution contract in a separate namespace, leaving old caches intact.
    identity = dict(frozen_vae_identity(vae, torch.bfloat16),
                    precompute_encoding=encoding_settings(args))
    namespace = identity_hash(identity)
    directory = args.cache_dir / namespace
    identity_path = directory / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Existing cache identity differs')
    save_json(identity_path, identity)
    return vae, SimpleNamespace(identity=identity, namespace=namespace, directory=directory), device


def make_encoder(vae, args):
    return OnlineVaeEncoder(vae, cudagraphs=args.encode_mode == 'cudagraphs',
                            batch_size=args.batch_size)


def encode_batch(encoder, videos, batch_size):
    """Keep one GPU shape, including the last batch and smoke/parity samples."""
    count = videos.shape[0]
    if not 0 < count <= batch_size:
        raise ValueError('Input batch must contain 1..batch_size windows')
    if count < batch_size:
        videos = torch.cat([videos, videos[-1:].expand(batch_size - count, -1, -1, -1, -1)])
    return encoder(videos)[:count]


def prefetched_batches(canvas, batch_size, *, pin_memory):
    """Prepare one CPU batch ahead while the main thread runs the GPU encoder."""
    def prepare(start):
        before = time.monotonic()
        windows = [make_window(canvas, frame)
                   for frame in range(start, min(start + batch_size, len(canvas)))]
        keys = [tensor_digest(window) for window in windows]
        # Hash the original float32 pixels, then transfer only BF16 input bytes.
        videos = torch.stack(windows).to(dtype=torch.bfloat16)
        if pin_memory:
            videos = videos.pin_memory()
        return videos, keys, time.monotonic() - before

    with ThreadPoolExecutor(max_workers=1) as pool:
        starts = iter(range(0, len(canvas), batch_size))
        following = next(starts, None)
        future = pool.submit(prepare, following) if following is not None else None
        while future is not None:
            before = time.monotonic()
            videos, keys, seconds = future.result()
            waited = time.monotonic() - before
            following = next(starts, None)
            future = pool.submit(prepare, following) if following is not None else None
            yield videos, keys, seconds, waited


def numerical_report(actual, reference):
    actual, reference = actual.float().cpu(), reference.float().cpu()
    difference = actual - reference
    normalized_rmse = (difference.square().mean().sqrt()
                       / reference.square().mean().sqrt().clamp_min(1e-8)).item()
    maximum = difference.abs().max().item()
    if not torch.isfinite(actual).all() or normalized_rmse > 0.01 or maximum > 0.125:
        raise RuntimeError(f'Batched/reference VAE numerical check failed: NRMSE={normalized_rmse}, max={maximum}')
    return dict(bitwise_equal=torch.equal(actual, reference), normalized_rmse=normalized_rmse,
                max_abs_error=maximum, nrmse_limit=0.01, max_abs_error_limit=0.125)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def check_sources(episode, fingerprints):
    for p in [episode['parquet_path'],*episode['video_paths'].values()]:
        st=Path(p).stat()
        if fingerprints[p] != dict(size=st.st_size,mtime_ns=st.st_mtime_ns):
            raise RuntimeError(f'Input changed: {p}')

def packed_path(args, namespace, episode):
    return args.cache_dir/namespace/'episodes'/episode['dataset']/f"episode_{episode['episode_index']:06d}.safetensors"

def index_path(args, episode):
    return args.run_dir/'indices'/episode['dataset']/f"episode_{episode['episode_index']:06d}.jsonl"

def initialize(args):
    if (args.run_dir/'settings.json').exists():
        settings=json.loads((args.run_dir/'settings.json').read_text())
        check_settings(args, settings)
        return
    episodes=discover_episodes(args.root.resolve())
    if len(episodes)!=27500 or sum(e['length'] for e in episodes)!=6075103:
        raise RuntimeError('Expected the complete RIFT RoboTwin release')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    save_json(args.run_dir/'episodes.json',episodes)
    print('INVENTORY_READY',len(episodes),flush=True)
    save_json(args.run_dir/'input_fingerprints.json',input_fingerprints(episodes))
    save_json(args.run_dir/'settings.json',dict(**PREPROCESSING,source_root=str(args.root.resolve()),
        world_size=args.world_size,cache_root=str(args.cache_dir),batch_size=args.batch_size,
        encoding=encoding_settings(args),
        episodes_sha256=sha(args.run_dir/'episodes.json'),
        fingerprints_sha256=sha(args.run_dir/'input_fingerprints.json')))
    print('INITIALIZED',flush=True)

@torch.no_grad()
def smoke(args, episodes, vae, cache, device):
    reports=[]; examples=[]; seconds=0.; count=0
    encoder = make_encoder(vae, args)
    for number in (0,50,13750,27499):
        e=next(e for e in episodes if e['episode_index']==number)
        canvas=prepare_episode(e)
        report=validate_episode_inputs(e,canvas)
        starts=sorted({0,len(canvas)//2,len(canvas)-1})
        windows = [make_window(canvas,s) for s in starts]
        videos,keys=collate_windows(windows,device)
        reference = vae.encode(videos, device=device, tiled=False).cpu()
        # Graph capture/allocator warmup must not be counted as steady throughput.
        encode_batch(encoder, videos, args.batch_size)
        torch.cuda.synchronize(); before=time.monotonic()
        latents=encode_batch(encoder,videos,args.batch_size).cpu().contiguous()
        torch.cuda.synchronize(); elapsed=time.monotonic()-before
        seconds+=elapsed; count+=len(starts)
        if tuple(latents.shape[1:]) != LATENT_SHAPE or latents.dtype!=torch.bfloat16 or not torch.isfinite(latents).all():
            raise RuntimeError('Unexpected fresh VAE output')
        report['reference_comparison'] = numerical_report(latents, reference)
        out=args.run_dir/'smoke'/f'episode_{number}.safetensors'
        index_sha=hashlib.sha256(json.dumps(starts).encode()).hexdigest()
        checksums=[tensor_digest(t) for t in latents]
        if not out.exists():
            pack_episode(out,latents,namespace=cache.namespace,dataset=e['dataset'],episode_index=number,
                index_sha256=index_sha,window_sha256=checksums,window_keys=keys)
        for i,(start,key) in enumerate(zip(starts,keys)):
            cached=load_window(out,i,key=key,namespace=cache.namespace,dataset=e['dataset'],episode_index=number,
                index_sha256=index_sha,window_count=len(starts))
            if not torch.equal(cached,latents[i]): raise RuntimeError('Packed latent round-trip failed')
        example=args.run_dir/'examples'/f'episode_{number}.pt'; example.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(videos=torch.stack(windows),latents=latents,keys=keys),example)
        examples.append(str(example))
        report.update(latent_shape=list(LATENT_SHAPE),latent_dtype=str(latents.dtype),packed_bitwise_equal=True,encode_seconds=elapsed)
        reports.append(report)
        print('SMOKE_PASS',number,elapsed,flush=True)
        del canvas,videos,latents
    save_json(args.run_dir/'smoke.json',dict(status='passed',namespace=cache.namespace,identity=cache.identity,
        preprocessing=PREPROCESSING,encoding=encoding_settings(args),reports=reports,examples=examples,
        mean_encode_seconds_per_window=seconds/count))
    print('SMOKE_COMPLETE',seconds/count,flush=True)

def verify_done(args,e,namespace,fingerprints,read_payload=True):
    check_sources(e,fingerprints)
    index=index_path(args,e); done=json.loads(index.with_suffix('.done.json').read_text())
    raw=index.read_bytes(); rows=[json.loads(line) for line in raw.splitlines()]
    if (done['namespace']!=namespace or done['windows']!=e['length'] or done['episode_index']!=e['episode_index']
        or done['dataset']!=e['dataset'] or done['index_sha256']!=hashlib.sha256(raw).hexdigest()
        or [r['start_frame'] for r in rows]!=list(range(e['length']))):
        raise RuntimeError(f'Invalid episode completion: {index}')
    path=packed_path(args,namespace,e)
    if sha(path)!=done['packed_sha256']: raise RuntimeError(f'Corrupt packed episode: {path}')
    with safe_open(path,framework='pt',device='cpu') as h:
        keys,checksums=validate_header(h,namespace=namespace,dataset=e['dataset'],episode_index=e['episode_index'],
            index_sha256=done['index_sha256'],window_count=e['length'])
        if list(keys)!=[r['key'] for r in rows]: raise RuntimeError('Packed/index keys differ')
        if read_payload:
            latents=h.get_tensor('latents')
            for t,digest in zip(latents,checksums):
                if not torch.isfinite(t).all() or tensor_digest(t)!=digest: raise RuntimeError(f'Corrupt window: {path}')
    return done,keys

@torch.no_grad()
def worker(args,episodes,vae,cache,device):
    smoke_report=json.loads((args.run_dir/'smoke.json').read_text())
    if (smoke_report['status']!='passed' or smoke_report['namespace']!=cache.namespace
            or smoke_report.get('encoding') != encoding_settings(args)):
        raise RuntimeError('VAE identity or encoding mismatch')
    encoder = make_encoder(vae, args)
    for path in smoke_report['examples']:
        sample=torch.load(path,map_location='cpu',weights_only=True)
        videos,keys=collate_windows(list(sample['videos']),device)
        fresh=encode_batch(encoder,videos,args.batch_size).cpu()
        if keys!=sample['keys'] or not torch.equal(fresh,sample['latents']):
            raise RuntimeError(f'Cross GPU parity failed: {path}')
    save_json(args.run_dir/f'parity_rank{args.rank}.json',dict(status='passed',namespace=cache.namespace,
        gpu=str(torch.cuda.get_device_properties(device)),hostname=os.uname().nodename,
        encoding=encoding_settings(args)))
    fingerprints=json.loads((args.run_dir/'input_fingerprints.json').read_text())
    assigned=assigned_episodes(episodes,args.world_size)[args.rank]
    state=dict(state='running',rank=args.rank,namespace=cache.namespace,total_windows=sum(e['length'] for e in assigned),
        total_episodes=len(assigned),completed_windows=0,completed_episodes=0,pid=os.getpid(),host=os.uname().nodename,
        encoding=encoding_settings(args),stage_seconds=dict(input_wait=0.,collate=0.,batch_wait=0.,encode=0.,write=0.))
    status=args.run_dir/f'rank{args.rank}.json'; started=time.monotonic();save_json(status,state)
    todo=[e for e in assigned if not index_path(args,e).with_suffix('.done.json').exists()]
    try:
        # A single producer bounds RAM and overlaps sequential decoding with VAE work.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=iter(todo); following=next(pending,None)
            future=pool.submit(prepare_episode,following) if following else None
            for e in assigned:
                index=index_path(args,e);done_path=index.with_suffix('.done.json');path=packed_path(args,cache.namespace,e)
                check_sources(e,fingerprints)
                if done_path.exists():
                    verify_done(args,e,cache.namespace,fingerprints)
                else:
                    before=time.monotonic()
                    canvas=future.result(); following=next(pending,None)
                    state['stage_seconds']['input_wait'] += time.monotonic()-before
                    future=pool.submit(prepare_episode,following) if following else None
                    latents=[];keys=[]
                    for cpu_videos,batchkeys,prepared,waited in prefetched_batches(
                            canvas,args.batch_size,pin_memory=True):
                        state['stage_seconds']['collate'] += prepared
                        state['stage_seconds']['batch_wait'] += waited
                        before=time.monotonic()
                        videos=cpu_videos.to(device=device,non_blocking=True)
                        result=encode_batch(encoder,videos,args.batch_size).to('cpu').contiguous()
                        state['stage_seconds']['encode'] += time.monotonic()-before
                        if result.dtype!=torch.bfloat16 or tuple(result.shape[1:])!=LATENT_SHAPE or not torch.isfinite(result).all():
                            raise RuntimeError(f'Invalid encoded latents: {e["episode_index"]}')
                        latents.append(result); keys.extend(batchkeys)
                    before=time.monotonic()
                    latents=torch.cat(latents).contiguous(); del canvas,videos,result
                    raw=''.join(json.dumps(dict(start_frame=i,key=k),separators=(',',':'))+'\n' for i,k in enumerate(keys)).encode()
                    digest=hashlib.sha256(raw).hexdigest();index.parent.mkdir(parents=True,exist_ok=True)
                    temporary=index.with_suffix('.tmp');temporary.write_bytes(raw);os.replace(temporary,index)
                    if not path.exists():
                        filehash=pack_episode(path,latents,namespace=cache.namespace,dataset=e['dataset'],episode_index=e['episode_index'],
                            index_sha256=digest,window_sha256=[tensor_digest(t) for t in latents],window_keys=keys)
                    else:
                        # Recover a file published before an interrupted completion write.
                        for i,key in enumerate(keys):
                            cached=load_window(path,i,key=key,namespace=cache.namespace,dataset=e['dataset'],episode_index=e['episode_index'],index_sha256=digest,window_count=len(keys))
                            if not torch.equal(cached,latents[i]): raise RuntimeError('Interrupted episode differs from recomputation')
                        filehash=sha(path)
                    save_json(done_path,dict(namespace=cache.namespace,dataset=e['dataset'],episode_index=e['episode_index'],
                        windows=len(keys),index_sha256=digest,packed_sha256=filehash,all_window_checksums_verified=True))
                    state['stage_seconds']['write'] += time.monotonic()-before
                    del latents
                state['completed_windows']+=e['length'];state['completed_episodes']+=1
                state['elapsed_seconds']=time.monotonic()-started
                state['windows_per_second']=state['completed_windows']/state['elapsed_seconds']
                state['peak_gpu_gib']=torch.cuda.max_memory_allocated()/(1<<30)
                save_json(status,state)
                if state['completed_episodes']%10==0: print('PROGRESS',json.dumps(state),flush=True)
        state['state']='complete';save_json(status,state);print('WORKER_COMPLETE',args.rank,flush=True)
    except BaseException:
        state.update(state='failed',error=traceback.format_exc());save_json(status,state);raise

def publish(args,episodes):
    smoke_report=json.loads((args.run_dir/'smoke.json').read_text());namespace=smoke_report['namespace']
    fingerprints=json.loads((args.run_dir/'input_fingerprints.json').read_text())
    settings=json.loads((args.run_dir/'settings.json').read_text())
    if sha(args.run_dir/'episodes.json')!=settings['episodes_sha256'] or sha(args.run_dir/'input_fingerprints.json')!=settings['fingerprints_sha256']:
        raise RuntimeError('Run inventory changed')
    for rank in range(args.world_size):
        status=json.loads((args.run_dir/f'rank{rank}.json').read_text())
        if status['state']!='complete' or status['completed_windows']!=status['total_windows']: raise RuntimeError(f'Incomplete rank {rank}')
        if status.get('encoding') != settings['encoding']:
            raise RuntimeError(f'Encoding configuration differs for rank {rank}')
    for name,digest in episodes[0]['meta_sha256'].items():
        if sha(Path(settings['source_root'])/'meta'/name)!=digest: raise RuntimeError(f'Source metadata changed: {name}')
    manifest=[];offset=0; temporary=args.run_dir/'keys.bin.tmp'
    with temporary.open('wb') as f:
        for number,e in enumerate(episodes):
            done,keys=verify_done(args,e,namespace,fingerprints,read_payload=True)
            f.write(bytes.fromhex(''.join(keys)))
            manifest.append(dict(dataset=e['dataset'],episode_index=e['episode_index'],windows=e['length'],
                key_offset=offset,index_sha256=done['index_sha256'],packed_sha256=done['packed_sha256']))
            offset+=e['length']*32
            if number%100==0: print('AUDIT',number,len(episodes),flush=True)
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,args.run_dir/'keys.bin')
    save_json(args.run_dir/'packed_manifest.json',manifest)
    save_json(args.run_dir/'validation.json',dict(status='passed',namespace=namespace,identity=smoke_report['identity'],
        preprocessing=PREPROCESSING,encoding=settings['encoding'],latent_shape=list(LATENT_SHAPE),latent_dtype='torch.bfloat16',
        episode_count=len(episodes),window_count=sum(e['length'] for e in episodes),all_starts_covered_once=True,
        all_indices_sha256_verified=True,all_packed_payloads_verified=True,all_source_fingerprints_verified=True,
        episodes_sha256=sha(args.run_dir/'episodes.json'),fingerprints_sha256=sha(args.run_dir/'input_fingerprints.json'),
        keys_sha256=sha(args.run_dir/'keys.bin'),manifest_sha256=sha(args.run_dir/'packed_manifest.json')))
    print('VAE_CACHE_VALIDATED',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','smoke','worker','publish'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--vae-weights',type=Path)
    p.add_argument('--world-size',type=int,default=10);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--encode-mode',choices=['batched','cudagraphs'],default='batched')
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.world_size < 1:
        p.error('batch size, CPU threads and worker count must be positive')
    torch.set_num_threads(args.cpu_threads)
    if not 0<=args.rank<args.world_size: p.error('rank outside worker count')
    if args.mode=='init': initialize(args);return
    settings=json.loads((args.run_dir/'settings.json').read_text())
    check_settings(args,settings)
    episodes=json.loads((args.run_dir/'episodes.json').read_text())
    if args.mode=='publish':publish(args,episodes);return
    # A lock survives throughout model loading and work, preventing duplicate writers.
    with (args.run_dir/f'{args.mode}_rank{args.rank}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        vae,cache,device=load_cache(args)
        (smoke if args.mode=='smoke' else worker)(args,episodes,vae,cache,device)
if __name__=='__main__':main()
