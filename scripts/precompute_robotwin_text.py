"""Resumable, manually sharded RIFT T5 cache; native prompt names and payloads."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from precompute_text_embeds import _read_unique_prompts,_atomic_torch_save
from precompute_robocoin_vae import save_json
from rift.models.wan22.helpers.loader import _load_registered_model
from rift.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from rift.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from rift.utils.vae_latent_cache import tensor_digest

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def check(payload):
    c,m=payload['context'],payload['mask']
    if c.shape!=(128,4096) or c.dtype!=torch.bfloat16 or m.shape!=(128,) or m.dtype!=torch.bool or not torch.isfinite(c).all() or not m.any():
        raise RuntimeError('Invalid text embedding payload')

def filename(prompt):return hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt'

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','smoke','worker','verify'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(4);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.batch_size!=16:raise ValueError('Use the native RIFT batch size of 16')
    if not 0<=args.rank<args.world_size:p.error('Invalid rank')
    if args.mode=='init':
        if (args.run_dir/'text_settings.json').exists():raise RuntimeError('Text run already initialized')
        prompts=_read_unique_prompts([str(args.root)])
        if len(prompts)!=921032:raise RuntimeError('Incomplete official prompt set')
        path=args.run_dir/'text_prompts.json';save_json(path,prompts)
        save_json(args.run_dir/'text_settings.json',dict(prompts_sha256=sha(path),
            tasks_sha256=sha(args.root/'meta/tasks.jsonl'),source_root=str(args.root.resolve()),
            weights_sha256=sha(args.weights),tokenizer=str(args.tokenizer.resolve()),
            model='Wan-AI/Wan2.2-TI2V-5B',context_len=128,world_size=args.world_size,prompt_count=len(prompts)))
        print('TEXT_INITIALIZED',len(prompts),flush=True);return
    settings=json.loads((args.run_dir/'text_settings.json').read_text())
    if args.world_size!=settings['world_size'] or sha(args.run_dir/'text_prompts.json')!=settings['prompts_sha256'] or sha(args.root/'meta/tasks.jsonl')!=settings['tasks_sha256']:
        raise RuntimeError('Text run source settings changed')
    prompts=json.loads((args.run_dir/'text_prompts.json').read_text())
    if args.mode=='verify':
        expected=set(filename(t) for t in prompts);seen=set();total=0
        for rank in range(args.world_size):
            state=json.loads((args.run_dir/f'text_rank{rank}.json').read_text())
            if state['state']!='complete' or state['completed']!=state['total']:raise RuntimeError('Text worker incomplete')
            manifest=args.run_dir/f'text_rank{rank}.jsonl'
            if sha(manifest)!=state['manifest_sha256']:raise RuntimeError('Text worker manifest changed')
            with manifest.open() as f:
                for line in f:
                    row=json.loads(line);name=row['filename'];path=args.output/name
                    if name not in expected or name in seen or sha(path)!=row['file_sha256']:raise RuntimeError(f'Invalid text cache file: {path}')
                    payload=torch.load(path,map_location='cpu',weights_only=True);check(payload)
                    if tensor_digest(payload['context'])!=row['context_sha256'] or tensor_digest(payload['mask'])!=row['mask_sha256']:
                        raise RuntimeError('Text payload checksum mismatch')
                    seen.add(name);total+=1
                    if total%10000==0:print('TEXT_AUDIT',total,len(expected),flush=True)
        if seen!=expected:raise RuntimeError('Missing text prompt coverage')
        save_json(args.run_dir/'text_validation.json',dict(status='passed',prompts=total,all_files_sha256_verified=True,
            all_contexts_finite=True,all_masks_valid=True,settings=settings))
        print('TEXT_VALIDATED',flush=True);return
    if sha(args.weights)!=settings['weights_sha256']:raise RuntimeError('Text weights changed')
    if str(args.tokenizer.resolve())!=settings['tokenizer']:raise RuntimeError('Tokenizer changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(0);device='cuda:0'
    model=_load_registered_model(str(args.weights),'wan_video_text_encoder',torch.bfloat16,device).eval().requires_grad_(False)
    tokenizer=HuggingfaceTokenizer(name=str(args.tokenizer),seq_len=128,clean='whitespace')
    selected=prompts[args.rank::args.world_size]
    if args.mode=='smoke':selected=[prompts[i] for i in (0,1,100,1000,50000,500000,921031)]
    args.output.mkdir(parents=True,exist_ok=True)
    state=dict(state='running',rank=args.rank,total=len(selected),completed=0,pid=os.getpid(),host=os.uname().nodename)
    status=args.run_dir/(f'text_rank{args.rank}.json' if args.mode=='worker' else 'text_smoke_state.json')
    reader=RobotVideoDataset.__new__(RobotVideoDataset)
    reader.text_embedding_cache_dir=str(args.output);reader.context_len=128;reader.text_memory_cache_size=0
    reader._text_memory_cache=OrderedDict();reader._text_memory_cache_pid=os.getpid()
    manifest=args.run_dir/(f'text_rank{args.rank}.jsonl.tmp' if args.mode=='worker' else 'text_smoke.jsonl')
    before=time.monotonic()
    try:
        with manifest.open('w') as f:
            for start in range(0,len(selected),16):
                batch=selected[start:start+16];payloads={};missing=[]
                for prompt in batch:
                    path=args.output/filename(prompt)
                    if path.exists():
                        payload=torch.load(path,map_location='cpu',weights_only=True);check(payload);payloads[prompt]=payload
                    else:missing.append(prompt)
                if missing:
                    ids,mask=tokenizer(missing,return_mask=True,add_special_tokens=True)
                    mask=mask.to(device=device,dtype=torch.bool);context=model(ids.to(device),mask)
                    for i,prompt in enumerate(missing):
                        payload=dict(context=context[i].to('cpu',dtype=torch.bfloat16).contiguous(),mask=mask[i].to('cpu').contiguous())
                        check(payload);_atomic_torch_save(payload,args.output/filename(prompt));payloads[prompt]=payload
                _,all_masks=tokenizer(batch,return_mask=True,add_special_tokens=True)
                for i,prompt in enumerate(batch):
                    payload=payloads[prompt];path=args.output/filename(prompt)
                    if not torch.equal(payload['mask'],all_masks[i].bool()):raise RuntimeError('Cached tokenizer mask differs')
                    row=dict(filename=filename(prompt),context_sha256=tensor_digest(payload['context']),mask_sha256=tensor_digest(payload['mask']),file_sha256=sha(path))
                    f.write(json.dumps(row,separators=(',',':'))+'\n')
                    if start==0:
                        context_read,mask_read=reader._get_cached_text_context(prompt)
                        if not torch.equal(context_read,payload['context']) or not torch.equal(mask_read,payload['mask']):raise RuntimeError('Native RIFT text reader parity failed')
                state['completed']+=len(batch);state['elapsed_seconds']=time.monotonic()-before
                if start%160==0:f.flush();save_json(status,state);print('TEXT_PROGRESS',json.dumps(state),flush=True)
            f.flush();os.fsync(f.fileno())
        if args.mode=='worker':
            final=manifest.with_suffix('');os.replace(manifest,final)
            state['manifest_sha256']=sha(final)
        state['state']='complete';save_json(status,state)
        if args.mode=='smoke':save_json(args.run_dir/'text_smoke.json',dict(status='passed',prompts=len(selected),native_reader_bitwise_equal=True,
            context_shape=[128,4096],context_dtype='torch.bfloat16',mask_dtype='torch.bool',elapsed_seconds=state['elapsed_seconds']))
        print('TEXT_COMPLETE',args.mode,args.rank,flush=True)
    except BaseException:
        state.update(state='failed',error=traceback.format_exc());save_json(status,state);raise
if __name__=='__main__':main()
