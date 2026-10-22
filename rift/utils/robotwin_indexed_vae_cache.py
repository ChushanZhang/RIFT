"""Read fully audited RoboTwin episode safetensors without decoding images."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .vae_latent_cache import identity_hash
from .robotwin_episode_cache import EpisodeHeaderCache, load_window, LATENT_SHAPE

CONTRACT='robotwin_3cam384_v1'
CAMERAS=('cam_high','cam_left_wrist','cam_right_wrist')
SOURCE_CAMERAS=tuple('observation.images.'+c for c in CAMERAS)
PREPROCESSING=dict(contract=CONTRACT,num_frames=33,video_offsets=list(range(0,33,4)),
    per_camera_resize=[240,320],canvas=[384,320],normalization=dict(mean=0.5,std=0.5),
    window_policy='each frame, repeat final frame at episode boundary')

def _json(p):return json.loads(Path(p).read_text())
def _require(ok,message):
    if not ok:raise ValueError(message)
def _sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class RobotwinIndexedVaeLatentReader:
    def __init__(self,index_dir,cache_dir,dataset_dirs,num_frames,video_size,
                 video_sample_indices,concat_multi_camera,processor):
        from torchvision.transforms import Resize,InterpolationMode
        from rift.datasets.lerobot.transforms.image import ToTensor
        _require(num_frames==33 and list(video_sample_indices)==PREPROCESSING['video_offsets']
            and list(video_size)==[384,320] and concat_multi_camera=='robotwin',
            'RoboTwin cache requires 33 observations, stride 4, robotwin composition and 384x320 canvas')
        _require([m['key'] for m in processor.shape_meta['images']]==list(CAMERAS)
            and all(list(m['shape'])==[3,240,320] for m in processor.shape_meta['images'])
            and processor.num_output_cameras==3 and processor.num_obs_steps==33,
            'RoboTwin cache camera order or dimensions differ')
        for transforms in (processor.train_transforms,processor.val_transforms):
            if isinstance(transforms,dict):_require(set(transforms)==set(CAMERAS),'Camera transforms differ')
            for pipeline in transforms.values() if isinstance(transforms,dict) else [transforms]:
                _require(pipeline is not None and len(pipeline)==2 and type(pipeline[0]) is ToTensor
                    and type(pipeline[1]) is Resize,'RoboTwin cache requires ToTensor followed by Resize')
                resize=pipeline[1]
                _require(list(resize.size)==[240,320] and resize.interpolation==InterpolationMode.BILINEAR
                    and resize.antialias is True and resize.max_size is None,'RoboTwin resize settings differ')
        run=Path(index_dir);settings=_json(run/'settings.json');report=_json(run/'validation.json')
        _require(report.get('status')=='passed' and all(report.get(k) is True for k in
            ('all_starts_covered_once','all_indices_sha256_verified','all_packed_payloads_verified','all_source_fingerprints_verified')),
            'RoboTwin cache has not passed complete validation')
        for k,v in PREPROCESSING.items():
            _require(settings.get(k)==v and report['preprocessing'].get(k)==v,f'RoboTwin preprocessing changed: {k}')
        _require(report['latent_shape']==list(LATENT_SHAPE) and report['latent_dtype']=='torch.bfloat16','Latent shape/dtype differs')
        self.namespace=report['namespace'];_require(re.fullmatch('[0-9a-f]{64}',self.namespace),'Invalid namespace')
        self.directory=Path(cache_dir)/self.namespace
        identity=_json(self.directory/'identity.json')
        _require(identity_hash(identity)==self.namespace and identity==report['identity'],'VAE identity differs')
        for name,field in (('episodes.json','episodes_sha256'),('input_fingerprints.json','fingerprints_sha256'),
                           ('keys.bin','keys_sha256'),('packed_manifest.json','manifest_sha256')):
            _require(_sha(run/name)==report[field],f'Corrupt cache inventory: {name}')
        inventory=_json(run/'episodes.json');manifest=_json(run/'packed_manifest.json')
        self._episodes={};offset=0
        _require(len(inventory)==len(manifest)==report['episode_count']==27500,'Incomplete episode inventory')
        _require(report['window_count']==6075103,'Incomplete frame inventory')
        self.dataset_names=[Path(p).name for p in dataset_dirs]
        _require(self.dataset_names==['robotwin2.0'],'RoboTwin cache expects the complete original dataset source')
        source=Path(settings['source_root']).resolve(strict=True)
        _require(Path(dataset_dirs[0]).resolve(strict=True)==source,'Dataset source differs from cached RoboTwin source')
        meta_hash={n:_sha(source/'meta'/n) for n in ('info.json','episodes.jsonl','tasks.jsonl')}
        info=_json(source/'meta/info.json')
        _require(info['fps']==50 and info['total_episodes']==27500 and info['total_frames']==6075103,
            'RoboTwin source metadata differs')
        _require(all(info['features'].get(c,{}).get('dtype')=='video' for c in SOURCE_CAMERAS),'Missing RoboTwin cameras')
        fingerprints=_json(run/'input_fingerprints.json')
        for e,m in zip(inventory,manifest):
            _require(e['meta_sha256']==meta_hash and e['fps']==50 and e['camera_keys']==list(SOURCE_CAMERAS),
                'RoboTwin source metadata or cameras changed')
            ep=e['episode_index']
            _require(m['dataset']==e['dataset']=='robotwin2.0' and m['episode_index']==ep
                and m['windows']==e['length'] and m['key_offset']==offset and ep not in self._episodes,
                'Invalid packed episode mapping')
            for p in [e['parquet_path'],*e['video_paths'].values()]:
                st=Path(p).stat()
                _require(fingerprints.get(p)==dict(size=st.st_size,mtime_ns=st.st_mtime_ns),f'Input changed: {p}')
            self._episodes[ep]=m;offset+=e['length']*32
        _require(offset==report['window_count']*32==(run/'keys.bin').stat().st_size,'Incomplete packed key index')
        self._keys_path=run/'keys.bin';self._keys=None;self._headers=EpisodeHeaderCache()

    def key(self,dataset_index,episode_index,start_frame):
        _require(int(dataset_index)==0,'Unknown RoboTwin dataset')
        entry=self._episodes[int(episode_index)];start=int(start_frame)
        if not 0<=start<entry['windows']:raise IndexError('Window outside episode')
        if self._keys is None:self._keys=np.memmap(self._keys_path,mode='r',dtype=np.uint8)
        offset=entry['key_offset']+start*32
        return self._keys[offset:offset+32].tobytes().hex()

    def load(self,dataset_index,episode_index,start_frame):
        entry=self._episodes[int(episode_index)]
        key=self.key(dataset_index,episode_index,start_frame)
        path=self.directory/'episodes'/'robotwin2.0'/f'episode_{int(episode_index):06d}.safetensors'
        return load_window(path,start_frame,key=key,namespace=self.namespace,dataset='robotwin2.0',
            episode_index=int(episode_index),index_sha256=entry['index_sha256'],window_count=entry['windows'],header_cache=self._headers)

    def __getstate__(self):
        state=dict(self.__dict__);state['_keys']=None;state['_headers']=EpisodeHeaderCache();return state
