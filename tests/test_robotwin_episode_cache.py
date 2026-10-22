"""Guard packed random access and failure paths used by offline RoboTwin jobs."""
import hashlib
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors import safe_open
from rift.utils.robotwin_episode_cache import pack_episode,load_window,EpisodeHeaderCache,LATENT_SHAPE
from rift.utils.vae_latent_cache import tensor_digest
from rift.utils.robotwin_indexed_vae_cache import RobotwinIndexedVaeLatentReader

@pytest.fixture
def packed(tmp_path):
    torch.set_num_threads(1)
    latents=torch.arange(3*48*3*24*20,dtype=torch.float32).reshape(3,*LATENT_SHAPE).remainder(17).to(torch.bfloat16)
    keys=[hashlib.sha256(f'window{i}'.encode()).hexdigest() for i in range(3)]
    args=dict(namespace='a'*64,dataset='robotwin2.0',episode_index=7,index_sha256='b'*64,window_count=3)
    path=tmp_path/'episode.safetensors'
    pack_episode(path,latents,**{k:v for k,v in args.items() if k!='window_count'},window_keys=keys,
        window_sha256=[tensor_digest(x) for x in latents])
    return path,latents,keys,args

def test_random_access_and_cached_headers(packed):
    path,latents,keys,args=packed;headers=EpisodeHeaderCache()
    for i in (2,0,1,2):assert torch.equal(load_window(path,i,key=keys[i],header_cache=headers,**args),latents[i])

def test_wrong_identity_key_and_padding_bounds_fail(packed):
    path,_,keys,args=packed
    with pytest.raises(RuntimeError,match='key differs'):load_window(path,0,key=keys[1],**args)
    with pytest.raises(ValueError,match='metadata differs'):load_window(path,0,key=keys[0],**{**args,'episode_index':8})
    with pytest.raises(IndexError):load_window(path,3,key=keys[0],**args)
    with pytest.raises(IndexError):load_window(path,-1,key=keys[0],**args)
    with pytest.raises(TypeError):load_window(path,0.5,key=keys[0],**args)

def test_payload_corruption_detected_after_header_cache_hit(packed):
    path,_,keys,args=packed;headers=EpisodeHeaderCache()
    load_window(path,0,key=keys[0],header_cache=headers,**args)
    with path.open('r+b') as f:
        header_length=int.from_bytes(f.read(8),'little');f.seek(8+header_length+2)
        original=f.read(1);f.seek(-1,1);f.write(bytes([original[0]^1]))
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        load_window(path,0,key=keys[0],header_cache=headers,**args)

def test_no_overwrite(packed):
    path,latents,keys,args=packed
    with pytest.raises(FileExistsError):
        pack_episode(path,latents,**{k:v for k,v in args.items() if k!='window_count'},window_keys=keys,
            window_sha256=[tensor_digest(x) for x in latents])

def test_reader_rejects_wrong_robotwin_contract_before_loading(tmp_path):
    with pytest.raises(ValueError,match='stride 4'):
        RobotwinIndexedVaeLatentReader(tmp_path,tmp_path,[],33,[224,448],list(range(0,33,4)),'horizontal',None)
