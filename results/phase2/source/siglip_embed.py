"""Frozen SigLIP, immutable model revision, official transform, resumable batches."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import json
import time
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image, ImageOps
from huggingface_hub import HfApi,hf_hub_download
from benchmark import OUT,ROOT,DATA,inputs,sha,write_json,array,canonical,digest,now,environment

REPO='timm/ViT-B-16-SigLIP'

def main():
    assert (OUT/'clip/validation.json').exists(),'Complete Task 1 before SigLIP'
    df,_,clip=inputs();environment('siglip_embedding')
    cache=ROOT/'data/phase2/siglip_model';cache.mkdir(parents=True,exist_ok=True)
    info_path=cache/'model.json'
    if info_path.exists(): info=json.loads(info_path.read_text());revision=info['revision']
    else: revision=HfApi().model_info(REPO).sha
    kw={'repo_id':REPO,'revision':revision,'cache_dir':str(cache/'hub')}
    weight=hf_hub_download(filename='open_clip_model.safetensors',**kw)
    config_path=hf_hub_download(filename='open_clip_config.json',**kw)
    config=json.loads(Path(config_path).read_text())
    info={'repo':REPO,'revision':revision,'weight_path':weight,'weight_sha256':sha(weight),'config_sha256':sha(config_path),'config':config,'dimension':config['model_cfg']['embed_dim'],'libraries':{p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch','timm','pillow','numpy']},'device':'cpu','precision':'float32','threads':4,'seed':42,'input_adaptation':'EXIF transpose, RGB, white square padding shared with cached CLIP; then official checkpoint evaluation transform','inference_source_sha256':sha(__file__)}
    write_json(info_path,info)
    torch.manual_seed(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    localcfg=cache/'SiglipPhase2.json';write_json(localcfg,config['model_cfg']);open_clip.add_model_config(localcfg)
    cfg=config['preprocess_cfg']
    model,_,preprocess=open_clip.create_model_and_transforms('SiglipPhase2',pretrained=weight,device='cpu',**{'image_'+k:cfg[k] for k in ['mean','std','interpolation','resize_mode'] if k in cfg})
    model.eval().requires_grad_(False)
    info['actual_transform']=repr(preprocess)
    fingerprint=digest(canonical(info).encode())
    started=time.perf_counter();chunks=[];hits=0;new=0
    chunk_dir=ROOT/'data/phase2/siglip_vectors'/fingerprint;chunk_dir.mkdir(parents=True,exist_ok=True)
    all_inputs=clip['descriptor']['inputs']
    for start in range(0,len(df),16):
        rows=all_inputs[start:start+16];desc={'model_fingerprint':fingerprint,'inputs':rows}
        path=chunk_dir/f'{start:05d}.npy';meta=path.with_suffix('.json');x=None
        if path.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if m['descriptor']==desc and m['sha256']==sha(path):
                candidate=np.load(path,allow_pickle=False)
                if candidate.shape==(len(rows),768) and np.isfinite(candidate).all() and np.allclose(np.linalg.norm(candidate,axis=1),1,atol=1e-4):x=candidate;hits+=len(rows)
        if x is None:
            batch=[]
            for r in df.iloc[start:start+16].itertuples():
                assert sha(DATA/r.file)==r.sha256
                with Image.open(DATA/r.file) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');side=max(im.size);im=ImageOps.pad(im,(side,side),color=(255,255,255));batch.append(preprocess(im))
            with torch.inference_mode():
                x=torch.nn.functional.normalize(model.encode_image(torch.stack(batch)).float(),dim=-1).numpy()
            assert x.shape==(len(rows),768) and np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4)
            array(path,x);write_json(meta,{'descriptor':desc,'sha256':sha(path)});new+=len(rows)
        chunks.append(x)
        if start%160==0:print('SigLIP images',start+len(rows),'/',len(df),f'{time.perf_counter()-started:.1f}s',flush=True)
    X=np.concatenate(chunks);out=OUT/'siglip';array(out/'embeddings.npy',X)
    write_json(out/'embedding_manifest.json',{'utc':now(),'inputs':all_inputs,'ordered_inputs_sha256':digest(canonical(all_inputs).encode()),'model':info,'model_fingerprint':fingerprint,'shape':list(X.shape),'array_path':str(out/'embeddings.npy'),'array_sha256':sha(out/'embeddings.npy'),'new_images':new,'verified_cache_hits':hits,'embedding_seconds':time.perf_counter()-started})
    print('SigLIP complete',X.shape,'new',new,'verified hits',hits,flush=True)

if __name__=='__main__': main()
