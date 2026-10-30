"""Source E9-extract_mps.py: retained preprocessing plus GigaPath-only run derivative. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import argparse, json, time, gc, sys, os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np, h5py, torch, openslide, cv2
from PIL import Image
from . import config as C
from .encoders import REGISTRY, read_px


def build_mask(slide):
    thumb = np.array(slide.get_thumbnail(
        (slide.dimensions[0]//C.MASK_STRIDE, slide.dimensions[1]//C.MASK_STRIDE))
        .convert("RGB"))
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    m = (hsv[...,1] > C.STHRESH).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((C.CLOSE,C.CLOSE), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= C.A_T:
            keep[lab==i] = 255
    return keep

def tile_coords(slide, read):
    mask = build_mask(slide)
    W, H = slide.dimensions; ms = C.MASK_STRIDE
    coords = []
    for y in range(0, H-read+1, read):
        for x in range(0, W-read+1, read):
            win = mask[y//ms:max((y+read)//ms, y//ms+1),
                       x//ms:max((x+read)//ms, x//ms+1)]
            if win.size and (win>0).mean() >= C.TISSUE_FRAC:
                coords.append((x, y))
    return coords

def pen_reject(tile_rgb):
    hsv = cv2.cvtColor(np.asarray(tile_rgb), cv2.COLOR_RGB2HSV)
    return (hsv[...,2] < C.PEN_V_THR).mean() > C.PEN_FRAC_THR

@torch.no_grad()
def run(slide_path, enc_name, batch, dtype, tag, warmup=False):
    dev = "mps"
    spec = REGISTRY[enc_name]
    model = spec["load"]().to(dev)  # Public GigaPath-only derivative; excluded loaders are not reconstructed.
    if dtype == "fp16": model = model.half()
    slide = openslide.OpenSlide(slide_path)
    mpp = float(slide.properties.get("openslide.mpp-x"))
    read = read_px(spec["win_px"], mpp)
    coords = tile_coords(slide, read)
    amp_dtype = torch.float16 if dtype=="fp16" else torch.float32

    def one_pass(timed: bool):
        feats, t_enc, peak = [], 0.0, 0
        for i in range(0, len(coords), batch):
            tiles = []
            for (x,y) in coords[i:i+batch]:
                im = slide.read_region((x,y), 0, (read,read)).convert("RGB")
                if C.PEN_FILTER and pen_reject(im):
                    im = Image.new("RGB", (read,read), (255,255,255))
                tiles.append(spec["transform"](im))
            x = torch.stack(tiles).to(dev, dtype=amp_dtype)
            t0 = time.perf_counter()
            with torch.autocast(device_type="mps", dtype=amp_dtype,
                                enabled=(dtype=="fp16")):
                out = spec["postprocess"](model(x))
            if timed:
                torch.mps.synchronize()
                t_enc += time.perf_counter()-t0
            feats.append(out.float().cpu().numpy())
            peak = max(peak, torch.mps.current_allocated_memory())
        return np.concatenate(feats), t_enc, peak

    one_pass(timed=False)                       # 热机（永远执行，不计时）
    if warmup: return
    t0 = time.perf_counter()
    feats, t_enc, peak = one_pass(timed=True)
    t_total = time.perf_counter()-t0
    out = dict(slide=slide_path.split("/")[-1], encoder=enc_name, batch=batch,
               dtype=dtype, n_tiles=len(coords), read=read, mpp=mpp,
               win_px=spec["win_px"], net_px=spec["net_px"],
               t_encode_s=round(t_enc,2), t_total_s=round(t_total,2),
               tiles_per_s=round(len(coords)/max(t_enc,1e-9),2),
               peak_alloc_gb=round(peak/2**30,3),
               peak_driver_gb=round(torch.mps.driver_allocated_memory()/2**30,3))
    if tag:
        with h5py.File(f"features/{out['slide'][:-4]}_{enc_name}_{dtype}.h5","w") as f:
            f.create_dataset("features", data=feats.astype(np.float32))
            f.create_dataset("coords",   data=np.array(coords, dtype=np.int32))
    with open(f"runs/{tag or 'manual'}.json","w") as f: json.dump(out, f, indent=1)
    print(json.dumps(out))
    del model; gc.collect(); torch.mps.empty_cache()

# New public entry: run retains the historical loop; loader dispatch is GigaPath-only. No default execution.
def main():
    p=argparse.ArgumentParser(description='M5 E9 source. Explicit execution can overwrite historical-style outputs; use an empty working directory.')
    p.add_argument('--execute',action='store_true');p.add_argument('--slide');p.add_argument('--encoder',choices=list(REGISTRY));p.add_argument('--batch',type=int);p.add_argument('--dtype',choices=C.DTYPE_LIST);p.add_argument('--tag',default='');p.add_argument('--warmup',action='store_true')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='PLAN_ONLY',family='M5_E9',warmup='one discarded full pass',batch=a.batch,dtype=a.dtype)));return
    if not a.slide or not a.encoder or not a.batch or not a.dtype:p.error('explicit slide/encoder/batch/dtype required')
    from pathlib import Path
    if not a.tag or Path(a.tag).name!=a.tag:p.error('a simple unique --tag is required')
    target=Path('features')/(Path(a.slide).stem+'_'+a.encoder+'_'+a.dtype+'.h5')
    if target.exists() or (Path('runs')/(a.tag+'.json')).exists():raise FileExistsError('Refuse overwrite in public entry')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available():raise RuntimeError('MPS required; no CPU fallback')
    Path('features').mkdir(exist_ok=True);Path('runs').mkdir(exist_ok=True)
    torch.manual_seed(C.SEED)
    run(a.slide,a.encoder,a.batch,a.dtype,a.tag,a.warmup)
if __name__=='__main__':main()
