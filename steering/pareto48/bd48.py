"""Pareto pilot, Stage 5: BokehDiff with the frozen round 45 controller on the unsteered SD 1.5 references.

Inputs: <data>/input/<case>.png (512 reference upsampled to 768, Lanczos), prepared by BokehDiff's own prepare_data.py
(depth/, mask/) and round 40's regions40.py (regions40.json), all unchanged. Per case: gap = fit42.gap (inputs only);
K = fit45.plan_K(rule45.json, 1.5 * T, gap) (target converted from 512 to the controller's 768-px scale), clipped to
[4, 64] as frozen. Cases with < 2 behind tiles have no gap: reported controller-ineligible, not rendered (no
fallback). Render: round 45's driver path (proc_size, prepared_disp, fp32 PISA with an in-call float32 assertion,
seed 0, fp16 pipeline).
usage (BokehDiff repo as cwd): python bd48.py --test_data_dir <one input png> --output_dir <unused> ...
env: B48_DATA, B48_OUT, B48_T (target px at 512)
"""
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

DATA, OUT = os.environ["B48_DATA"], os.environ["B48_OUT"]
T512 = float(os.environ.get("B48_T", "3"))
os.environ["R40_DATA"] = DATA
HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = os.path.join(HERE, "..", "..", "bokeh", "controller")
sys.path.insert(0, CTRL)
sys.path.insert(0, os.path.join(HERE, "..", "..", "bokeh", "common"))
sys.path.insert(0, os.getcwd())
import fit42  # noqa: E402
import fit45  # noqa: E402
import inference_hf as I  # noqa: E402
import PISA_attn_processor as P  # noqa: E402

_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
CALLS = Counter()


def _f(t):
    return t.float() if torch.is_tensor(t) else t


def attn_fp32(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):                     # = bd45.attn_fp32
        out = _orig_attn(_f(query), _f(key), _f(value), _f(weight_matrix), _f(disp_coc), attn_mask=attn_mask,
                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                         occ_map=_f(occ_map))
    assert out.dtype == torch.float32, out.dtype
    CALLS[(str(query.dtype), str(out.dtype))] += 1
    return out.to(value.dtype)


P.customized_scaled_dot_product_attention = attn_fp32


def proc_size(w, h):                                                # = bd45.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared_disp(i, W2, H2):                                       # = bd45.prepared_disp
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    f = float(np.median(d[m > 0.5]))
    d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
    return d.astype(np.float32), f


def plan():
    regions = json.load(open(f"{DATA}/regions40.json"))
    rule = json.load(open(os.path.join(CTRL, "results", "rule45.json")))
    t768 = 1.5 * T512
    out = {}
    for i in sorted(regions):
        rg = regions[i]
        if len(rg["tiles"]) < 2:
            out[i] = dict(eligible=False, n_behind_tiles=len(rg["tiles"]))
            continue
        g = fit42.gap(i, regions)
        K, info = fit45.plan_K(rule, t768, g)
        out[i] = dict(eligible=True, gap=g, K=K, target_768=t768, n_behind_tiles=len(rg["tiles"]), **info)
    return out


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    pl = plan()
    rows = []
    os.makedirs(OUT, exist_ok=True)
    for i, p in pl.items():
        if not p["eligible"]:
            rows.append(dict(id=i, rendered=False, **p))
            continue
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d, f = prepared_disp(i, W2, H2)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([d, np.abs(d - f) * p["K"] / 10])[None]).to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{i}.png")
        rows.append(dict(id=i, rendered=True, render_s=float(dur), total_s=float(time.time() - t0), **p))
        print(i, f"gap {p['gap']:.3f} K {p['K']:.2f} ({p['flag']}{', clipped' if p['clipped'] else ''})",
              f"{dur:.2f}s", flush=True)
    json.dump(dict(rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}, target_512=T512),
              open(f"{OUT}/bd48.json", "w"), indent=1)
    print("bd48 done", sum(r["rendered"] for r in rows), "rendered; PISA calls:", dict(CALLS), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
