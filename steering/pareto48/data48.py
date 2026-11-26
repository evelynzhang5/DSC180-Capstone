"""Pareto pilot, Stage 1 data: CamEdit50K focal-plane shard 001 -> same-scene sharp / background-blurred pairs.

Scene = input image id; split by sha1(id) mod 20 (0-13 train, 14-16 val, 17-19 test). A pair is eligible if its
focal value >= 0.7 (near focus: far background defocused; convention established by inspection, README) and, on the
512 center crop, the far region (farthest 40% by Depth Anything V2 Small) shows 1-12 px of added disk blur (round 40
estimator, >= 2 tiles with R^2 >= 0.9) while the near region (nearest 30%) keeps DoG band energy >= 0.75. One pair
per scene (highest focal), at most 600 scenes.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import common48 as C
import metrics48 as Mx

REPO, SHARD = "wangzike/CamEdit50K", "CamEdit_Focal_001"
MAX_SCENES, FOCAL_MIN = 600, 0.7
R_LO, R_HI, NEAR_MIN = 1.0, 12.0, 0.75


def split_of(sid):
    h = int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 20
    return "train" if h < 14 else ("val" if h < 17 else "test")


def crop512(img):
    w, h = img.size
    s = 512 / min(w, h)
    img = img.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)
    w, h = img.size
    return img.crop(((w - 512) // 2, (h - 512) // 2, (w - 512) // 2 + 512, (h - 512) // 2 + 512))


def far_tiles(sharp, far):
    gy, gx = np.gradient(Mx.gray(sharp))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    cands = [(float(grad[y:y + 64, x:x + 64].mean()), y, x) for y in range(0, 449, 32) for x in range(0, 449, 32)
             if far[y:y + 64, x:x + 64].all()]
    tiles = []
    for tex, y, x in sorted(cands, reverse=True):
        if tex < Mx.TEX:
            break
        if all(abs(y - t["y"]) >= 64 or abs(x - t["x"]) >= 64 for t in tiles):
            tiles.append(dict(y=y, x=x))
        if len(tiles) >= 8:
            break
    return tiles


def _fit(args):
    sid, sp, bp, dp = args
    s, b = Mx.arr(Image.open(sp)), Mx.arr(Image.open(bp))
    disp = np.load(dp).astype(np.float64)
    far, near = disp <= np.quantile(disp, 0.4), disp >= np.quantile(disp, 0.7)
    tiles = far_tiles(s, far)
    r, ng = float("nan"), 0
    if len(tiles) >= 2:
        m = Mx.blur40.TileStack(s, tiles).measure(b)
        r, ng = m["r"], m["n_good"]
    near_ratio = Mx.dog_ratio(s, b, near)
    ok = bool(np.isfinite(r) and R_LO <= r <= R_HI and near_ratio >= NEAR_MIN)
    return dict(id=sid, r=r, n_tiles=len(tiles), n_good=ng, near_ratio=near_ratio, eligible=ok)


def build(out, smoke=False):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    d = out / "camedit"
    (d / "pairs").mkdir(parents=True, exist_ok=True)
    pq = hf_hub_download(REPO, f"{SHARD}.parquet", repo_type="dataset", local_dir=d)
    df = pd.read_parquet(pq)
    df["focal"] = df["parameter"].astype(float)
    df["sid"] = df["original_image_path"].map(lambda p: Path(p).stem)
    cand = df[df.focal >= FOCAL_MIN].sort_values("focal", ascending=False).drop_duplicates("sid")
    cand = cand.assign(h=cand.sid.map(lambda s: hashlib.sha1(s.encode()).hexdigest())).sort_values("h")
    C.log(f"CamEdit {SHARD}: {len(df)} pairs, {df.sid.nunique()} scenes; candidates with focal >= {FOCAL_MIN}: "
          f"{len(cand)} scenes")
    if smoke:
        cand = cand.head(40)
    want = {r.original_image_path: r for r in cand.itertuples()}
    want_gt = {r.edited_image_path: r for r in cand.itertuples()}
    tar = d / f"{SHARD}.tar"
    if not tar.exists() and not (d / "extracted.ok").exists():
        C.log("downloading shard tar (~4 GB)")
        hf_hub_download(REPO, f"{SHARD}.tar", repo_type="dataset", local_dir=d)
    if not (d / "extracted.ok").exists():
        n = 0
        with tarfile.open(tar, "r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key = "input" if m.name in want else ("gt" if m.name in want_gt else None)
                if key is None:
                    continue
                r = want[m.name] if key == "input" else want_gt[m.name]
                img = crop512(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
                img.save(d / "pairs" / f"{r.sid}_{'s' if key == 'input' else 'b'}.png")
                n += 1
        (d / "extracted.ok").write_text(str(n))
        C.log(f"extracted {n} images")
        if not smoke:
            tar.unlink(missing_ok=True)
            C.log("deleted shard tar")
    rows = [r for r in cand.itertuples() if (d / "pairs" / f"{r.sid}_s.png").exists()
            and (d / "pairs" / f"{r.sid}_b.png").exists()]
    # depth on GPU, blur fits on CPU workers
    M = Mx.Models("cuda")
    jobs = []
    for r in rows:
        dp = d / "pairs" / f"{r.sid}_disp.npy"
        if not dp.exists():
            np.save(dp, M.disparity(Mx.arr(Image.open(d / "pairs" / f"{r.sid}_s.png"))).astype(np.float16))
        jobs.append((r.sid, str(d / "pairs" / f"{r.sid}_s.png"), str(d / "pairs" / f"{r.sid}_b.png"), str(dp)))
    del M
    with Pool(8) as pool:
        fits = pool.map(_fit, jobs, chunksize=8)
    by = {f["id"]: f for f in fits}
    pairs = []
    for r in rows:
        f = by[r.sid]
        if f["eligible"] and len(pairs) < MAX_SCENES:
            pairs.append(dict(id=r.sid, focal=float(r.focal), split=split_of(r.sid), r=f["r"],
                              near_ratio=f["near_ratio"], n_good=f["n_good"]))
    summ = dict(shard=SHARD, n_rows=len(df), n_scenes=int(df.sid.nunique()), n_candidates=len(cand),
                n_extracted=len(rows), n_eligible=sum(f["eligible"] for f in fits), n_used=len(pairs),
                by_split={s: sum(p["split"] == s for p in pairs) for s in ("train", "val", "test")},
                r_median=float(np.median([p["r"] for p in pairs])) if pairs else float("nan"),
                reject=dict(unmeasurable=sum(not np.isfinite(f["r"]) for f in fits),
                            r_out_of_range=sum(np.isfinite(f["r"]) and not R_LO <= f["r"] <= R_HI for f in fits),
                            near_blurred=sum(f["near_ratio"] < NEAR_MIN for f in fits)))
    C.jdump(dict(summary=summ, pairs=pairs, all_fits=fits), out / "camedit" / "pairs.json")
    C.log(f"CamEdit pairs: {summ}")
    return summ
