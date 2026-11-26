"""Pareto pilot, Stage 2 NA-RFM: activations of same-scene sharp / background-blurred CamEdit pairs at three layers,
RFM directions (upstream DualRFM, scene-split train/val/test), latent-space PCA Wiener filters for noise alignment.

Collection: VAE latent mean x 0.18215 of each 512 crop; the sharp and blurred image of a pair share one noise draw
(seeded by scene); timestep with sigma closest to 0.344 (the paper's); empty prompt; one UNet pass hooks all three
layers. The learned direction is applied at every steered sampling step with the adaptive norm scaling w |h|.
"""
from __future__ import annotations

import hashlib
import sys
import time

import numpy as np
import torch
from PIL import Image

import common48 as C


def scene_seed(sid):
    return int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)


@torch.no_grad()
def collect(sd, out, pairs):
    d = out / "camedit"
    t_c = int((sd.sig_all - C.SIG_COLLECT).abs().argmin())
    C.log(f"collect at t = {t_c}, sigma = {float(sd.sig_all[t_c]):.4f}")
    hooks = {k: sd.hook(k) for k in C.LAYERS}
    for h in hooks.values():
        h.capture = True
    empty = sd.embed([""])
    feats = {k: [] for k in C.LAYERS}
    lat, y, sids, splits = [], [], [], []
    for i, p in enumerate(pairs):
        ims = [np.asarray(Image.open(d / "pairs" / f"{p['id']}_{s}.png").convert("RGB")) for s in ("s", "b")]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(sd.dev) / 127.5 - 1
        z0 = sd.vae32.encode(x).latent_dist.mean * sd.sf
        g = torch.Generator().manual_seed(scene_seed(p["id"]))
        eps = torch.randn((1, 4, 64, 64), generator=g).to(sd.dev).expand(2, -1, -1, -1)
        tt = torch.full((2,), t_c, device=sd.dev, dtype=torch.long)
        zt = sd.sched.add_noise(z0, eps, tt)
        sd.unet(zt.to(sd.dt), tt, encoder_hidden_states=empty.expand(2, -1, -1))
        for k, h in hooks.items():
            feats[k].append(h.last.float().flatten(1).half().cpu())
        lat.append(z0.float().flatten(1).cpu())
        y += [0, 1]
        sids += [p["id"], p["id"]]
        splits += [p["split"], p["split"]]
        if i % 100 == 0:
            C.log(f"  collected {i + 1}/{len(pairs)}")
    for h in hooks.values():
        h.capture, h.last = False, None
    res = dict(t=t_c, sigma=float(sd.sig_all[t_c]), y=torch.tensor(y), sids=sids, splits=splits,
               lat=torch.cat(lat), **{f"H_{k}": torch.cat(v) for k, v in feats.items()})
    torch.save(res, out / "narfm" / "acts.pt")
    return res


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def paired_acc(score, y, sids):
    """Fraction of pairs whose blurred image scores above its own sharp image (same scene, same noise)."""
    s = {}
    for v, lab, sid in zip(score, y, sids):
        s.setdefault(sid, [None, None])[int(lab)] = v
    ok = [b > a for a, b in s.values() if a is not None and b is not None]
    return float(np.mean(ok)) if ok else float("nan")


def train(acts, out):
    sys.path.insert(0, str(C.NARFM))
    from na_rfm.dual_rfm import DualRFM
    from na_rfm.rfm_training import get_direction
    y = acts["y"].numpy()
    sp = np.array(acts["splits"])
    sids = np.array(acts["sids"])
    idx = {s: np.flatnonzero(sp == s) for s in ("train", "val", "test")}
    report = {}
    dirs = {}
    for k in C.LAYERS:
        H = acts[f"H_{k}"]
        D = H.shape[1]
        bw = 5000.0 * (D / 327680) ** 0.5
        t0 = time.time()
        res = None
        for dev in ("cuda", "cpu"):
            try:
                rfm = DualRFM(bandwidth=bw, reg=1e-3, center_grads=False, center_data=True, normalize=True,
                              device=dev, verbose=False)
                res = rfm.fit(H[idx["train"]].float(), torch.from_numpy(y[idx["train"]]).float(),
                              H[idx["val"]].float(), torch.from_numpy(y[idx["val"]]).float(),
                              n_iters=5, top_k=5, early_stop=True, early_stop_threshold=1.1)
                break
            except torch.cuda.OutOfMemoryError:
                C.log(f"RFM {k}: GPU OOM, retrying on CPU (predefined retry)")
                torch.cuda.empty_cache()
        d = get_direction(res, topk=1).float().cpu()
        mean, std = torch.from_numpy(res.mean).float(), torch.from_numpy(res.std).float()
        r = dict(D=D, bandwidth=bw, best_iter=int(res.best_iter), upstream_val_auc=float(res.best_auc),
                 seconds=time.time() - t0)
        for s in ("val", "test"):
            Xs = H[idx[s]].float()
            raw = (Xs @ d).numpy()                               # what the hook adds along (raw activations)
            nrm = (((Xs - mean) / std) @ d).numpy()              # upstream's normalised projection
            r[f"{s}_auc_raw"] = auc(raw, y[idx[s]])
            r[f"{s}_auc_norm"] = auc(nrm, y[idx[s]])
            r[f"{s}_paired_acc_raw"] = paired_acc(raw, y[idx[s]], sids[idx[s]])
            rnd = []
            for j in range(10):
                v = C.seeded_random_direction(D, 1000 + j)
                rnd.append(auc((Xs @ v).numpy(), y[idx[s]]))
            r[f"{s}_random_auc"] = [float(np.min(rnd)), float(np.median(rnd)), float(np.max(rnd))]
        md = H[idx["train"]][y[idx["train"]] == 1].float().mean(0) - H[idx["train"]][y[idx["train"]] == 0].float().mean(0)
        r["cos_with_meandiff"] = float(torch.nn.functional.cosine_similarity(d, md, dim=0))
        r["eligible"] = bool(r["val_auc_raw"] >= 0.65 or r["val_auc_norm"] >= 0.65)
        report[k] = r
        dirs[k] = d
        C.log(f"RFM {k}: D {D}, val AUC raw {r['val_auc_raw']:.3f} / norm {r['val_auc_norm']:.3f}, test AUC raw "
              f"{r['test_auc_raw']:.3f}, paired acc test {r['test_paired_acc_raw']:.2f}, random val AUC "
              f"{r['val_random_auc']}, cos(meandiff) {r['cos_with_meandiff']:+.3f} [{r['seconds']:.0f} s]")
        del H
        torch.cuda.empty_cache()
    torch.save(dirs, out / "narfm" / "directions.pt")
    return report


def build_na(acts, out, k_max=512):
    """Latent-space PCA filters for upstream GaussianDenoiser: class = blurred train latents, full = all train."""
    y = acts["y"].numpy()
    tr = np.flatnonzero(np.array(acts["splits"]) == "train")
    L = acts["lat"].float()
    filt = {}
    for name, rows in (("class", tr[y[tr] == 1]), ("full", tr)):
        X = L[rows]
        mu = X.mean(0)
        U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
        k = min(len(rows) - 1, k_max)
        ev = (S[:k] ** 2) / (len(rows) - 1)
        filt[name] = dict(mean_patch=mu, eigenvectors=Vh[:k].T.contiguous(), eigenvalues=ev, img_resolution=64,
                          img_channels=4)
    torch.save(filt, out / "narfm" / "na_filters.pt")
    return dict(k_class=int(filt["class"]["eigenvalues"].numel()), k_full=int(filt["full"]["eigenvalues"].numel()),
                ev_class_top=float(filt["class"]["eigenvalues"][0]), ev_full_top=float(filt["full"]["eigenvalues"][0]))


def load_na(out, device="cuda"):
    sys.path.insert(0, str(C.NARFM))
    from pca.gaussian_denoiser import GaussianDenoiser
    f = torch.load(out / "narfm" / "na_filters.pt")
    mk = lambda d: GaussianDenoiser({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()},
                                    num_components=None, denoising_mode="wiener").to(device)
    return mk(f["class"]), mk(f["full"])
