"""Pareto pilot measurements. Regions are computed once on the unsteered reference and frozen (regions/<case>.npz).

blur      disk-fit added radius on background tiles (round 40 estimator, bokeh/common/blur40.py) and the
          alignment-free DoG(1->3) band-energy ratio over the background region
subject   log DoG band-energy ratio in the subject interior
identity  against the BLUR-MATCHED reference: scene change = 1 - mean DINOv2 patch cosine; subject DINOv2 (CLS of
          the reference subject box); CLIPSeg mask IoU and centroid shift; depth Spearman
colour    mean Delta E76 of Gaussian(8)-low-passed Lab images
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label

import common48 as C

import blur40  # noqa: E402  (bokeh/common on sys.path via common48)
import common40  # noqa: E402
import measure18  # noqa: E402

SEG = "CIDAS/clipseg-rd64-refined"
DINO = "facebook/dinov2-base"
DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"
TILE, STRIDE, MAXT, TEX = 64, 32, 8, 4.0
# frozen per-image rules (README); thresholds may be relaxed once by validation (a), before steered outputs exist
RULES = dict(dino_subj=0.85, iou=0.75, shift=16.0, subj_sharp=-0.35, de=4.0, scene=0.15)
RESP_R, RESP_DOG = 1.0, 0.8


def arr(img):
    return np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img).astype(np.float64)


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def dog_energy(a, m):
    g = gray(a)
    d = gaussian_filter(g, 1.0) - gaussian_filter(g, 3.0)
    return float((d[m] ** 2).mean()) if m.any() else float("nan")


def dog_ratio(ref, out, m):
    return dog_energy(out, m) / max(dog_energy(ref, m), 1e-9)


# ---------------------------------------------------------------- colour
def _lab(a):
    x = common40.to_lin(a)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(ref, out):
    lp = lambda a: np.stack([gaussian_filter(a[..., c], 8.0) for c in range(3)], -1)
    return float(np.sqrt(((_lab(lp(ref)) - _lab(lp(out))) ** 2).sum(-1)).mean())


# ---------------------------------------------------------------- blur operators (linear light)
def disk_blur(a, r):
    if r <= 1e-6:
        return a.copy()
    lin = common40.to_lin(a)
    b = measure18.blur(lin, measure18.disk(r))
    b = np.clip(b, 0, 1)
    return np.where(b <= 0.0031308, 12.92 * b, 1.055 * b ** (1 / 2.4) - 0.055) * 255


def feather(subj, s=3.0):
    return np.clip(2 * gaussian_filter(subj.astype(float), s), 0, 1)[..., None]


def posthoc(ref, subj, r):
    """Reference with its background disk-blurred by r px and the subject composited back (feathered)."""
    al = feather(subj)
    return al * ref + (1 - al) * disk_blur(ref, r)


# ---------------------------------------------------------------- models
class Models:
    def __init__(self, device="cuda"):
        from transformers import (AutoImageProcessor, AutoModel, AutoModelForDepthEstimation,
                                  CLIPSegForImageSegmentation, CLIPSegProcessor)
        self.dev = device
        self.sp = CLIPSegProcessor.from_pretrained(SEG)
        self.seg = CLIPSegForImageSegmentation.from_pretrained(SEG).to(device).eval()
        self.dp = AutoImageProcessor.from_pretrained(DINO)
        self.dino = AutoModel.from_pretrained(DINO).to(device).eval()
        self.ep = AutoImageProcessor.from_pretrained(DEPTH)
        self.depth = AutoModelForDepthEstimation.from_pretrained(DEPTH).to(device).eval()

    @torch.no_grad()
    def seg_prob(self, a, noun):
        img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
        x = self.sp(text=[f"a photo of a {noun}"], images=[img], padding="max_length", return_tensors="pt")
        lg = self.seg(**x.to(self.dev)).logits
        lg = lg.reshape(-1, lg.shape[-2], lg.shape[-1])[:1, None].float()
        p = torch.nn.functional.interpolate(lg, size=a.shape[:2], mode="bilinear", align_corners=False)
        return torch.sigmoid(p)[0, 0].cpu().numpy()

    @staticmethod
    def mask_from_prob(prob, thr):
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(prob > thr)
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.1 * sizes.max()))
        return m

    @torch.no_grad()
    def disparity(self, a):
        img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
        inp = self.ep(images=img, return_tensors="pt").to(self.dev)
        pred = self.depth(**inp).predicted_depth
        pred = torch.nn.functional.interpolate(pred[:, None], size=a.shape[:2], mode="bicubic", align_corners=False)
        d = pred[0, 0].cpu().numpy()
        return (d - d.min()) / max(d.max() - d.min(), 1e-9)              # larger = nearer

    @torch.no_grad()
    def dino_tokens(self, a):
        img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
        x = self.dp(images=img, return_tensors="pt", do_center_crop=False, size={"height": 448, "width": 448},
                    do_resize=True).to(self.dev)
        h = self.dino(**x).last_hidden_state[0].float()
        h = h / h.norm(dim=-1, keepdim=True)
        return h[0].cpu().numpy(), h[1:].cpu().numpy()                    # CLS, patches (1024)


# ---------------------------------------------------------------- regions (reference only, frozen)
def regions(M, ref, noun):
    prob = M.seg_prob(ref, noun)
    thr, subj = None, None
    for t in (0.4, 0.3, 0.2):
        m = M.mask_from_prob(prob, t)
        if m.mean() >= 0.005:
            thr, subj = t, m
            break
    if subj is None:
        thr, subj = 0.2, M.mask_from_prob(prob, 0.2)
    disp = M.disparity(ref)
    dout = distance_transform_edt(~subj)
    sd = float(np.median(disp[subj])) if subj.any() else float(np.quantile(disp, 0.7))
    bg = (dout > 16) & (disp <= sd - 0.10)
    interior = binary_erosion(subj, iterations=6) if subj.sum() > 2000 else subj.copy()
    gy, gx = np.gradient(gray(ref))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    cands = []
    H, W = bg.shape
    for y in range(0, H - TILE + 1, STRIDE):
        for x in range(0, W - TILE + 1, STRIDE):
            if bg[y:y + TILE, x:x + TILE].all():
                tex = float(grad[y:y + TILE, x:x + TILE].mean())
                if tex >= TEX:
                    cands.append((tex, y, x))
    tiles = []
    for tex, y, x in sorted(cands, reverse=True):
        if all(abs(y - t["y"]) >= TILE or abs(x - t["x"]) >= TILE for t in tiles):
            tiles.append(dict(y=int(y), x=int(x), tex=tex))
        if len(tiles) >= MAXT:
            break
    return dict(subject=subj, interior=interior, background=bg, disp=disp, thr=thr, noun=noun, tiles=tiles,
                subj_area=float(subj.mean()), bg_area=float(bg.mean()))


def save_regions(path, R):
    np.savez_compressed(path, subject=R["subject"], interior=R["interior"], background=R["background"],
                        disp=R["disp"].astype(np.float32))
    C.jdump({k: R[k] for k in ("thr", "noun", "tiles", "subj_area", "bg_area")}, str(path) + ".json")


def load_regions(path):
    z = np.load(path)
    R = {k: z[k] for k in ("subject", "interior", "background", "disp")}
    R.update(C.jload(str(path) + ".json"))
    return R


# ---------------------------------------------------------------- scorer
class Scorer:
    """Per reference: cached tile stacks and reference features; score(out) -> dict of measures."""

    def __init__(self, M, ref, R):
        self.M, self.ref, self.R = M, ref, R
        self.stack = blur40.TileStack(ref, R["tiles"]) if len(R["tiles"]) >= common40.MIN_TILES else None
        self.cls, self.pat = M.dino_tokens(ref)
        s = R["subject"]
        if s.any():
            ys, xs = np.nonzero(s)
            self.box = (max(xs.min() - 8, 0), max(ys.min() - 8, 0), min(xs.max() + 9, s.shape[1]),
                        min(ys.max() + 9, s.shape[0]))
        else:
            self.box = (0, 0, s.shape[1], s.shape[0])
        self.ref_box_cls = self._box_cls(ref)
        self.c0 = np.array(np.nonzero(s)).mean(1) if s.any() else np.array([s.shape[0] / 2, s.shape[1] / 2])
        self._matched = {}

    def _box_cls(self, a):
        x0, y0, x1, y1 = self.box
        return self.M.dino_tokens(a[y0:y1, x0:x1])[0]

    def matched_ref(self, r):
        r = float(np.round(max(r, 0.0) * 4) / 4)                      # 0.25 px grid, cached
        if r not in self._matched:
            self._matched[r] = posthoc(self.ref, self.R["subject"], r)
        return r, self._matched[r]

    def radius_for_dog(self, ratio):
        """Radius whose post-hoc render matches an observed background DoG ratio (for unmeasurable outputs)."""
        grid = np.arange(0, 12.01, 0.5)
        vals = [dog_ratio(self.ref, self.matched_ref(r)[1], self.R["background"]) for r in grid]
        return float(grid[int(np.argmin(np.abs(np.array(vals) - ratio)))])

    def score(self, out, p_reading=None):
        R, ref = self.R, self.ref
        res = dict(r=float("nan"), n_tiles_ok=0, n_tiles=len(R["tiles"]))
        if self.stack is not None:
            m = self.stack.measure(out)
            res.update(r=m["r"], n_tiles_ok=m["n_good"])
        res["measurable"] = bool(np.isfinite(res["r"]))
        res["dog_bg"] = dog_ratio(ref, out, R["background"]) if R["background"].any() else float("nan")
        res["subj_sharp"] = float(np.log(max(dog_ratio(ref, out, R["interior"]), 1e-9))) if R["interior"].any() \
            else float("nan")
        r_match = res["r"] if res["measurable"] else (self.radius_for_dog(res["dog_bg"])
                                                      if np.isfinite(res["dog_bg"]) else 0.0)
        res["r_matched"], mref = self.matched_ref(r_match)
        mcls, mpat = self.M.dino_tokens(mref)
        ocls, opat = self.M.dino_tokens(out)
        res["scene"] = float(1 - (mpat * opat).sum(-1).mean())
        res["scene_vs_ref"] = float(1 - (self.pat * opat).sum(-1).mean())
        mb = self._box_cls(mref)
        res["dino_subj"] = float(mb @ self._box_cls(out))
        prob = self.M.seg_prob(out, R["noun"])
        mo = self.M.mask_from_prob(prob, R["thr"])
        s = R["subject"]
        res["iou"] = float((mo & s).sum() / max((mo | s).sum(), 1))
        res["shift"] = float(np.linalg.norm(np.array(np.nonzero(mo)).mean(1) - self.c0)) if mo.any() else 512.0
        from scipy.stats import spearmanr
        d0 = R["disp"][::4, ::4].ravel()
        d1 = self.M.disparity(out)[::4, ::4].ravel()
        res["depth_rho"] = float(spearmanr(d0, d1)[0])
        res["de"] = delta_e(ref, out)
        res["pix_rms_vs_matched"] = float(np.sqrt(((out - mref) ** 2).mean()) / 255)
        if p_reading is not None:
            res["p_reading"] = float(p_reading)
        return judge(res)


def judge(res, rules=None):
    rules = rules or RULES
    resp = (res["measurable"] and res["r"] >= RESP_R) or (not res["measurable"] and np.isfinite(res["dog_bg"])
                                                          and res["dog_bg"] <= RESP_DOG)
    pres = (res["dino_subj"] >= rules["dino_subj"] and res["iou"] >= rules["iou"] and res["shift"] <= rules["shift"]
            and (not np.isfinite(res["subj_sharp"]) or res["subj_sharp"] >= rules["subj_sharp"])
            and res["de"] <= rules["de"] and res["scene"] <= rules["scene"])
    res["response"], res["preserved"] = bool(resp), bool(pres)
    res["on_target"] = bool(res["measurable"] and abs(res["r"] - C.T_BLUR) <= C.TOL)
    res["target_err"] = float(abs(res["r"] - C.T_BLUR)) if res["measurable"] else float("nan")
    return res


# ---------------------------------------------------------------- validation transforms (Stage 1)
def transforms(ref, R):
    s = R["subject"]
    out = {f"bgblur_r{r:g}": (posthoc(ref, s, r), dict(kind="bgblur", r=r)) for r in (1, 2, 3, 4, 6)}
    al = feather(s)
    mean = ref[R["background"]].mean(0) if R["background"].any() else ref.reshape(-1, 3).mean(0)
    out["contrast0.6"] = (al * ref + (1 - al) * (mean + 0.6 * (ref - mean)), dict(kind="contrast"))
    for d in (8, 16):
        out[f"shift{d}"] = (np.roll(ref, (d, d), axis=(0, 1)), dict(kind="shift", d=d))
    out["global_r2"] = (disk_blur(ref, 2.0), dict(kind="global", r=2.0))
    return out
