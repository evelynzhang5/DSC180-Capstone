"""Pareto pilot, Stage 2 TFG: a differentiable background-blur predictor P and the bridge to upstream TFG.

P: RGB 256x256 in [-1, 1] -> per-pixel added-blur radius map 64x64 (px at 512 scale). Trained on train-scene sharp
CamEdit crops with synthetic depth-layered disk blur rendered in linear light (8 radius levels, focus on the near half,
max radius ~ U[0, 10]), brightness/contrast jitter (label unchanged) and unblurred images; L1 loss; checkpoint chosen on
val scenes. Validated on test scenes (README gates).

TFG: upstream methods/tfg.py TFGGuidance (guide_step, schedules, MC smoothing) used verbatim. Its task registry imports
audio/molecule packages, so tasks.base, tasks.utils and utils.configs are stubbed; rescale_grad is executed from the
upstream source text (it is the identity for image tensors). The guider decodes the latent x0 with the fp32 VAE,
average-pools to 256, runs P and returns logp = -(mean over the reference's background of P - T)^2.
"""
from __future__ import annotations

import ast
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import common48 as C


# ---------------------------------------------------------------- predictor
class Res(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)

    def forward(self, x):
        return x + self.b(F.silu(self.n2(self.a(F.silu(self.n1(x))))))


class BlurNet(nn.Module):
    def __init__(self, c=96):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                  nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                  nn.Conv2d(64, c, 3, stride=2, padding=1))
        self.body = nn.Sequential(*[Res(c, d) for d in (1, 2, 4, 8, 1, 2)])
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):                                      # x: (B, 3, 256, 256) in [-1, 1]
        return F.softplus(self.head(F.silu(self.body(self.stem(x)))))[:, 0]      # (B, 64, 64) px


# ---------------------------------------------------------------- synthetic renders (GPU)
def to_lin(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


_K = {}


def disk_kernel(r, device):
    key = (round(float(r), 3), str(device))
    if key not in _K:
        import measure18
        k = torch.from_numpy(measure18.disk(float(r))).float().to(device)
        _K[key] = k
    return _K[key]


def disk_t(lin, r):
    """lin: (B, 3, H, W); uniform disk blur, reflect padding. Radii < 0.25 px = no blur (the anti-aliased disk of a
    sub-sample radius is empty, and normalising it gave NaN)."""
    if r < 0.25:
        return lin
    k = disk_kernel(r, lin.device)
    p = k.shape[0] // 2
    x = F.pad(lin, (p, p, p, p), mode="reflect")
    return F.conv2d(x, k[None, None].expand(3, 1, -1, -1), groups=3)


def render(sharp, disp, rmax, fq, n_levels=8):
    """sharp (B,3,512,512) in [0,1]; disp (B,512,512) larger = nearer. Returns (blurred sRGB, radius map px)."""
    B = sharp.shape[0]
    lin = to_lin(sharp)
    outs, rmaps = [], []
    for b in range(B):
        d = disp[b]
        f = torch.quantile(d.flatten()[::64], fq[b])
        dist = (f - d).clamp(min=0)                            # only farther than focus gets blurred
        rm = rmax[b] * dist / dist.max().clamp(min=1e-6)
        if rmax[b] <= 0:
            outs.append(sharp[b])
            rmaps.append(torch.zeros_like(d))
            continue
        levels = torch.linspace(0, float(rmax[b]), n_levels, device=sharp.device)
        stack = torch.stack([disk_t(lin[b:b + 1], float(r))[0] for r in levels])        # (L, 3, H, W)
        pos = (rm / levels[-1] * (n_levels - 1)).clamp(0, n_levels - 1)
        lo = pos.floor().long().clamp(max=n_levels - 2)
        w = (pos - lo.float())[None]
        g = lambda idx: stack.gather(0, idx[None, None].expand(1, 3, -1, -1))[0]
        out = (1 - w) * g(lo) + w * g(lo + 1)
        outs.append(to_srgb(out))
        rmaps.append(rm)
    return torch.stack(outs), torch.stack(rmaps)


def predict(P, img01):
    """img01: (B, 3, 512, 512) in [0, 1] -> (B, 64, 64) radius map."""
    return P(F.avg_pool2d(img01 * 2 - 1, 2))


class PairSet:
    def __init__(self, out, pairs, split, device):
        d = out / "camedit" / "pairs"
        ps = [p for p in pairs if p["split"] == split]
        self.ids = [p["id"] for p in ps]
        self.r = np.array([p["r"] for p in ps])
        self.sharp = torch.from_numpy(np.stack([np.asarray(Image.open(d / f"{i}_s.png").convert("RGB"))
                                                for i in self.ids])).permute(0, 3, 1, 2)          # uint8 CPU
        self.blur = torch.from_numpy(np.stack([np.asarray(Image.open(d / f"{i}_b.png").convert("RGB"))
                                               for i in self.ids])).permute(0, 3, 1, 2)
        self.disp = torch.from_numpy(np.stack([np.load(d / f"{i}_disp.npy").astype(np.float32) for i in self.ids]))
        self.dev = device

    def batch(self, idx, rng, jitter=True, rmax=None, fq=None):
        x = self.sharp[idx].float().to(self.dev) / 255
        dd = self.disp[idx].to(self.dev)
        if rng.random() < 0.5:
            x, dd = x.flip(-1), dd.flip(-1)
        B = len(idx)
        rmax = rmax if rmax is not None else np.where(rng.random(B) < 0.2, 0.0, rng.uniform(0, 10, B))
        fq = fq if fq is not None else rng.uniform(0.55, 0.95, B)
        y, rm = render(x, dd, rmax, fq)
        if jitter:
            c = torch.from_numpy(rng.uniform(0.7, 1.3, (B, 1, 1, 1))).float().to(self.dev)
            bb = torch.from_numpy(rng.uniform(-0.1, 0.1, (B, 1, 1, 1))).float().to(self.dev)
            m = y.mean((1, 2, 3), keepdim=True)
            y = ((y - m) * c + m + bb).clamp(0, 1)
        return y, F.avg_pool2d(rm[:, None], 8)[:, 0]


def train(out, pairs, iters=2500, bs=16, smoke=False):
    dev = "cuda"
    tr, va = PairSet(out, pairs, "train", dev), PairSet(out, pairs, "val", dev)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    P = BlurNet().to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=2e-3, weight_decay=1e-4)
    iters = 60 if smoke else iters
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=iters, pct_start=0.1)
    vrng = np.random.default_rng(123)
    vb = [va.batch(np.arange(i, min(i + 8, len(va.ids))), vrng, jitter=False) for i in range(0, len(va.ids), 8)]
    best, hist, t0 = float("inf"), [], time.time()
    for it in range(1, iters + 1):
        idx = rng.choice(len(tr.ids), bs, replace=len(tr.ids) < bs)
        x, y = tr.batch(idx, rng)
        loss = (predict(P, x) - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % (20 if smoke else 250) == 0 or it == iters:
            P.eval()
            with torch.no_grad():
                vl = float(np.mean([(predict(P, x) - y).abs().mean().item() for x, y in vb]))
            P.train()
            hist.append(dict(it=it, train_l1=float(loss), val_l1=vl))
            C.log(f"  P it {it}: train L1 {float(loss):.3f}, val L1 {vl:.3f} [{time.time() - t0:.0f} s]")
            if vl < best:
                best = vl
                torch.save(P.state_dict(), out / "tfg" / "P.pt")
    return dict(best_val_l1=best, hist=hist, n_params=sum(p.numel() for p in P.parameters()),
                n_train=len(tr.ids), n_val=len(va.ids), seconds=time.time() - t0)


def load_P(out, device="cuda"):
    P = BlurNet().to(device)
    P.load_state_dict(torch.load(out / "tfg" / "P.pt", map_location=device))
    P.eval().requires_grad_(False)
    return P


@torch.no_grad()
def validate(out, pairs):
    """README predictor gates on test scenes; real-pair transfer on val+test."""
    from scipy.stats import spearmanr
    dev = "cuda"
    P = load_P(out, dev)
    te = PairSet(out, pairs, "test", dev)
    res = dict(n_test=len(te.ids))
    rows = []
    for i in range(len(te.ids)):
        x = te.sharp[i:i + 1].float().to(dev) / 255
        lin = to_lin(x)
        base = float(predict(P, x).mean())
        row = dict(id=te.ids[i], base=base)
        for r in (0, 1, 2, 4, 8):
            row[f"r{r}"] = float(predict(P, to_srgb(disk_t(lin, r))).mean())
        m = x.mean()
        row["contrast"] = float(predict(P, ((x - m) * 0.5 + m).clamp(0, 1)).mean())
        med = torch.from_numpy(np.stack([__import__("scipy.ndimage", fromlist=["median_filter"]).median_filter(
            x[0, c].cpu().numpy(), size=5) for c in range(3)]))[None].to(dev)
        row["median5"] = float(predict(P, med).mean())
        rows.append(row)
    tr_r = [(r, row[f"r{r}"]) for row in rows for r in (0, 1, 2, 4, 8)]
    res["blur_spearman"] = float(spearmanr([a for a, _ in tr_r], [b for _, b in tr_r])[0])
    res["blur_mae_le4"] = float(np.mean([abs(row[f"r{r}"] - r) for row in rows for r in (0, 1, 2, 4)]))
    resp4 = float(np.mean([row["r4"] - row["r0"] for row in rows]))
    resp2 = float(np.mean([row["r2"] - row["r0"] for row in rows]))
    res["resp_r4"], res["resp_r2"] = resp4, resp2
    res["contrast_delta"] = float(np.mean([row["contrast"] - row["r0"] for row in rows]))
    res["median_delta"] = float(np.mean([row["median5"] - row["r0"] for row in rows]))
    res["gate_blur"] = bool(res["blur_spearman"] >= 0.9 and res["blur_mae_le4"] <= 1.5)
    res["gate_contrast"] = bool(abs(res["contrast_delta"]) <= max(0.5, 0.25 * resp4))
    res["texture_ok"] = bool(res["median_delta"] <= resp2)
    # scene category (CLIP zero-shot on unblurred test images)
    from transformers import CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    cats = ["an indoor photo", "a photo of a city street or buildings", "a nature landscape photo",
            "a close-up photo of an object"]
    groups = {c: [] for c in cats}
    for i, row in enumerate(rows):
        img = Image.fromarray(te.sharp[i].permute(1, 2, 0).numpy())
        lg = cm(**cp(text=cats, images=img, return_tensors="pt", padding=True).to(dev)).logits_per_image[0]
        groups[cats[int(lg.argmax())]].append(row["r0"])
    gm = {c: (float(np.mean(v)), len(v)) for c, v in groups.items() if v}
    res["category_means"] = gm
    means = [m for m, n in gm.values() if n >= 3]
    res["category_spread"] = float(max(means) - min(means)) if len(means) >= 2 else float("nan")
    res["category_ok"] = bool(not np.isfinite(res["category_spread"]) or res["category_spread"] <= 1.0)
    # real CamEdit pairs: P on the far region of the edited image vs the disk-fit radius
    pr, rr = [], []
    for split in ("val", "test"):
        ps = PairSet(out, pairs, split, dev)
        for i in range(len(ps.ids)):
            x = ps.blur[i:i + 1].float().to(dev) / 255
            d = ps.disp[i]
            far = (d <= torch.quantile(d.flatten()[::16], 0.4)).float()[None, None]
            far64 = F.avg_pool2d(far, 8)[0, 0].to(dev) > 0.5
            pm = predict(P, x)[0]
            if far64.any():
                pr.append(float(pm[far64].mean()))
                rr.append(float(ps.r[i]))
    res["real_pairs_spearman"] = float(spearmanr(rr, pr)[0]) if len(pr) > 3 else float("nan")
    res["real_pairs_n"] = len(pr)
    res["real_pairs_mae"] = float(np.mean(np.abs(np.array(pr) - np.array(rr)))) if pr else float("nan")
    res["ok"] = bool(res["gate_blur"] and res["gate_contrast"])
    C.jdump(dict(summary=res, rows=rows), out / "tfg" / "validation.json")
    return res


# ---------------------------------------------------------------- upstream TFG bridge
def load_tfg_class():
    if "methods.tfg" in sys.modules:
        return sys.modules["methods.tfg"].TFGGuidance
    src = (C.TFG / "tasks" / "utils.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rescale_grad")
    tasks = types.ModuleType("tasks")
    tasks.__path__ = []
    tb = types.ModuleType("tasks.base")

    class BaseGuider:                                          # replaced by our guider after construction
        def __init__(self, args):
            self.args = args
    tb.BaseGuider = BaseGuider
    tu = types.ModuleType("tasks.utils")
    tu.torch = torch
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(C.TFG / "tasks" / "utils.py"), "exec"), tu.__dict__)
    ut = types.ModuleType("utils")
    ut.__path__ = []
    uc = types.ModuleType("utils.configs")
    uc.Arguments = object
    for name, mod in (("tasks", tasks), ("tasks.base", tb), ("tasks.utils", tu), ("utils", ut),
                      ("utils.configs", uc)):
        sys.modules[name] = mod
    sys.path.insert(0, str(C.TFG))
    from methods.tfg import TFGGuidance
    return TFGGuidance


class BlurGuider:
    """TFG guider: logp(x0 latent) = -(mean over background of P(decode(x0)) - T)^2."""

    def __init__(self, sd, P, target):
        self.sd, self.P, self.T = sd, P, target
        self.m64 = None
        self.last = None

    def set_mask(self, bg512):
        m = torch.from_numpy(bg512.astype(np.float32))[None, None]
        self.m64 = (F.avg_pool2d(m, 8)[0, 0] > 0.5).to(self.sd.dev)
        if not self.m64.any():
            self.m64 = torch.ones_like(self.m64)

    def reading(self, x0):
        img = (self.sd.decode32(x0, grad=torch.is_grad_enabled()).clamp(-1, 1) + 1) / 2
        rm = predict(self.P, img)
        return rm[:, self.m64].mean(1)

    def get_guidance(self, x, return_logp=False, check_grad=True, **kw):
        with torch.enable_grad():
            r = self.reading(x)
            logp = -(r - self.T) ** 2
            if x.requires_grad:
                self.sd.counts["vae_bwd"] += x.shape[0]
            self.last = r.detach()
            if return_logp:
                return logp
            return torch.autograd.grad(logp.sum(), x)[0]


def make_tfg(sd, P, m, iter_steps=2):
    TFGGuidance = load_tfg_class()
    args = types.SimpleNamespace(device=torch.device(sd.dev), seed=0, rho=0.25 * m, mu=2.0 * m, sigma=0.1, eps_bsz=1,
                                 recur_steps=1, iter_steps=iter_steps, rho_schedule="increase",
                                 mu_schedule="increase", sigma_schedule="decrease", clip_x0=False,
                                 clip_sample_range=1.0, clip_scale=100.0)
    g = TFGGuidance(args, noise_fn=None)
    g.guider = BlurGuider(sd, P, C.T_BLUR)
    return g
