"""Pareto pilot: shared settings, logging, stage manifests and the Stable Diffusion sampler used by every arm.

One sampler serves the baseline, NA-RFM and TFG so that prompts, initial noise, scheduler, step count and CFG are
identical. Frozen settings are in README.md (Design).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("P48_OUT", ROOT / "outputs" / "pareto48"))
NARFM = ROOT / "downloads" / "narfm" / "na-rfm"
TFG = ROOT / "downloads" / "narfm" / "Training-Free-Guidance"
sys.path.insert(0, str(ROOT / "bokeh" / "common"))

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
STEPS, CFG, SIZE = 50, 7.5, 512
T_BLUR = 3.0                                    # target added background blur, px at 512
TOL = max(1.33, 0.2 * T_BLUR)
SCHEDULES = {"early": list(range(0, 17)), "middle": list(range(17, 34)), "late": list(range(34, 50)),
             "all": list(range(0, 50))}
SIG_COLLECT = 0.344
W_HOOK, S_BASE = 0.7, 4.0
LAYERS = {"mid": "mid_block", "attn": "up_blocks.1.attentions.1", "late": "up_blocks.2.resnets.2"}


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def jload(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


# ---------------------------------------------------------------- stage manifests
def manifest_path(stage):
    return OUT / "manifests" / f"{stage}.json"


def stage_done(stage):
    """A stage counts as done only if its manifest exists, says ok, and every file it lists still exists."""
    m = jload(manifest_path(stage))
    if not m or m.get("status") != "ok":
        return False
    missing = [f for f in m.get("files", []) if not (OUT / f).exists()]
    if missing:
        log(f"stage {stage}: manifest present but {len(missing)} files missing (e.g. {missing[0]}); rerunning")
        return False
    return True


def finish_stage(stage, files, **info):
    jdump(dict(status="ok", finished=time.strftime("%F %T"), files=[str(f) for f in files], **info),
          manifest_path(stage))


def fail_stage(stage, reason, **info):
    jdump(dict(status="failed", finished=time.strftime("%F %T"), reason=reason, **info), manifest_path(stage))


def require(*stages):
    bad = [s for s in stages if not stage_done(s)]
    if bad:
        raise RuntimeError(f"dependency not satisfied: {bad}")


def design_ok():
    t = (HERE / "README.md").read_text()
    s = t[t.index("## Design"):t.index("## Status")]
    return hashlib.sha256(s.encode()).hexdigest() == (HERE / "design.sha256").read_text().split()[0]


def module_checksum(mod):
    """Cheap exact fingerprint of a module's weights (sum and sum of squares per tensor, float64)."""
    acc = hashlib.sha256()
    with torch.no_grad():
        for n, p in sorted(mod.state_dict().items()):
            v = p.detach().double()
            acc.update(f"{n}:{float(v.sum()):.10e}:{float((v * v).sum()):.10e}".encode())
    return acc.hexdigest()


def resolve(mod, dotted):
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


# ---------------------------------------------------------------- hooks
class SteerHook:
    """NA-RFM's additive adaptive edit h <- h + w |h| d/|d| (upstream ActivationGuidanceHook, guidance_type
    'additive', adaptive_strength=True), applied to batch rows [lo, hi) only while armed. Counts calls."""

    def __init__(self, module, name):
        self.name, self.d, self.w, self.rows = name, None, 0.0, None
        self.calls, self.applied, self.last = 0, 0, None
        self.capture = False
        self.handle = module.register_forward_hook(self.fn)

    def fn(self, mod, inp, out):
        main = out[0] if isinstance(out, tuple) else out
        if hasattr(main, "sample"):
            main = main.sample
        self.calls += 1
        if self.capture:
            self.last = main.detach()
        if self.d is None or self.rows is None:
            return out
        lo, hi = self.rows
        h = main[lo:hi].float()
        flat = h.flatten(1)
        norm = flat.norm(dim=1, keepdim=True)
        d = self.d.reshape(1, -1)
        g = (flat + self.w * norm * d / (d.norm() + 1e-8)).reshape_as(h).to(main.dtype)
        new = main.clone()
        new[lo:hi] = g
        self.applied += 1
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        if hasattr(out, "sample"):
            out.sample = new
            return out
        return new


# ---------------------------------------------------------------- sampler
class SD:
    def __init__(self, device="cuda", memsave=False):
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        self.dev = device
        self.dt = torch.float16
        pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=self.dt, safety_checker=None,
                                                       requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(STEPS)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(device)
        self.unet, self.sched = pipe.unet, pipe.scheduler
        self.unet.requires_grad_(False)
        self.vae32 = pipe.vae.to(torch.float32)                 # one VAE, kept in fp32 (decode + TFG guidance)
        self.vae32.requires_grad_(False)
        self.sf = pipe.vae.config.scaling_factor
        if memsave:                          # predefined OOM retry: slicing + gradient checkpointing (no dropout in
            pipe.enable_attention_slicing()  # SD 1.5's UNet/VAE, so train() only switches checkpointing on)
            self.unet.enable_gradient_checkpointing()
            self.unet.train()
            self.vae32.enable_gradient_checkpointing()
            self.vae32.decoder.train()
        self.memsave = memsave
        self.ts = list(self.sched.timesteps)
        ac = self.sched.alphas_cumprod.double()
        self.ac = ac
        self.sig_all = ((1 - ac) / ac).sqrt()
        self.hooks = {}
        self.counts = dict(unet_fwd=0, unet_bwd=0, vae_bwd=0)
        self.dtypes = dict(unet=str(self.unet.dtype), vae=str(self.vae32.dtype))

    def hook(self, key):
        if key not in self.hooks:
            self.hooks[key] = SteerHook(resolve(self.unet, LAYERS[key]), key)
        return self.hooks[key]

    def sigma_at(self, k):
        return float(self.sig_all[int(self.ts[k])])

    def alpha_at(self, k):
        return float(self.ac[int(self.ts[k])])

    @torch.no_grad()
    def embed(self, texts):
        p = self.pipe
        ids = p.tokenizer(texts, padding="max_length", max_length=p.tokenizer.model_max_length, truncation=True,
                          return_tensors="pt").input_ids
        return p.text_encoder(ids.to(self.dev))[0]

    def init_latent(self, seed):
        z = torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(seed))[None]
        return (z * self.sched.init_noise_sigma).to(self.dev, self.dt)

    def eps(self, z, t, emb):
        self.counts["unet_fwd"] += z.shape[0]
        return self.unet(z, t, encoder_hidden_states=emb).sample

    def decode32(self, z, grad=False):
        x = z.float() / self.sf
        return self.vae32.decode(x).sample

    @torch.no_grad()
    def to_uint8(self, z):
        x = self.decode32(z).clamp(-1, 1)
        return ((x[0] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    # -------------------------------------------------- baseline and NA-RFM
    @torch.no_grad()
    def sample(self, prompt, seed, narfm=None):
        """narfm = dict(layer, d, w, s, gamma, steps, na) or None for plain CFG. Returns (uint8 image, info)."""
        uncond, cond = self.embed([""]), self.embed([prompt])
        z = self.init_latent(seed)
        info = dict(steps_steered=[], na_ratio=[])
        h = None
        if narfm is not None:
            h = self.hook(narfm["layer"])
            h.calls = h.applied = 0
        for k, t in enumerate(self.ts):
            on = narfm is not None and k in narfm["steps"]
            out = self.eps(torch.cat([z, z]), t, torch.cat([uncond, cond]))   # same call as the baseline
            eu, ec = out[:1].float(), out[1:2].float()
            if on:                                                  # hooked conditional branch: its own call
                h.d, h.w, h.rows = narfm["d"], narfm["w"], (0, 1)
                try:
                    eh = self.eps(z, t, cond).float()
                finally:
                    h.d, h.rows = None, None
            e = eu + CFG * (ec - eu)
            a = self.alpha_at(k)
            zf = z.float()
            x0 = (zf - (1 - a) ** 0.5 * e) / a ** 0.5            # every step goes through x0 (same arithmetic)
            if on:
                x0h = (zf - (1 - a) ** 0.5 * eh) / a ** 0.5
                x0c = (zf - (1 - a) ** 0.5 * ec) / a ** 0.5
                rfm_term = narfm["s"] * (x0h - x0c)               # upstream RFMCFGWrapper, x0 level, no clipping
                x0 = x0 + rfm_term
                if narfm.get("na") is not None and narfm["gamma"] != 0:
                    cls, full = narfm["na"]
                    sig, xe = self.sigma_at(k), zf / a ** 0.5      # upstream GaussianX0Guidance: x_edm = x / sqrt(a)
                    na_term = narfm["gamma"] * (cls.denoise(xe, sig) - full.denoise(xe, sig)).view_as(x0)
                    x0 = x0 + na_term
                    info["na_ratio"].append(float(na_term.norm() / (rfm_term.norm() + 1e-12)))
                info["steps_steered"].append(k)
            e = (zf - a ** 0.5 * x0) / (1 - a) ** 0.5
            z = self.sched.step(e, t, zf).prev_sample.to(self.dt)   # DDIM update in fp32, state kept in fp16
        if h is not None:
            info["hook_calls"], info["hook_applied"] = h.calls, h.applied
        return self.to_uint8(z), info

    @torch.no_grad()
    def sample_split(self, prompt, seed):
        """Numerical-floor variant of plain CFG: identical maths, but uncond and cond are two batch-1 UNet calls
        instead of one batch-2 call. No steering code is involved."""
        uncond, cond = self.embed([""]), self.embed([prompt])
        z = self.init_latent(seed)
        for k, t in enumerate(self.ts):
            eu, ec = self.eps(z, t, uncond).float(), self.eps(z, t, cond).float()
            e = eu + CFG * (ec - eu)
            a = self.alpha_at(k)
            zf = z.float()
            x0 = (zf - (1 - a) ** 0.5 * e) / a ** 0.5
            e = (zf - a ** 0.5 * x0) / (1 - a) ** 0.5
            z = self.sched.step(e, t, zf).prev_sample.to(self.dt)
        return self.to_uint8(z)

    # -------------------------------------------------- TFG (upstream TFGGuidance.guide_step)
    def sample_tfg(self, prompt, seed, guidance, steps):
        """guidance: an upstream TFGGuidance whose args.rho / args.mu are switched off outside `steps`."""
        uncond, cond = self.embed([""]), self.embed([prompt])
        emb = torch.cat([uncond, cond])
        z = self.init_latent(seed).float()
        ts = torch.stack([torch.as_tensor(t) for t in self.ts]).to(self.dev)
        apt = self.sched.alphas_cumprod[torch.as_tensor([int(t) for t in self.ts])].to(self.dev)
        app = torch.cat([apt[1:], torch.ones(1, device=self.dev) * self.sched.final_alpha_cumprod.to(self.dev)])
        rho0, mu0 = guidance.args.rho, guidance.args.mu
        sd = self

        def unet_cfg(x, t):
            xx = torch.cat([x, x]).to(sd.dt)
            sd.counts["unet_fwd"] += 2
            if torch.is_grad_enabled() and x.requires_grad:
                sd.counts["unet_bwd"] += 2
            out = sd.unet(xx, t, encoder_hidden_states=emb).sample.float()
            eu, ec = out.chunk(2)
            return eu + CFG * (ec - eu)

        info = dict(steps_steered=[])
        guidance.generator = torch.Generator().manual_seed(seed)          # MC noise reproducible per image
        try:
            for k in range(len(self.ts)):
                on = k in steps
                guidance.args.rho = rho0 if on else 0.0
                guidance.args.mu = mu0 if on else 0.0
                if on:
                    info["steps_steered"].append(k)
                with torch.no_grad():                 # upstream enables grad itself where it differentiates
                    z = guidance.guide_step(z, k, unet_cfg, ts, apt.float(), app.float(), 0.0).detach()
                z = z.to(self.dt).float()              # same fp16 state rounding as the baseline sampler
        finally:
            guidance.args.rho, guidance.args.mu = rho0, mu0
        return self.to_uint8(z.to(self.dt)), info


def seeded_random_direction(dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def gpu_mem():
    if not torch.cuda.is_available():
        return {}
    return dict(peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                total_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
                name=torch.cuda.get_device_name(0))
