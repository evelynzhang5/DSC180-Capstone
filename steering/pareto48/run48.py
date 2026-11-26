"""Pareto pilot driver: resumable stages with manifests and explicit dependencies (README: Sequence and decision rules).

    python steering/pareto48/run48.py <stage> [<stage> ...]

Stages: s0_env s1_data s1_refs s1_validate s2_narfm s2_tfg s3_smoke s4a_layers s4b_sweep s4c_select
        bprep_dev bscore_dev s6_gate s6_confirm bprep_confirm bscore_confirm s7_report
A stage is skipped only if its manifest says ok and its listed outputs exist. A stage whose dependencies are not
satisfied raises. One OOM retry per stage with memory-saving settings; no other retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

import common48 as C
import metrics48 as Mx
import prompts48 as PR

OUT = C.OUT
CAP_H = 8.0
PROJECT_H = 7.0


class Ctx:
    def __init__(self):
        self.memsave = os.environ.get("P48_MEMSAVE", "0") == "1"
        self._sd = self._M = self._P = self._dirs = self._na = None
        self.scorers = {}
        self.records = {}
        rp = OUT / "records.jsonl"
        if rp.exists():
            for line in rp.open():
                r = json.loads(line)
                self.records[r["key"]] = r

    @property
    def sd(self):
        if self._sd is None:
            self._sd = C.SD("cuda", memsave=self.memsave)
        return self._sd

    @property
    def M(self):
        if self._M is None:
            self._M = Mx.Models("cuda")
        return self._M

    @property
    def P(self):
        if self._P is None:
            import tfg48
            self._P = tfg48.load_P(OUT)
        return self._P

    @property
    def dirs(self):
        if self._dirs is None:
            self._dirs = torch.load(OUT / "narfm" / "directions.pt")
        return self._dirs

    @property
    def na(self):
        if self._na is None:
            import narfm48
            self._na = narfm48.load_na(OUT)
        return self._na

    def reset_gpu(self):
        self._sd = self._M = self._P = self._na = None
        self.scorers = {}
        torch.cuda.empty_cache()


def elapsed_h():
    s0 = C.jload(C.manifest_path("s0_env"))
    if not s0:
        return 0.0
    return (time.time() - s0["t_start"]) / 3600


def cases(split):
    return {c[0]: c for c in PR.cases(split)}


def ref_path(split, cid):
    return OUT / "refs" / split / f"{cid}.png"


def reg_path(split, cid):
    return OUT / "regions" / split / f"{cid}.npz"


def rules():
    return C.jload(OUT / "rules48.json", {}).get("rules", Mx.RULES)


def scorer(ctx, split, cid):
    k = (split, cid)
    if k not in ctx.scorers:
        ref = Mx.arr(Image.open(ref_path(split, cid)))
        ctx.scorers[k] = Mx.Scorer(ctx.M, ref, Mx.load_regions(reg_path(split, cid)))
    return ctx.scorers[k]


def p_reading(ctx, a, split, cid):
    """The TFG predictor's background reading on an image (logged for comparison; never the evaluator)."""
    if not (OUT / "tfg" / "P.pt").exists():
        return None
    import tfg48
    R = scorer(ctx, split, cid).R
    x = torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        rm = tfg48.predict(ctx.P, x)[0]
    m = torch.nn.functional.avg_pool2d(torch.from_numpy(R["background"].astype(np.float32))[None, None], 8)[0, 0]
    m = (m > 0.5).cuda()
    return float(rm[m].mean()) if m.any() else float(rm.mean())


def cfg_key(method, cfg):
    if method in ("base", "posthoc", "bokeh"):
        return method
    if method in ("narfm", "random"):
        return f"{method}|{cfg['layer']}|m{cfg['m']:g}|{cfg['sched']}"
    probe = f"|probe{cfg['m0']:g}" if "m0" in cfg else ""
    return f"tfg|m{cfg['m']:g}|{cfg['sched']}|it{cfg.get('iter_steps', 2)}{probe}"


def run_one(ctx, split, cid, method, cfg=None, save=True, phase=""):
    """Generate (if needed) and score one output; resumable via records.jsonl."""
    cfg = cfg or {}
    ck = cfg_key(method, cfg)
    key = f"{split}|{cid}|{ck}"
    img_p = OUT / "images" / split / ck.replace("|", "_") / f"{cid}.png"
    if key in ctx.records and img_p.exists():
        return ctx.records[key]
    _, prompt, noun, seed = cases(split)[cid]
    sd = ctx.sd if method in ("narfm", "random", "tfg") else None
    c0 = dict(sd.counts) if sd else {}
    torch.cuda.synchronize()
    t0 = time.time()
    info = {}
    if method in ("narfm", "random"):
        d = ctx.dirs[cfg["layer"]] if method == "narfm" else C.seeded_random_direction(
            ctx.dirs[cfg["layer"]].numel(), 777)
        g = C.jload(OUT / "smoke.json")["narfm_gamma0"]
        a, info = sd.sample(prompt, seed, dict(layer=cfg["layer"], d=d.cuda(), w=C.W_HOOK, s=C.S_BASE * cfg["m"],
                                               gamma=g * cfg["m"], steps=C.SCHEDULES[cfg["sched"]], na=ctx.na))
    elif method == "tfg":
        import tfg48
        m0 = cfg["m0"] if "m0" in cfg else C.jload(OUT / "smoke.json")["tfg_m0"]
        g = tfg48.make_tfg(sd, ctx.P, cfg["m"] * m0, iter_steps=cfg.get("iter_steps", 2))
        g.guider.set_mask(scorer(ctx, split, cid).R["background"])
        a, info = sd.sample_tfg(prompt, seed, g, set(C.SCHEDULES[cfg["sched"]]))
    elif method == "posthoc":
        S = scorer(ctx, split, cid)
        a = np.clip(Mx.posthoc(S.ref, S.R["subject"], C.T_BLUR), 0, 255).round().astype(np.uint8)
    elif method == "bokeh":
        a = np.asarray(Image.open(OUT / "bokeh" / split / "run" / f"{cid}.png").convert("RGB")
                       .resize((512, 512), Image.LANCZOS))
    else:
        raise ValueError(method)
    torch.cuda.synchronize()
    wall = time.time() - t0
    a = np.asarray(a)
    finite = bool(np.isfinite(a.astype(np.float64)).all())
    img_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(img_p)
    S = scorer(ctx, split, cid)
    m = S.score(a.astype(np.float64), p_reading(ctx, a, split, cid))
    Mx.judge(m, rules())
    rec = dict(key=key, split=split, case=cid, method=method, cfg=cfg, cfg_key=ck, phase=phase, wall_s=wall,
               finite=finite, img_std=float(a.std()), steps_steered=info.get("steps_steered", []),
               hook_applied=info.get("hook_applied"), na_ratio=float(np.median(info["na_ratio"]))
               if info.get("na_ratio") else None, **{k: v - c0.get(k, 0) for k, v in (sd.counts if sd else {}).items()},
               **m)
    with (OUT / "records.jsonl").open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    ctx.records[key] = rec
    return rec


def recs(ctx, split=None, method=None, ck=None, phase=None):
    return [r for r in ctx.records.values() if (split is None or r["split"] == split)
            and (method is None or r["method"] == method) and (ck is None or r["cfg_key"] == ck)
            and (phase is None or r.get("phase") == phase)]


# ================================================================ stages
def s0_env(ctx):
    assert C.design_ok(), "README Design section changed since design.sha256"
    import diffusers
    import transformers
    info = dict(t_start=time.time(), gpu=torch.cuda.get_device_name(0),
                gpu_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30, torch=torch.__version__,
                diffusers=diffusers.__version__, transformers=transformers.__version__,
                narfm=os.popen(f"git -C {C.NARFM} rev-parse HEAD").read().strip(),
                tfg=os.popen(f"git -C {C.TFG} rev-parse HEAD").read().strip())
    prev = C.jload(C.manifest_path("s0_env"))
    if prev and prev.get("t_start"):
        info["t_start"] = prev["t_start"]
    C.log(f"env: {info}")
    C.finish_stage("s0_env", [], **info)


def s1_data(ctx):
    import data48
    summ = data48.build(OUT, smoke=os.environ.get("SMOKE") == "1")
    ok = summ["by_split"]["train"] >= 40 and summ["by_split"]["val"] >= 10 and summ["by_split"]["test"] >= 10
    if not ok:
        C.fail_stage("s1_data", "too few eligible CamEdit pairs", **summ)
        raise RuntimeError(f"too few eligible pairs: {summ}")
    C.finish_stage("s1_data", ["camedit/pairs.json"], **summ)


def make_refs(ctx, split):
    files = []
    for cid, prompt, noun, seed in PR.cases(split):
        rp = ref_path(split, cid)
        if not rp.exists():
            a, _ = ctx.sd.sample(prompt, seed)
            rp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a).save(rp)
        gp = reg_path(split, cid)
        if not gp.exists():
            gp.parent.mkdir(parents=True, exist_ok=True)
            R = Mx.regions(ctx.M, Mx.arr(Image.open(rp)), noun)
            Mx.save_regions(gp, R)
        files += [rp.relative_to(OUT), gp.relative_to(OUT)]
    cov = {cid: dict(n_tiles=len(Mx.load_regions(reg_path(split, cid))["tiles"]),
                     thr=Mx.load_regions(reg_path(split, cid))["thr"],
                     subj_area=Mx.load_regions(reg_path(split, cid))["subj_area"],
                     bg_area=Mx.load_regions(reg_path(split, cid))["bg_area"]) for cid in cases(split)}
    sheet(OUT / "refs" / f"{split}_regions.jpg", [[overlay(split, cid)] for cid in cases(split)], cols=6)
    return files, cov


def overlay(split, cid):
    a = Mx.arr(Image.open(ref_path(split, cid)))
    R = Mx.load_regions(reg_path(split, cid))
    a[R["subject"]] = 0.55 * a[R["subject"]] + 0.45 * np.array([255, 0, 0])
    a[R["background"]] = 0.8 * a[R["background"]] + 0.2 * np.array([0, 0, 255])
    for t in R["tiles"]:
        y, x = t["y"], t["x"]
        a[y:y + 64, [x, x + 63]] = [0, 255, 255]
        a[[y, y + 63], x:x + 64] = [0, 255, 255]
    return Image.fromarray(a.astype(np.uint8))


def sheet(path, rows, cols=None, w=192):
    flat = [im for r in rows for im in r]
    cols = cols or max(len(r) for r in rows)
    n = len(flat)
    H = (n + cols - 1) // cols
    sh = Image.new("RGB", (cols * w, H * w), "white")
    for k, im in enumerate(flat):
        sh.paste(im.resize((w, w), Image.LANCZOS), ((k % cols) * w, (k // cols) * w))
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.save(path, quality=90)


def s1_refs(ctx):
    files, cov = make_refs(ctx, "dev")
    C.finish_stage("s1_refs", files + ["refs/dev_regions.jpg"], coverage=cov)


def s1_validate(ctx):
    C.require("s1_refs")
    rows = []
    for cid in cases("dev"):
        S = scorer(ctx, "dev", cid)
        for name, (img, meta) in Mx.transforms(S.ref, S.R).items():
            m = S.score(np.clip(img, 0, 255).round())
            rows.append(dict(case=cid, name=name, **meta, **{k: m[k] for k in (
                "r", "measurable", "dog_bg", "subj_sharp", "scene", "dino_subj", "iou", "shift", "de",
                "depth_rho")}))
    by = lambda n: [r for r in rows if r["name"] == n]
    med = lambda n, k: float(np.nanmedian([r[k] for r in by(n)]))
    chk = {}
    within = []
    for r in (1, 2, 3, 4, 6):
        rs = [x for x in by(f"bgblur_r{r}") if x["measurable"]]
        frac = np.mean([abs(x["r"] - r) <= max(1.33, 0.2 * r) for x in rs]) if rs else 0.0
        within.append(dict(r=r, n_measurable=len(rs), frac_within=float(frac)))
    chk["a_diskfit"] = all(w["frac_within"] >= 0.8 for w in within)
    chk["a_preservation"] = all(med(f"bgblur_r{r}", "scene") <= 0.05 and med(f"bgblur_r{r}", "dino_subj") >= 0.95
                                and med(f"bgblur_r{r}", "iou") >= 0.9 and med(f"bgblur_r{r}", "de") <= 2
                                for r in (1, 2, 3))
    cm = [x["r"] for x in by("contrast0.6") if x["measurable"]]
    chk["b_contrast_diskfit"] = bool(not cm or np.median(cm) <= 1.0)
    chk["b_contrast_dog"] = med("contrast0.6", "dog_bg")
    chk["c_shift"] = all(med(f"shift{d}", "scene") >= 0.1 and med(f"shift{d}", "shift") >= d / 2 for d in (8, 16))
    chk["d_global"] = med("global_r2", "subj_sharp") <= -0.3
    # threshold relaxation (README): only where pure background blur at r = 3 itself fails a threshold
    new, changes = dict(Mx.RULES), {}
    r3 = by("bgblur_r3")
    for k, direction in (("dino_subj", ">="), ("iou", ">="), ("shift", "<="), ("subj_sharp", ">="), ("de", "<="),
                         ("scene", "<=")):
        vals = np.array([x[k] for x in r3 if np.isfinite(x[k])])
        if not len(vals):
            continue
        fails = (vals < new[k]).any() if direction == ">=" else (vals > new[k]).any()
        if fails:
            q = float(np.quantile(vals, 0.1 if direction == ">=" else 0.9))
            nv = min(new[k], q) if direction == ">=" else max(new[k], q)
            changes[k] = dict(old=new[k], new=nv)
            new[k] = nv
    exploratory = not chk["a_diskfit"]
    C.jdump(dict(rules=new, changes=changes, checks=chk, diskfit_within=within, exploratory_blur=exploratory,
                 rows=rows), OUT / "rules48.json")
    C.log(f"validation: {chk}; disk-fit within {within}; threshold changes {changes}; exploratory {exploratory}")
    C.finish_stage("s1_validate", ["rules48.json"], checks=chk, changes=changes, exploratory_blur=exploratory)


def s2_narfm(ctx):
    C.require("s1_data")
    import narfm48
    (OUT / "narfm").mkdir(parents=True, exist_ok=True)
    pairs = C.jload(OUT / "camedit" / "pairs.json")["pairs"]
    acts = torch.load(OUT / "narfm" / "acts.pt") if (OUT / "narfm" / "acts.pt").exists() else \
        narfm48.collect(ctx.sd, OUT, pairs)
    ctx.reset_gpu()                                     # RFM needs the GPU memory
    rep = narfm48.train(acts, OUT)
    na = narfm48.build_na(acts, OUT)
    del acts
    elig = [k for k, v in rep.items() if v["eligible"]]
    C.jdump(dict(layers=rep, na=na, eligible=elig), OUT / "narfm" / "report.json")
    if not elig:
        C.fail_stage("s2_narfm", "no layer reached val AUC 0.65", layers=rep)
        return
    C.finish_stage("s2_narfm", ["narfm/directions.pt", "narfm/na_filters.pt", "narfm/report.json"], eligible=elig,
                   na=na)


def s2_tfg(ctx):
    C.require("s1_data")
    import tfg48
    (OUT / "tfg").mkdir(parents=True, exist_ok=True)
    pairs = C.jload(OUT / "camedit" / "pairs.json")["pairs"]
    tr = tfg48.train(OUT, pairs, smoke=os.environ.get("SMOKE") == "1")
    val = tfg48.validate(OUT, pairs)
    C.jdump(dict(train=tr, validation=val), OUT / "tfg" / "report.json")
    C.log(f"P: {tr['n_params']} params, best val L1 {tr['best_val_l1']:.3f}; validation {val}")
    if not val["ok"]:
        C.fail_stage("s2_tfg", "predictor failed a hard gate", validation=val)
        return
    C.finish_stage("s2_tfg", ["tfg/P.pt", "tfg/report.json", "tfg/validation.json"], validation=val)


def s3_smoke(ctx):
    C.require("s1_refs", "s1_validate")
    narfm_ok, tfg_ok = C.stage_done("s2_narfm"), C.stage_done("s2_tfg")
    sd = ctx.sd
    torch.cuda.reset_peak_memory_stats()
    sums0 = dict(unet=C.module_checksum(sd.unet), vae=C.module_checksum(sd.vae32),
                 text=C.module_checksum(sd.pipe.text_encoder))
    if tfg_ok:
        sums0["P"] = C.module_checksum(ctx.P)
    res, checks = dict(dtypes=sd.dtypes), {}
    smoke_cases = ["d01", "d02"]
    base = {}
    for cid in smoke_cases:
        _, prompt, _, seed = cases("dev")[cid]
        a1, _ = sd.sample(prompt, seed)
        a2, _ = sd.sample(prompt, seed)
        ref = np.asarray(Image.open(ref_path("dev", cid)))
        base[cid] = a1
        checks[f"baseline_repro_{cid}"] = bool(np.array_equal(a1, a2) and np.array_equal(a1, ref))
    diff = lambda a, b: (float(np.abs(a.astype(float) - b.astype(float)).max()),
                         float(np.abs(a.astype(float) - b.astype(float)).mean()))
    if narfm_ok:
        g_ratio = []
        for layer in C.jload(OUT / "narfm" / "report.json")["eligible"]:
            d = ctx.dirs[layer].cuda()
            for cid in smoke_cases:
                _, prompt, _, seed = cases("dev")[cid]
                z0, inf0 = sd.sample(prompt, seed, dict(layer=layer, d=d, w=C.W_HOOK, s=0.0, gamma=0.0,
                                                        steps=C.SCHEDULES["all"], na=ctx.na))
                checks[f"narfm_zero_strength_exact_{layer}_{cid}"] = bool(np.array_equal(z0, base[cid]))
                zw, _ = sd.sample(prompt, seed, dict(layer=layer, d=d, w=0.0, s=C.S_BASE, gamma=0.0,
                                                     steps=C.SCHEDULES["all"], na=None))
                mx, mn = diff(zw, base[cid])
                res[f"narfm_w0_{layer}_{cid}"] = dict(max=mx, mean=mn)
                checks[f"narfm_w0_tol_{layer}_{cid}"] = bool(mx <= 2 and mn <= 0.1)
            _, prompt, _, seed = cases("dev")["d01"]
            _, inf = sd.sample(prompt, seed, dict(layer=layer, d=d, w=C.W_HOOK, s=C.S_BASE, gamma=1.0,
                                                  steps=C.SCHEDULES["middle"], na=ctx.na))
            checks[f"narfm_hook_steps_{layer}"] = bool(inf["hook_applied"] == len(C.SCHEDULES["middle"])
                                                       and inf["steps_steered"] == C.SCHEDULES["middle"])
            res[f"narfm_hook_{layer}"] = dict(applied=inf["hook_applied"], calls=inf["hook_calls"])
            g_ratio.append(float(np.median(inf["na_ratio"])))
            _, inf = sd.sample(prompt, seed, dict(layer=layer, d=d, w=C.W_HOOK, s=C.S_BASE, gamma=1.0,
                                                  steps=C.SCHEDULES["all"], na=ctx.na))
            g_ratio.append(float(np.median(inf["na_ratio"])))
        rho_n = float(np.median(g_ratio))
        gamma0 = 1.0 if 0.1 <= rho_n <= 10 else 1.0 / rho_n
        res.update(narfm_na_ratio=rho_n, narfm_gamma0=gamma0)
    res["narfm_gamma0"] = res.get("narfm_gamma0", 1.0)
    C.jdump(res, OUT / "smoke.json")                        # gamma0 needed by run_one below
    if narfm_ok:
        grids = {}
        for layer in C.jload(OUT / "narfm" / "report.json")["eligible"]:
            out2 = [run_one(ctx, "dev", cid, "narfm", dict(layer=layer, m=2.0, sched="all"), phase="smoke")
                    for cid in smoke_cases]
            degen = any((not r["finite"]) or r["img_std"] < 0.5 * float(np.asarray(Image.open(
                ref_path("dev", r["case"]))).std()) for r in out2)
            noresp = all(np.isfinite(r["dog_bg"]) and r["dog_bg"] > 0.9 for r in out2)
            grids[layer] = [0.25, 0.5, 1.0] if degen else ([1.0, 2.0, 4.0] if noresp else [0.5, 1.0, 2.0])
            res[f"narfm_m2_{layer}"] = [dict(case=r["case"], dog_bg=r["dog_bg"], r=r["r"], std=r["img_std"])
                                        for r in out2]
        res["narfm_grids"] = grids
        res["narfm_s_per_image"] = float(np.median([r["wall_s"] for r in recs(ctx, phase="smoke", method="narfm")]))
    if tfg_ok:
        import tfg48
        g = tfg48.make_tfg(sd, ctx.P, 1.0)
        g.guider.set_mask(scorer(ctx, "dev", "d01").R["background"])
        z = sd.init_latent(101).float().requires_grad_(True)
        grad = g.guider.get_guidance(z, return_logp=False)
        gn = float(grad.norm())
        checks["tfg_grad_finite_nonzero"] = bool(torch.isfinite(grad).all() and gn > 0)
        res["tfg_grad_norm"] = gn
        for cid in smoke_cases:
            _, prompt, _, seed = cases("dev")[cid]
            g0 = tfg48.make_tfg(sd, ctx.P, 0.0)
            g0.guider.set_mask(scorer(ctx, "dev", cid).R["background"])
            a0, inf0 = sd.sample_tfg(prompt, seed, g0, set(C.SCHEDULES["all"]))
            mx, mn = diff(a0, base[cid])
            res[f"tfg_zero_{cid}"] = dict(max=mx, mean=mn)
            checks[f"tfg_zero_tol_{cid}"] = bool(mx <= 2 and mn <= 0.1)
        res["smoke_json_partial"] = True
        C.jdump(res, OUT / "smoke.json")
        base_read = {cid: p_reading(ctx, base[cid], "dev", cid) for cid in smoke_cases}
        m0, tries = None, []
        for cand in (1.0, 4.0, 16.0):
            rr = [run_one(ctx, "dev", cid, "tfg", dict(m=1.0, sched="all", iter_steps=2, m0=cand), phase="smoke")
                  for cid in smoke_cases]
            for r in rr:
                checks[f"tfg_steps_m0_{cand:g}_{r['case']}"] = bool(r["steps_steered"] == C.SCHEDULES["all"]
                                                                   and r["unet_bwd"] > 0 and r["vae_bwd"] > 0)
            ok = all(r["finite"] and r.get("p_reading") is not None and r["p_reading"] - base_read[r["case"]] >= 1.0
                     for r in rr)
            tries.append(dict(m0=cand, readings=[(r["case"], r.get("p_reading"), base_read[r["case"]], r["wall_s"],
                                                  r["unet_bwd"], r["vae_bwd"]) for r in rr], ok=ok))
            if ok:
                m0 = cand
                break
        res["tfg_m0_tries"] = tries
        res["tfg_m0"] = m0 if m0 is not None else 16.0
        res["tfg_weak"] = m0 is None
        res["tfg_s_per_steered_step"] = float(np.median([t[3] for tr in tries for t in tr["readings"]])) / 50
    sums1 = dict(unet=C.module_checksum(sd.unet), vae=C.module_checksum(sd.vae32),
                 text=C.module_checksum(sd.pipe.text_encoder))
    if tfg_ok:
        sums1["P"] = C.module_checksum(ctx.P)
    checks["weights_unchanged"] = sums0 == sums1
    res["memory"] = C.gpu_mem()
    checks["memory_fits"] = res["memory"]["peak_gb"] < 0.95 * res["memory"]["total_gb"]
    # timing projection and predeclared reductions (README), decided before any sweep output exists
    red = []
    if tfg_ok:
        per_step = res["tfg_s_per_steered_step"]
        steered = {k: len(v) for k, v in C.SCHEDULES.items()}
        def proj(it, n_cases, n_m):
            f = (1 + it) / 3                                 # rho pass + iter_steps decodes, relative to it = 2
            tfg_h = n_cases * n_m * sum(steered.values()) * per_step * f / 3600
            conf_h = 12 * 2 * steered["all"] * per_step * f / 3600
            return elapsed_h() + tfg_h + conf_h + 1.0          # + NA-RFM sweep, BokehDiff, metrics, report
        it, nc, nm = 2, 12, 3
        if proj(it, nc, nm) > PROJECT_H:
            it = 1
            red.append("R1")
        if proj(it, nc, nm) > PROJECT_H:
            nc = 8
            red.append("R2")
        if proj(it, nc, nm) > PROJECT_H:
            nm = 2
            red.append("R3")
        res["tfg_plan"] = dict(iter_steps=it, n_cases=nc, strengths=[0.5, 1.0, 2.0][3 - nm:], projected_h=proj(it, nc, nm))
    res["reductions"] = red
    res["checks"] = checks
    res["elapsed_h"] = elapsed_h()
    C.jdump(res, OUT / "smoke.json")
    C.log(f"smoke checks: {checks}")
    C.log(f"smoke settings: gamma0 {res.get('narfm_gamma0')}, grids {res.get('narfm_grids')}, tfg m0 "
          f"{res.get('tfg_m0')}, plan {res.get('tfg_plan')}, reductions {red}")
    failed = [k for k, v in checks.items() if not v]
    narfm_bad = [k for k in failed if k.startswith("narfm") or k.startswith("baseline") or k in
                 ("weights_unchanged", "memory_fits")]
    tfg_bad = [k for k in failed if k.startswith("tfg") or k.startswith("baseline") or k in
               ("weights_unchanged", "memory_fits")]
    C.finish_stage("s3_smoke", ["smoke.json"], checks=checks, failed=failed,
                   narfm_valid=bool(narfm_ok and not narfm_bad), tfg_valid=bool(tfg_ok and not tfg_bad))


def valid(method):
    m = C.jload(C.manifest_path("s3_smoke"))
    return bool(m and m.get("status") == "ok" and m.get(f"{method}_valid"))


def s4a_layers(ctx):
    C.require("s3_smoke")
    if not valid("narfm"):
        C.finish_stage("s4a_layers", [], skipped="NA-RFM not valid after smoke")
        return
    elig = C.jload(OUT / "narfm" / "report.json")["eligible"]
    score = {}
    for layer in elig:
        rr = [run_one(ctx, "dev", f"d0{i}", "narfm", dict(layer=layer, m=1.0, sched="middle"), phase="layers")
              for i in range(1, 5)]
        n = sum(r["response"] and r["dino_subj"] >= rules()["dino_subj"] and r["iou"] >= rules()["iou"] for r in rr)
        score[layer] = dict(n=int(n), mean_scene=float(np.mean([r["scene"] for r in rr])))
    pick = sorted(score, key=lambda k: (-score[k]["n"], score[k]["mean_scene"]))[0]
    C.log(f"layer screen: {score} -> {pick}")
    C.finish_stage("s4a_layers", [], scores=score, layer=pick)


def sweep_configs(method):
    sm = C.jload(OUT / "smoke.json")
    if method == "narfm":
        layer = C.jload(C.manifest_path("s4a_layers"))["layer"]
        return [dict(layer=layer, m=m, sched=s) for m in sm["narfm_grids"][layer] for s in C.SCHEDULES], \
            [f"d{i:02d}" for i in range(1, 13)]
    plan = sm["tfg_plan"]
    return [dict(m=m, sched=s, iter_steps=plan["iter_steps"]) for m in plan["strengths"] for s in C.SCHEDULES], \
        [f"d{i:02d}" for i in range(1, plan["n_cases"] + 1)]


def s4b_sweep(ctx):
    C.require("s3_smoke")
    done = {}
    for method in ("narfm", "tfg"):
        if not valid(method) or (method == "narfm" and not C.stage_done("s4a_layers")):
            done[method] = "skipped"
            continue
        cfgs, cids = sweep_configs(method)
        for cfg in cfgs:
            for cid in cids:
                if elapsed_h() > CAP_H - 1.0:
                    C.fail_stage("s4b_sweep", "GPU-hour cap reached during sweep", elapsed_h=elapsed_h())
                    raise RuntimeError("budget cap")
                r = run_one(ctx, "dev", cid, method, cfg, phase="sweep")
            C.log(f"{method} {cfg}: done ({elapsed_h():.2f} h)")
        done[method] = len(cfgs) * len(cids)
    C.finish_stage("s4b_sweep", ["records.jsonl"], done=done)


def config_table(ctx, method, split="dev"):
    cfgs, _ = sweep_configs(method)
    tab = []
    for cfg in cfgs:
        rr = recs(ctx, split, method, cfg_key(method, cfg))
        if not rr:
            continue
        tab.append(dict(cfg=cfg, key=cfg_key(method, cfg), n=len(rr), response=sum(r["response"] for r in rr),
                        preserved=sum(r["preserved"] for r in rr),
                        both=sum(r["response"] and r["preserved"] for r in rr),
                        med_scene=float(np.median([r["scene"] for r in rr])),
                        med_err=float(np.nanmedian([r["target_err"] for r in rr]))
                        if any(np.isfinite(r["target_err"]) for r in rr) else float("nan"),
                        measurable=sum(r["measurable"] for r in rr)))
    return tab


def s4c_select(ctx):
    C.require("s4b_sweep")
    sel = {}
    for method in ("narfm", "tfg"):
        if not valid(method):
            sel[method] = dict(status="invalid")
            continue
        tab = config_table(ctx, method)
        need = lambda t, frac: t["response"] >= np.ceil(frac * t["n"])      # 8/12 and 6/12 scaled if R2 applies
        resp = [t for t in tab if need(t, 2 / 3)]
        if not resp:
            sel[method] = dict(status="no repeatable response", table=tab)
            continue
        P_ = sorted(resp, key=lambda t: (-t["both"], t["med_scene"]))[0]
        pool = [t for t in resp if t["preserved"] >= np.ceil(t["n"] / 2)] or resp
        B_ = sorted(pool, key=lambda t: (t["med_err"] if np.isfinite(t["med_err"]) else 1e9, t["med_scene"]))[0]
        chosen = [P_] if B_["key"] == P_["key"] else [P_, B_]
        sel[method] = dict(status="selected", P=P_, B=B_, configs=[t["cfg"] for t in chosen], table=tab)
    # NA-RFM random-direction comparison at the selected configurations (dev only)
    if sel.get("narfm", {}).get("status") == "selected":
        for cfg in sel["narfm"]["configs"]:
            for i in range(1, 13):
                run_one(ctx, "dev", f"d{i:02d}", "random", cfg, phase="sweep")
    text = json.dumps(sel, indent=1, default=float, sort_keys=True)
    (OUT / "selection48.json").write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    C.log(f"selection frozen: sha256 {h}; " + "; ".join(f"{k}: {v['status']}" for k, v in sel.items()))
    C.finish_stage("s4c_select", ["selection48.json"], sha256=h)


def bprep(ctx, split):
    d = OUT / "bokeh" / split / "data" / "input"
    d.mkdir(parents=True, exist_ok=True)
    for cid in cases(split):
        Image.open(ref_path(split, cid)).convert("RGB").resize((768, 768), Image.LANCZOS).save(d / f"{cid}.png")
    C.finish_stage(f"bprep_{split}", [f"bokeh/{split}/data/input/{c}.png" for c in cases(split)])


def bscore(ctx, split):
    j = C.jload(OUT / "bokeh" / split / "run" / "bd48.json")
    if j is None:
        C.fail_stage(f"bscore_{split}", "no BokehDiff output")
        raise RuntimeError("BokehDiff output missing")
    for row in j["rows"]:
        if row["rendered"]:
            run_one(ctx, split, row["id"], "bokeh", phase="bokeh")
    for cid in cases(split):
        run_one(ctx, split, cid, "posthoc", phase="anchor")
    C.finish_stage(f"bscore_{split}", ["records.jsonl"], n_rendered=sum(r["rendered"] for r in j["rows"]),
                   ineligible=[r["id"] for r in j["rows"] if not r["rendered"]])


def s6_gate(ctx):
    C.require("s4c_select")
    sel = json.loads((OUT / "selection48.json").read_text())
    go = {}
    for method, s in sel.items():
        if s.get("status") != "selected":
            go[method] = dict(go=False, why=s.get("status"))
            continue
        tab = s["table"]
        n = max(t["n"] for t in tab)
        need = int(np.ceil(2 * n / 3))
        go[method] = dict(go=bool(max(t["response"] for t in tab) >= need and max(t["both"] for t in tab) >= need),
                          max_response=max(t["response"] for t in tab), max_both=max(t["both"] for t in tab),
                          n_cases=n, need=need)
    C.jdump(go, OUT / "gate48.json")
    C.log(f"confirmation gate: {go}")
    C.finish_stage("s6_gate", ["gate48.json"], go=go)


def s6_confirm(ctx):
    C.require("s6_gate")
    go = C.jload(OUT / "gate48.json")
    sel = json.loads((OUT / "selection48.json").read_text())
    methods = [m for m, g in go.items() if g["go"]]
    if not methods:
        (OUT / "confirm_go").unlink(missing_ok=True)
        C.finish_stage("s6_confirm", [], skipped="no method qualified for confirmation")
        return
    files, cov = make_refs(ctx, "confirm")                  # generated only now, after the selection is frozen
    for method in methods:
        for cfg in sel[method]["configs"]:
            for cid in cases("confirm"):
                if elapsed_h() > CAP_H - 0.3:
                    C.fail_stage("s6_confirm", "GPU-hour cap", elapsed_h=elapsed_h())
                    raise RuntimeError("budget cap")
                run_one(ctx, "confirm", cid, method, cfg, phase="confirm")
    (OUT / "confirm_go").write_text(",".join(methods))
    C.finish_stage("s6_confirm", files, methods=methods, coverage=cov)


def s7_report(ctx):
    import report48
    report48.main(ctx)
    C.finish_stage("s7_report", ["report.md"])


STAGES = dict(s0_env=s0_env, s1_data=s1_data, s1_refs=s1_refs, s1_validate=s1_validate, s2_narfm=s2_narfm,
              s2_tfg=s2_tfg, s3_smoke=s3_smoke, s4a_layers=s4a_layers, s4b_sweep=s4b_sweep, s4c_select=s4c_select,
              bprep_dev=lambda c: bprep(c, "dev"), bscore_dev=lambda c: bscore(c, "dev"), s6_gate=s6_gate,
              s6_confirm=s6_confirm, bprep_confirm=lambda c: bprep(c, "confirm"),
              bscore_confirm=lambda c: bscore(c, "confirm"), s7_report=s7_report)
ALWAYS = {"s0_env", "s7_report"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = Ctx()
    for st in a.stages:
        if st not in ALWAYS and C.stage_done(st) and not a.force:
            C.log(f"stage {st}: done (manifest valid), skipping")
            continue
        C.log(f"stage {st} (elapsed {elapsed_h():.2f} h)")
        for attempt in (0, 1):
            try:
                STAGES[st](ctx)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == 1 or ctx.memsave:
                    C.fail_stage(st, "OOM after the predefined memory-saving retry")
                    raise
                C.log(f"stage {st}: OOM; one retry with memory-saving settings (attention slicing, checkpointed "
                      f"VAE decode)")
                ctx.memsave = True
                ctx.reset_gpu()
            except Exception as e:
                if C.jload(C.manifest_path(st), {}).get("status") != "failed":
                    C.fail_stage(st, f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                raise
    C.log("driver done")


if __name__ == "__main__":
    main()
