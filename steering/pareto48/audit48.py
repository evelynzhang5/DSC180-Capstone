"""Audit of round 48 (README: "Audit of round 48"). Dev cases only; the confirmation prompts are never generated.

    python steering/pareto48/audit48.py numerics p2 trace analyze

numerics  A: baseline / zero-strength / floor / TFG-zero pairs under identical execution conditions (pixel percentiles
          and metric deltas), and NA-RFM's sensitivity to round 48's mixed batch shapes (m 0.5 late, m 1 all)
p2        a second blur predictor (other seed, even-sha1 half of the train scenes), same validation gates
trace     C: TFG late m1, late m2 (10 cases) and all m0.5 (4 cases) re-run with recording wrappers around upstream
          TFGGuidance (pre/post x0 estimates per guided step, P readings, gradient norms, mu/rho, finite-difference
          gradient checks); finals compared bitwise with round 48's saved outputs
analyze   independent measures on every pre/post pair and final, P2 readings, known-blur controls (offline accuracy),
          E1-E4 classification, B: shared-eligible method comparison; writes audit/report.md and audit/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common48 as C
import metrics48 as Mx
import run48 as D
import tfg48

OUT = C.OUT
A = OUT / "audit"
TRACE_CFGS = [("late", 1.0, None), ("late", 2.0, None), ("all", 0.5, ["d01", "d03", "d05", "d07"])]
RES = dict(scene=0.01, r=0.25, dog_bg=0.02, dino_subj=0.01)


def pstats(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).ravel()
    q = lambda p: float(np.quantile(d, p))
    return dict(mean=float(d.mean()), p50=q(0.5), p90=q(0.9), p99=q(0.99), p999=q(0.999), max=float(d.max()),
                frac_nonzero=float((d > 0).mean()))


def mdiff(ma, mb):
    out = {}
    for k in RES:
        a, b = ma.get(k), mb.get(k)
        out[k] = float(a - b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
    return out


def save(a, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(a, np.uint8)).save(p)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def base_guidance(sd):
    """Upstream TFG's own unguided sampler (methods/base.py BaseGuidance.guide_step), same args as make_tfg."""
    import types
    tfg48.load_tfg_class()
    from methods.base import BaseGuidance
    args = types.SimpleNamespace(device=torch.device(sd.dev), seed=0, rho=0.0, mu=0.0, sigma=0.1, eps_bsz=1,
                                 recur_steps=1, iter_steps=2, rho_schedule="increase", mu_schedule="increase",
                                 sigma_schedule="decrease", clip_x0=False, clip_sample_range=1.0, clip_scale=100.0)
    return BaseGuidance(args, noise_fn=None)


def narfm_cfg(ctx, m, sched, **extra):
    g0 = C.jload(OUT / "smoke.json")["narfm_gamma0"]
    return dict(layer="mid", d=ctx.dirs["mid"].cuda(), w=C.W_HOOK, s=C.S_BASE * m, gamma=g0 * m,
                steps=C.SCHEDULES[sched], na=ctx.na, **extra)


# ================================================================ A: numerics
def numerics(ctx):
    sd = ctx.sd
    rows, sens = [], []
    for cid in D.eligible_cases("dev"):
        _, prompt, _, seed = D.cases("dev")[cid]
        S = D.scorer(ctx, "dev", cid)
        sc = lambda a: Mx.judge(S.score(np.asarray(a, np.float64)), D.rules())
        base, _ = sd.sample(prompt, seed)
        mb = sc(base)
        d = ctx.dirs["mid"].cuda()
        v = {}
        v["2_narfm_s0"], _ = sd.sample(prompt, seed, dict(layer="mid", d=d, w=C.W_HOOK, s=0.0, gamma=0.0,
                                                          steps=C.SCHEDULES["all"], na=ctx.na))
        v["3_narfm_w0_mixed"], _ = sd.sample(prompt, seed, dict(layer="mid", d=d, w=0.0, s=C.S_BASE, gamma=0.0,
                                                                steps=C.SCHEDULES["all"], na=None))
        v["4_narfm_w0_same"], _ = sd.sample(prompt, seed, dict(layer="mid", d=d, w=0.0, s=C.S_BASE, gamma=0.0,
                                                               steps=C.SCHEDULES["all"], na=None, same_batch=True))
        v["5_floor"] = sd.sample_split(prompt, seed)
        g0 = tfg48.make_tfg(sd, ctx.P, 0.0)
        g0.guider.set_mask(S.R["background"])
        v["6_tfg0"], _ = sd.sample_tfg(prompt, seed, g0, set(C.SCHEDULES["all"]))
        v7, _ = sd.sample_tfg(prompt, seed, base_guidance(sd), set())
        pairs = {"1_base_vs_saved": (base, load(D.ref_path("dev", cid)), None),
                 "7_tfg0_vs_tfg_unguided": (v["6_tfg0"], v7, None)}
        for k in ("2_narfm_s0", "3_narfm_w0_mixed", "4_narfm_w0_same", "5_floor", "6_tfg0"):
            pairs[f"{k}_vs_base"] = (v[k], base, mdiff(sc(v[k]), mb))
        for k, (a, b, md) in pairs.items():
            rows.append(dict(case=cid, pair=k, identical=bool(np.array_equal(a, b)), **pstats(a, b), metric_delta=md))
            save(a, A / "numerics" / k / f"{cid}.png")
        C.log(f"numerics {cid}: " + ", ".join(f"{r['pair']} max {r['max']:.0f}" for r in rows if r["case"] == cid))
        # NA-RFM sensitivity to the batch-shape choice (round 48 = mixed)
        for m, sched in ((0.5, "late"), (1.0, "all")):
            out = {}
            for mode in ("mixed", "same"):
                a, _ = sd.sample(prompt, seed, narfm_cfg(ctx, m, sched, **({"same_batch": True} if mode == "same"
                                                                             else {})))
                save(a, A / "batchshape" / f"m{m:g}_{sched}_{mode}" / f"{cid}.png")
                out[mode] = (a, sc(a))
            saved = load(OUT / "images" / "dev" / f"narfm_mid_m{m:g}_{sched}" / f"{cid}.png")
            sens.append(dict(case=cid, cfg=f"m{m:g}_{sched}", mixed_rerun_identical_to_saved=bool(
                np.array_equal(out["mixed"][0], saved)), pix=pstats(out["mixed"][0], out["same"][0]),
                **{f"{mode}_{k}": out[mode][1][k] for mode in out for k in
                   ("r", "measurable", "dog_bg", "scene", "dino_subj", "response", "preserved")}))
    C.jdump(dict(rows=rows, batchshape=sens), A / "numerics.json")
    C.finish_stage("audit_numerics", ["audit/numerics.json"])


# ================================================================ C: second predictor
def p2(ctx):
    pairs = C.jload(OUT / "camedit" / "pairs.json")["pairs"]
    even = {p["id"] for p in pairs if p["split"] == "train" and int(hashlib.sha1(p["id"].encode()).hexdigest(), 16) % 2 == 0}
    tr = tfg48.train(OUT, pairs, seed=1, name="P2.pt", train_ids=even)
    val = tfg48.validate(OUT, pairs, name="P2.pt", report="validation_P2.json")
    C.jdump(dict(train=tr, validation=val, n_train_scenes=len(even)), A / "p2.json")
    C.log(f"P2: {len(even)} train scenes, best val L1 {tr['best_val_l1']:.3f}; gates blur {val['gate_blur']} "
          f"contrast {val['gate_contrast']}")
    C.finish_stage("audit_p2", ["audit/p2.json", "tfg/P2.pt"])


# ================================================================ C: traced TFG
def trace(ctx):
    sd, P = ctx.sd, ctx.P
    m0 = C.jload(OUT / "smoke.json")["tfg_m0"]
    traces = []
    for sched, m, only in TRACE_CFGS:
        steps = C.SCHEDULES[sched]
        keep = set(steps) if sched == "late" else set(steps[::5]) | set(steps[-3:])
        tag = f"tfg_m{m:g}_{sched}_it2"
        for cid in (only or D.eligible_cases("dev")):
            _, prompt, _, seed = D.cases("dev")[cid]
            S = D.scorer(ctx, "dev", cid)
            g = tfg48.make_tfg(sd, P, m * m0, iter_steps=2)
            g.guider.set_mask(S.R["background"])
            st = dict(inputs=[], grads=[], mu=None, rho=None)
            rec = []
            o_tilde, o_mu, o_rho, o_step = g.tilde_get_guidance, g.get_mu, g.get_rho, g.guide_step

            def tilde(x0, mc_eps, return_logp=False, **kw):
                out = o_tilde(x0, mc_eps, return_logp=return_logp, **kw)
                if not return_logp:
                    st["inputs"].append(x0.detach().clone())
                    st["grads"].append(out.detach().clone())
                return out

            def get_mu(*a):
                v = o_mu(*a)
                st["mu"] = float(v)
                return v

            def get_rho(*a):
                v = o_rho(*a)
                st["rho"] = float(v)
                return v

            def guide_step(x, t, *a, **kw):
                st.update(inputs=[], grads=[], mu=None, rho=None)
                out = o_step(x, t, *a, **kw)
                if st["inputs"]:
                    pre = st["inputs"][0]
                    post = st["inputs"][-1] + st["mu"] * st["grads"][-1]
                    r = dict(k=int(t), mu=st["mu"], rho=st["rho"], grad_norms=[float(q.norm()) for q in st["grads"]],
                             finite=bool(all(torch.isfinite(q).all() for q in st["grads"])),
                             update_norm=float((post - pre).norm()), x0_norm=float(pre.norm()),
                             P_pre=float(g.guider.reading(pre)[0]), P_post=float(g.guider.reading(post)[0]))
                    if int(t) in keep:
                        for nm, z in (("pre", pre), ("post", post)):
                            save(sd.to_uint8(z.to(sd.dt)), A / "trace" / tag / cid / f"k{int(t):02d}_{nm}.png")
                        r["saved"] = True
                    if sched == "late" and m == 1.0 and int(t) in (steps[0], steps[len(steps) // 2], steps[-1]):
                        r["fd"] = fd_check(g.guider, pre, seed * 100 + int(t))
                    rec.append(r)
                return out

            g.tilde_get_guidance, g.get_mu, g.get_rho, g.guide_step = tilde, get_mu, get_rho, guide_step
            a, info = sd.sample_tfg(prompt, seed, g, set(steps))
            save(a, A / "trace" / tag / cid / "final.png")
            saved = OUT / "images" / "dev" / tag / f"{cid}.png"
            traces.append(dict(cfg=tag, case=cid, steps=rec, final_identical_to_round48=bool(
                saved.exists() and np.array_equal(a, load(saved))),
                final_vs_round48=pstats(a, load(saved)) if saved.exists() else None))
            C.log(f"trace {tag} {cid}: P {rec[0]['P_pre']:.2f} -> {rec[-1]['P_post']:.2f}; identical to round 48: "
                  f"{traces[-1]['final_identical_to_round48']}")
            C.jdump(traces, A / "trace.json")
    C.finish_stage("audit_trace", ["audit/trace.json"])


def fd_check(guider, x, seed):
    """Central finite difference of logp along a random unit direction vs autograd, fp32, two step sizes."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(x.shape, generator=g).to(x.device)
    v = v / v.norm()
    x = x.detach().float()
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        grad = guider.get_guidance(xg, return_logp=False)
    ana = float((grad * v).sum())
    out = dict(analytic=ana)
    for eps in (1e-1, 1e-2):
        with torch.no_grad():
            lp = float(guider.get_guidance(x + eps * v, return_logp=True)[0])
            lm = float(guider.get_guidance(x - eps * v, return_logp=True)[0])
        fd = (lp - lm) / (2 * eps)
        out[f"fd_{eps:g}"] = fd
        out[f"relerr_{eps:g}"] = abs(fd - ana) / max(abs(ana), 1e-12)
    out["relerr"] = min(out["relerr_0.1"], out["relerr_0.01"])
    return out


# ================================================================ analysis
def reading(P, a, bg):
    x = torch.from_numpy(np.asarray(a, np.float32) / 255).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        rm = tfg48.predict(P, x)[0]
    m = F.avg_pool2d(torch.from_numpy(bg.astype(np.float32))[None, None], 8)[0, 0] > 0.5
    m = m.cuda()
    return float(rm[m].mean()) if m.any() else float(rm.mean())


def hf_frac(a, b):
    g = Mx.gray(np.asarray(b, np.float64)) - Mx.gray(np.asarray(a, np.float64))
    Fp = np.abs(np.fft.fft2(g)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(g.shape[0]), np.fft.fftfreq(g.shape[1]), indexing="ij")
    f = np.sqrt(fx ** 2 + fy ** 2)
    return float(Fp[f > 0.25].sum() / max(Fp.sum(), 1e-12))


def diskfit(pre, post, tiles):
    if len(tiles) < 2:
        return float("nan")
    return Mx.blur40.TileStack(np.asarray(pre, np.float64), tiles).measure(np.asarray(post, np.float64))["r"]


def slope_rho(rows, key):
    from scipy.stats import spearmanr
    x = np.array([r["r_true"] for r in rows], float)
    y = np.array([r[key] - r[key + "_0"] for r in rows], float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return dict(slope=float("nan"), spearman=float("nan"), n=int(ok.sum()))
    return dict(slope=float((x[ok] @ y[ok]) / (x[ok] @ x[ok])), spearman=float(spearmanr(x[ok], y[ok])[0]),
                n=int(ok.sum()))


def boot(x, n=4000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return [float("nan")] * 3
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def sign_test(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def analyze(ctx):
    P, P2 = ctx.P, tfg48.load_P(OUT, name="P2.pt")
    traces = C.jload(A / "trace.json")
    rep = dict()
    # ---- per-step pre/post independent measures
    step_rows = []
    for t in traces:
        S = D.scorer(ctx, "dev", t["case"])
        bg, tiles = S.R["background"], S.R["tiles"]
        for r in t["steps"]:
            if not r.get("saved"):
                continue
            d = A / "trace" / t["cfg"] / t["case"]
            pre, post = load(d / f"k{r['k']:02d}_pre.png"), load(d / f"k{r['k']:02d}_post.png")
            step_rows.append(dict(cfg=t["cfg"], case=t["case"], k=r["k"], P_gain=r["P_post"] - r["P_pre"],
                                  P2_gain=reading(P2, post, bg) - reading(P2, pre, bg),
                                  P_gain_png=reading(P, post, bg) - reading(P, pre, bg),
                                  r_ind=diskfit(pre, post, tiles), dog=Mx.dog_ratio(pre, post, bg),
                                  hf=hf_frac(pre, post),
                                  hf_known_r1=hf_frac(pre, Mx.posthoc(pre.astype(float), S.R["subject"], 1.0))))
    rep["steps"] = step_rows
    # ---- finals vs reference (cumulative effect of guidance)
    finals = []
    for t in traces:
        S = D.scorer(ctx, "dev", t["case"])
        bg = S.R["background"]
        ref = S.ref
        fa = load(A / "trace" / t["cfg"] / t["case"] / "final.png")
        m = Mx.judge(S.score(fa.astype(np.float64)), D.rules())
        finals.append(dict(cfg=t["cfg"], case=t["case"], identical_to_round48=t["final_identical_to_round48"],
                           P_ref=reading(P, ref, bg), P_fin=reading(P, fa, bg), P2_ref=reading(P2, ref, bg),
                           P2_fin=reading(P2, fa, bg), r=m["r"], measurable=m["measurable"], dog=m["dog_bg"],
                           scene=m["scene"], preserved=m["preserved"], response=m["response"],
                           fd=[s["fd"] for s in t["steps"] if "fd" in s],
                           all_finite=all(s["finite"] for s in t["steps"]),
                           grad_norm_med=float(np.median([np.mean(s["grad_norms"]) for s in t["steps"]])),
                           update_rel_med=float(np.median([s["update_norm"] / s["x0_norm"] for s in t["steps"]]))))
    rep["finals"] = finals
    # ---- known-blur controls (offline accuracy): dev references and last pre estimate of late-m1 traces
    kb = []
    for src in ("reference", "trajectory"):
        for cid in D.eligible_cases("dev"):
            S = D.scorer(ctx, "dev", cid)
            if src == "reference":
                img = S.ref
            else:
                d = A / "trace" / "tfg_m1_late_it2" / cid
                ks = sorted(int(p.name[1:3]) for p in d.glob("k*_pre.png"))
                img = load(d / f"k{ks[-1]:02d}_pre.png").astype(np.float64)
            bg = S.R["background"]
            p0, q0 = reading(P, img, bg), reading(P2, img, bg)
            for rt in (1, 2, 3, 4, 6):
                b = np.clip(Mx.posthoc(img, S.R["subject"], rt), 0, 255).round()
                kb.append(dict(src=src, case=cid, r_true=rt, P=reading(P, b, bg), P_0=p0, P2=reading(P2, b, bg),
                               P2_0=q0, diskfit=diskfit(img, b, S.R["tiles"]), diskfit_0=0.0))
    rep["known_blur"] = kb
    acc = {src: {k: slope_rho([r for r in kb if r["src"] == src], k) for k in ("P", "P2", "diskfit")}
           for src in ("reference", "trajectory")}
    rep["offline_accuracy"] = acc
    # ---- classification E1-E4 (README)
    late = [f for f in finals if f["cfg"] in ("tfg_m1_late_it2", "tfg_m2_late_it2")]
    e1c = {}
    for cfg in ("tfg_m1_late_it2", "tfg_m2_late_it2"):
        ff = [f for f in late if f["cfg"] == cfg]
        hits = [f for f in ff if (f["P_fin"] - f["P_ref"]) >= 1.0 and f["dog"] >= 0.9 and
                ((not f["measurable"]) or f["r"] <= 0.5 * (f["P_fin"] - f["P_ref"]))]
        e1c[cfg] = dict(n=len(ff), hits=[f["case"] for f in hits])
    E1 = any(len(v["hits"]) >= 6 for v in e1c.values())
    E2 = all(0.7 <= acc[s]["P"]["slope"] <= 1.3 and acc[s]["P"]["spearman"] >= 0.9 for s in acc)
    e1_cases = [f for f in late if f["case"] in e1c[f["cfg"]]["hits"]]
    ratio = [(f["P2_fin"] - f["P2_ref"]) / (f["P_fin"] - f["P_ref"]) for f in e1_cases]
    E3 = bool(e1_cases) and float(np.median(ratio)) <= 0.5
    fds = [x["relerr"] for f in finals for x in f["fd"]]
    E4 = all(f["all_finite"] for f in finals) and bool(fds) and float(np.median(fds)) <= 0.10
    if not E4:
        rec = "guidance bug suspected: fix before any interpretation"
    elif not E2:
        rec = "P inaccurate on SD images (offline accuracy problem): no ensemble experiment"
    elif not E1:
        rec = "late TFG barely moves P: BokehDiff improvements are the more productive direction"
    elif E3:
        rec = "predictor exploitation: a small late-only TFG ensemble experiment is justified (proposed, not launched)"
    else:
        rec = ("score gain without measurable blur is not specific to P (P2 also gains): BokehDiff improvements are "
               "the more productive direction")
    rep["classification"] = dict(E1=E1, E1_detail=e1c, E2=E2, E3=E3, E3_ratio_median=float(np.median(ratio))
                                 if ratio else None, E3_ratios=ratio, E4=E4, fd_relerr_median=float(np.median(fds))
                                 if fds else None, recommendation=rec)
    # ---- B: shared-eligible comparison from round 48 records
    recs = {}
    for line in (OUT / "records.jsonl").open():
        r = json.loads(line)
        recs[r["key"]] = r
    bok = {r["case"] for r in recs.values() if r["split"] == "dev" and r["method"] == "bokeh"}
    shared = [c for c in D.eligible_cases("dev") if c in bok]
    keys = sorted({r["cfg_key"] for r in recs.values() if r["split"] == "dev" and "probe" not in r["cfg_key"]})
    table = []
    for k in keys:
        rr = {c: recs.get(f"dev|{c}|{k}") for c in shared}
        if not all(rr.values()):
            continue
        both = {c: r["response"] and r["preserved"] for c, r in rr.items()}
        bb = {c: recs[f"dev|{c}|bokeh"]["response"] and recs[f"dev|{c}|bokeh"]["preserved"] for c in shared}
        n10 = sum(both[c] and not bb[c] for c in shared)
        n01 = sum(bb[c] and not both[c] for c in shared)
        table.append(dict(arm=k, n=len(shared), response=sum(r["response"] for r in rr.values()),
                          on_target=sum(r["on_target"] for r in rr.values()),
                          preserved=sum(r["preserved"] for r in rr.values()), both=sum(both.values()),
                          vs_bokeh_method_only=n10, vs_bokeh_bokeh_only=n01, sign_test_p=sign_test(n10, n01),
                          scene_diff_vs_bokeh=boot([rr[c]["scene"] - recs[f"dev|{c}|bokeh"]["scene"] for c in shared]),
                          median_scene=float(np.median([r["scene"] for r in rr.values()]))))
    rep["shared"] = dict(cases=shared, table=table)
    C.jdump(rep, A / "audit.json")
    write_report(rep)
    C.finish_stage("audit_analyze", ["audit/audit.json", "audit/report.md"])


def write_report(rep):
    nm = C.jload(A / "numerics.json")
    L = ["# Audit of round 48 (generated by audit48.py)", "", "## A. Numerics under identical execution conditions",
         "", "Per-pixel |Δ| on 0-255, pooled over the 10 eligible dev cases (median of per-case values; max = worst "
         "case). Metric deltas: median over cases.", "",
         "| pair | bit-identical | mean | p50 | p90 | p99 | p99.9 | max | frac ≠ | Δ scene | Δ r | Δ DoG | Δ DINOv2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pair in sorted({r["pair"] for r in nm["rows"]}):
        rr = [r for r in nm["rows"] if r["pair"] == pair]
        med = lambda k: float(np.median([r[k] for r in rr]))
        md = lambda k: (lambda v: f"{np.median(v):+.4f}" if v else "–")(
            [r["metric_delta"][k] for r in rr if r["metric_delta"] and r["metric_delta"][k] is not None])
        L.append(f"| {pair} | {sum(r['identical'] for r in rr)}/{len(rr)} | {med('mean'):.3f} | {med('p50'):.0f} | "
                 f"{med('p90'):.0f} | {med('p99'):.0f} | {med('p999'):.0f} | {max(r['max'] for r in rr):.0f} | "
                 f"{med('frac_nonzero'):.3f} | {md('scene')} | {md('r')} | {md('dog_bg')} | {md('dino_subj')} |")
    L += ["", "NA-RFM steered outputs, round 48 mixed batch shapes vs same-batch reference (counts over 10 cases):", "",
          "| config | mixed rerun = saved | mixed: respond / preserved / both | same: respond / preserved / both | "
          "median |Δ| pixel mean | median scene mixed / same |", "|---|---|---|---|---|---|"]
    for cfg in sorted({r["cfg"] for r in nm["batchshape"]}):
        rr = [r for r in nm["batchshape"] if r["cfg"] == cfg]
        cnt = lambda mode: (f"{sum(r[mode + '_response'] for r in rr)} / {sum(r[mode + '_preserved'] for r in rr)} / "
                            f"{sum(r[mode + '_response'] and r[mode + '_preserved'] for r in rr)}")
        L.append(f"| {cfg} | {sum(r['mixed_rerun_identical_to_saved'] for r in rr)}/{len(rr)} | {cnt('mixed')} | "
                 f"{cnt('same')} | {np.median([r['pix']['mean'] for r in rr]):.2f} | "
                 f"{np.median([r['mixed_scene'] for r in rr]):.3f} / {np.median([r['same_scene'] for r in rr]):.3f} |")
    L += ["", "## B. Method comparison on shared eligible scenes", "", f"Cases: {rep['shared']['cases']}", "",
          "| arm | respond | on target | preserved | respond ∧ preserved | only arm / only BokehDiff | sign test p | "
          "scene change − BokehDiff (mean [95%]) |", "|---|---|---|---|---|---|---|---|"]
    for t in sorted(rep["shared"]["table"], key=lambda t: -t["both"]):
        s = t["scene_diff_vs_bokeh"]
        L.append(f"| {t['arm']} | {t['response']} | {t['on_target']} | {t['preserved']} | {t['both']} | "
                 f"{t['vs_bokeh_method_only']} / {t['vs_bokeh_bokeh_only']} | {t['sign_test_p']:.3f} | "
                 f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}] |")
    acc = rep["offline_accuracy"]
    L += ["", "## C. TFG: offline prediction accuracy vs intervention effectiveness", "",
          "**Offline accuracy** (known background disk blur r ∈ {1, 2, 3, 4, 6} added to images, reading minus the "
          "unblurred reading; slope 1 = reads true radius):", "",
          "| images | P slope / Spearman | P2 slope / Spearman | disk fit slope / Spearman |", "|---|---|---|---|"]
    for s, v in acc.items():
        L.append(f"| {s} | {v['P']['slope']:.2f} / {v['P']['spearman']:.2f} | {v['P2']['slope']:.2f} / "
                 f"{v['P2']['spearman']:.2f} | {v['diskfit']['slope']:.2f} / {v['diskfit']['spearman']:.2f} |")
    L += ["", "**Intervention effectiveness** (finals vs the unsteered reference; medians over cases):", "",
          "| config | n | identical to round 48 | P gain | P2 gain | disk-fit r (measurable n) | DoG ratio | "
          "scene | update ‖Δx0‖/‖x0‖ | grad norm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in sorted({f["cfg"] for f in rep["finals"]}):
        ff = [f for f in rep["finals"] if f["cfg"] == cfg]
        meas = [f["r"] for f in ff if f["measurable"]]
        L.append(f"| {cfg} | {len(ff)} | {sum(f['identical_to_round48'] for f in ff)} | "
                 f"{np.median([f['P_fin'] - f['P_ref'] for f in ff]):+.2f} | "
                 f"{np.median([f['P2_fin'] - f['P2_ref'] for f in ff]):+.2f} | "
                 f"{(np.median(meas) if meas else float('nan')):.2f} ({len(meas)}) | "
                 f"{np.median([f['dog'] for f in ff]):.2f} | {np.median([f['scene'] for f in ff]):.3f} | "
                 f"{np.median([f['update_rel_med'] for f in ff]):.4f} | {np.median([f['grad_norm_med'] for f in ff]):.3g} |")
    L += ["", "**Per guided step** (pre → post x0 estimate at the same step; aligned by construction; medians):", "",
          "| config | steps | P gain / step | P2 gain / step | disk-fit r / step (measurable) | DoG ratio | "
          "update energy > 0.25 cyc/px | same for a known 1 px blur |", "|---|---|---|---|---|---|---|---|"]
    for cfg in sorted({s["cfg"] for s in rep["steps"]}):
        ss = [s for s in rep["steps"] if s["cfg"] == cfg]
        rr = [s["r_ind"] for s in ss if np.isfinite(s["r_ind"])]
        L.append(f"| {cfg} | {len(ss)} | {np.median([s['P_gain'] for s in ss]):+.3f} | "
                 f"{np.median([s['P2_gain'] for s in ss]):+.3f} | {(np.median(rr) if rr else float('nan')):.2f} "
                 f"({len(rr)}) | {np.median([s['dog'] for s in ss]):.3f} | {np.median([s['hf'] for s in ss]):.2f} | "
                 f"{np.median([s['hf_known_r1'] for s in ss]):.2f} |")
    c = rep["classification"]
    L += ["", "## Classification (README rule)", "",
          f"- E1 score gain without blur: **{c['E1']}** {json.dumps(c['E1_detail'])}",
          f"- E2 P accurate offline in this domain: **{c['E2']}**",
          f"- E3 specific to P (P2 reads ≤ 50% of P's gain): **{c['E3']}** (median ratio {c['E3_ratio_median']})",
          f"- E4 guidance numerically sound: **{c['E4']}** (finite-difference relative error median "
          f"{c['fd_relerr_median']})", "", f"**Recommendation: {c['recommendation']}.**"]
    (A / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


STAGES = dict(numerics=numerics, p2=p2, trace=trace, analyze=analyze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    a = ap.parse_args()
    A.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = D.Ctx()
    for st in a.stages:
        if C.stage_done(f"audit_{st}"):
            C.log(f"audit {st}: done, skipping")
            continue
        C.log(f"audit stage {st}")
        try:
            STAGES[st](ctx)
        except Exception as e:
            import traceback
            C.fail_stage(f"audit_{st}", f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            raise
    C.log("audit done")


if __name__ == "__main__":
    main()
