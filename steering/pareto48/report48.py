"""Pareto pilot report: implementation checks, development findings, held-out findings, tables, Pareto plots, matched
comparison grids and a manifest (splits, checkpoints, seeds, configurations). Verdict lines follow README rule 7.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

import common48 as C
import prompts48 as PR

OUT = C.OUT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def boot(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def f(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def row_summary(rr):
    med = lambda k: float(np.nanmedian([r[k] for r in rr])) if rr else float("nan")
    meas = [r for r in rr if r["measurable"]]
    return dict(n=len(rr), measurable=len(meas), response=sum(r["response"] for r in rr),
                on_target=sum(r["on_target"] for r in rr), preserved=sum(r["preserved"] for r in rr),
                both=sum(r["response"] and r["preserved"] for r in rr),
                r=float(np.median([r["r"] for r in meas])) if meas else float("nan"), dog=med("dog_bg"),
                scene=med("scene"), dino=med("dino_subj"), iou=med("iou"), shift=med("shift"),
                sharp=med("subj_sharp"), de=med("de"), depth=med("depth_rho"), wall=med("wall_s"),
                fwd=med("unet_fwd") if rr and "unet_fwd" in rr[0] else float("nan"),
                bwd=med("unet_bwd") if rr and "unet_bwd" in rr[0] else float("nan"))


HEAD = ("| arm | n | blur measurable | blur response | on target (±1.33 px of 3) | median added blur r (px) | "
        "bg DoG ratio | scene change | subject DINOv2 | mask IoU | centre shift (px) | subject sharpness (log) | "
        "ΔE | depth ρ | preserved | response ∧ preserved | s / image | UNet fwd | UNet bwd |")
SEP = "|" + "---|" * 19


def line(name, s):
    return (f"| {name} | {s['n']} | {s['measurable']} | {s['response']} | {s['on_target']} | {f(s['r'])} | "
            f"{f(s['dog'])} | {f(s['scene'], 3)} | {f(s['dino'], 3)} | {f(s['iou'])} | {f(s['shift'], 1)} | "
            f"{f(s['sharp'])} | {f(s['de'])} | {f(s['depth'])} | {s['preserved']} | {s['both']} | {f(s['wall'], 1)} | "
            f"{f(s['fwd'], 0)} | {f(s['bwd'], 0)} |")


def pareto_plot(ctx, split, path, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = dict(narfm="#2a6fdb", tfg="#d1495b", random="#8d99ae", bokeh="#2e933c", posthoc="#111111",
                  base="#aaaaaa")
    for ck, label in keys:
        rr = [r for r in ctx.records.values() if r["split"] == split and r["cfg_key"] == ck]
        if not rr:
            continue
        m = rr[0]["method"]
        meas = [r for r in rr if r["measurable"]]
        x = np.median([r["target_err"] for r in meas]) if meas else np.nan
        y = np.median([r["scene"] for r in rr])
        sty = dict(color=colors.get(m, "k"), s=70 if m in ("bokeh", "posthoc") else 30,
                   marker="s" if m in ("bokeh", "posthoc") else "o")
        if np.isfinite(x):
            axs[0].scatter(x, y, **sty)
            axs[0].annotate(label, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
        axs[1].scatter(np.median([r["dog_bg"] for r in rr]), y, **sty)
        axs[1].annotate(label, (np.median([r["dog_bg"] for r in rr]), y), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
    axs[0].set_xlabel("median |delivered blur − 3 px| (disk fit; measurable outputs only)")
    axs[0].set_ylabel("median scene change\n(1 − DINOv2 patch cosine vs blur-matched reference)")
    axs[1].set_ylabel("median scene change")
    axs[1].set_xlabel("median background DoG band-energy ratio (alignment-free; 1 = no change)")
    axs[1].invert_xaxis()
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title(f"{split}: blur-target error vs scene change")
    axs[1].set_title(f"{split}: background detail loss vs scene change (all outputs)")
    fig.tight_layout()
    fig.subplots_adjust(left=0.08)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def grid(ctx, split, cols, path, cids, w=200):
    font_h = 28
    sheet = Image.new("RGB", (w * len(cols), (w + 4) * len(cids) + font_h), "white")
    d = ImageDraw.Draw(sheet)
    for j, (ck, label) in enumerate(cols):
        d.text((j * w + 4, 4), label, fill="black")
    for i, cid in enumerate(cids):
        for j, (ck, label) in enumerate(cols):
            p = OUT / "refs" / split / f"{cid}.png" if ck == "ref" else \
                OUT / "images" / split / ck.replace("|", "_") / f"{cid}.png"
            y = font_h + i * (w + 4)
            if p.exists():
                sheet.paste(Image.open(p).convert("RGB").resize((w, w), Image.LANCZOS), (j * w, y))
                r = ctx.records.get(f"{split}|{cid}|{ck}")
                if r:
                    tag = (f"r {r['r']:.1f}" if r["measurable"] else "r n/a") + f"  sc {r['scene']:.2f}" + \
                          ("  P" if r["preserved"] else "")
                    d.rectangle([j * w, y + w - 14, j * w + w, y + w], fill="white")
                    d.text((j * w + 3, y + w - 13), tag, fill="black")
            else:
                d.text((j * w + 4, y + w // 2), "not run", fill="gray")
    sheet.save(path, quality=90)


def matched(ctx, split, a_key, b_key):
    """Paired comparison where both are on target or within 1 px of each other (disk fit)."""
    rows, unmatched = [], []
    for cid, *_ in PR.cases(split):
        a, b = ctx.records.get(f"{split}|{cid}|{a_key}"), ctx.records.get(f"{split}|{cid}|{b_key}")
        if not a or not b:
            continue
        if a["measurable"] and b["measurable"] and ((a["on_target"] and b["on_target"]) or abs(a["r"] - b["r"]) <= 1):
            rows.append((cid, a, b))
        else:
            unmatched.append(dict(case=cid, a_r=a["r"], b_r=b["r"]))
    d = {k: boot([a[k] - b[k] for _, a, b in rows]) for k in ("scene", "dino_subj", "subj_sharp", "iou", "de")}
    return dict(n=len(rows), unmatched=unmatched, diff=d)


def main(ctx):
    L = ["# Pareto pilot report (generated by report48.py)", ""]
    rules = C.jload(OUT / "rules48.json", {})
    smoke = C.jload(OUT / "smoke.json", {})
    nr = C.jload(OUT / "narfm" / "report.json", {})
    tr = C.jload(OUT / "tfg" / "report.json", {})
    sel = C.jload(OUT / "selection48.json", {})
    gate = C.jload(OUT / "gate48.json", {})
    pairs = C.jload(OUT / "camedit" / "pairs.json", {})
    man = {s: C.jload(C.manifest_path(s)) for s in ("s0_env", "s1_data", "s1_refs", "s1_validate", "s2_narfm",
                                                    "s2_tfg", "s3_smoke", "s4a_layers", "s4b_sweep", "s4c_select",
                                                    "bscore_dev", "s6_gate", "s6_confirm", "bscore_confirm")}
    L += ["## 1. Implementation checks", "", "Stage status:", ""]
    L += [f"- `{k}`: {v.get('status') if v else 'not run'}" + (f" ({v.get('reason') or v.get('skipped')})"
                                                                if v and (v.get('reason') or v.get('skipped')) else "")
          for k, v in man.items()]
    if man["s0_env"]:
        e = man["s0_env"]
        L += ["", f"Environment: {e.get('gpu')} ({e.get('gpu_gb', 0):.0f} GB), torch {e.get('torch')}, diffusers "
                  f"{e.get('diffusers')}, transformers {e.get('transformers')}; NA-RFM {e.get('narfm', '')[:8]}, TFG "
                  f"{e.get('tfg', '')[:8]}."]
    if pairs:
        L += ["", f"CamEdit pairs: {json.dumps(pairs.get('summary', {}))}"]
    if rules:
        L += ["", f"Measurement validation: {json.dumps(rules.get('checks'))}; disk-fit within tolerance "
                  f"{json.dumps(rules.get('diskfit_within'))}; threshold changes {json.dumps(rules.get('changes'))}; "
                  f"blur numbers exploratory: {rules.get('exploratory_blur')}."]
    if nr:
        L += ["", "NA-RFM directions (scene-split):", "", "| layer | D | val AUC raw / norm | test AUC raw | test paired "
              "acc | random val AUC (min, median, max) | cos with mean difference | eligible |", "|---|---|---|---|---|"
              "---|---|---|"]
        for k, v in nr.get("layers", {}).items():
            L.append(f"| {k} | {v['D']} | {v['val_auc_raw']:.3f} / {v['val_auc_norm']:.3f} | {v['test_auc_raw']:.3f} "
                     f"| {v['test_paired_acc_raw']:.2f} | {v['val_random_auc']} | {v['cos_with_meandiff']:+.3f} | "
                     f"{v['eligible']} |")
        L += ["", f"Noise-alignment filters: {json.dumps(nr.get('na'))}"]
    if tr:
        v = tr.get("validation", {})
        L += ["", f"TFG predictor: {tr.get('train', {}).get('n_params')} parameters, best val L1 "
                  f"{f(tr.get('train', {}).get('best_val_l1'))} px.",
              f"Validation (test scenes): blur Spearman {f(v.get('blur_spearman'))}, MAE r ≤ 4 {f(v.get('blur_mae_le4'))}"
              f" px, contrast Δ {f(v.get('contrast_delta'))} px (r = 4 response {f(v.get('resp_r4'))}), median-filter Δ "
              f"{f(v.get('median_delta'))} (r = 2 response {f(v.get('resp_r2'))}), category spread "
              f"{f(v.get('category_spread'))} px, real CamEdit pairs Spearman {f(v.get('real_pairs_spearman'))} (n "
              f"{v.get('real_pairs_n')}, MAE {f(v.get('real_pairs_mae'))} px). Gates: blur {v.get('gate_blur')}, "
              f"contrast {v.get('gate_contrast')}."]
    if smoke:
        L += ["", f"Smoke checks: {json.dumps(smoke.get('checks'))}", "",
              f"Smoke-set strengths: NA-RFM γ0 {smoke.get('narfm_gamma0')} (NA/RFM norm ratio "
              f"{f(smoke.get('narfm_na_ratio'))}), grids {smoke.get('narfm_grids')}; TFG m0 {smoke.get('tfg_m0')} "
              f"(weak: {smoke.get('tfg_weak')}), plan {smoke.get('tfg_plan')}; reductions {smoke.get('reductions')}; "
              f"dtypes {smoke.get('dtypes')}; memory {smoke.get('memory')}."]
    # ---------------- development
    L += ["", "## 2. Development findings (12 dev prompt–seed pairs)", ""]
    if man["s4a_layers"] and man["s4a_layers"].get("scores"):
        L += [f"NA-RFM layer screen (dev 1-4, m = 1, middle): {json.dumps(man['s4a_layers']['scores'])} → "
              f"**{man['s4a_layers']['layer']}**.", ""]
    seen = sorted({(r["method"], r["cfg_key"]) for r in ctx.records.values() if r["split"] == "dev"
                   and r["method"] in ("narfm", "tfg", "random") and r.get("phase") in ("sweep", "layers")})
    keys = [(k, k.replace("|", " ")) for _, k in seen]
    keys += [("bokeh", "SD + BokehDiff (round 45 controller)"), ("posthoc", "post-hoc disk blur at 3 px")]
    L += ["Every configuration (medians over cases; counts out of n):", "", HEAD, SEP]
    for ck, label in keys:
        rr = [r for r in ctx.records.values() if r["split"] == "dev" and r["cfg_key"] == ck]
        if rr:
            L.append(line(label, row_summary(rr)))
    pareto_plot(ctx, "dev", OUT / "pareto_dev.png", keys)
    L += ["", "![dev Pareto](pareto_dev.png)", ""]
    for method in ("narfm", "tfg"):
        s = sel.get(method)
        if s:
            L.append(f"- **{method}** selection: {s.get('status')}" + (
                f"; P = `{s['P']['key']}`, B = `{s['B']['key']}`" if s.get("status") == "selected" else ""))
    L += ["", f"Confirmation gate: {json.dumps(gate)}", ""]
    bd = C.jload(OUT / "bokeh" / "dev" / "run" / "bd48.json")
    if bd:
        L += [f"BokehDiff (dev): {sum(r['rendered'] for r in bd['rows'])} rendered; ineligible (< 2 behind tiles, no "
              f"gap): {[r['id'] for r in bd['rows'] if not r['rendered']]}; K: "
              f"{ {r['id']: round(r['K'], 1) for r in bd['rows'] if r['rendered']} }; clipped "
              f"{[r['id'] for r in bd['rows'] if r.get('clipped')]}; flags "
              f"{ {r['id']: r['flag'] for r in bd['rows'] if r.get('flag') not in (None, 'ok')} }.", ""]
    cmp_keys = []
    for method in ("narfm", "tfg"):
        s = sel.get(method, {})
        if s.get("status") == "selected":
            cmp_keys += [(s["P"]["key"], f"{method} P")] + ([(s["B"]["key"], f"{method} B")]
                                                             if s["B"]["key"] != s["P"]["key"] else [])
    L += ["Matched-blur comparisons against SD + BokehDiff (dev; paired differences, method − BokehDiff, mean [95% "
          "bootstrap]):", ""]
    for ck, label in cmp_keys:
        mm = matched(ctx, "dev", ck, "bokeh")
        L.append(f"- {label}: n matched {mm['n']}; scene {mm['diff']['scene']}; subject DINOv2 "
                 f"{mm['diff']['dino_subj']}; subject sharpness {mm['diff']['subj_sharp']}; unmatched "
                 f"{len(mm['unmatched'])}")
    cids = [c[0] for c in PR.cases("dev")]
    for k in range(0, len(cids), 6):
        grid(ctx, "dev", [("ref", "unsteered")] + cmp_keys + [("bokeh", "BokehDiff"), ("posthoc", "post-hoc 3px")],
             OUT / f"grid_dev_{k // 6}.jpg", cids[k:k + 6])
    L += ["", "Grids: `grid_dev_0.jpg`, `grid_dev_1.jpg` (label: measured added blur r, scene change, P = preserved)."]
    # ---------------- confirmation
    L += ["", "## 3. Held-out confirmation (12 confirm prompts)", ""]
    if not man["s6_confirm"] or man["s6_confirm"].get("skipped") or man["s6_confirm"].get("status") != "ok":
        L += [f"Not run: {(man['s6_confirm'] or {}).get('skipped') or (man['s6_confirm'] or {}).get('reason') or 'stage not reached'}."]
    else:
        L += [HEAD, SEP]
        for ck, label in [("bokeh", "SD + BokehDiff"), ("posthoc", "post-hoc 3 px")] + cmp_keys:
            rr = [r for r in ctx.records.values() if r["split"] == "confirm" and r["cfg_key"] == ck]
            if rr:
                L.append(line(label, row_summary(rr)))
        pareto_plot(ctx, "confirm", OUT / "pareto_confirm.png", cmp_keys + [("bokeh", "BokehDiff"),
                                                                            ("posthoc", "post-hoc")])
        L += ["", "![confirm Pareto](pareto_confirm.png)", ""]
        for ck, label in cmp_keys:
            mm = matched(ctx, "confirm", ck, "bokeh")
            L.append(f"- {label} vs BokehDiff: n matched {mm['n']}; scene {mm['diff']['scene']}; subject DINOv2 "
                     f"{mm['diff']['dino_subj']}; unmatched {mm['unmatched']}")
        cc = [c[0] for c in PR.cases("confirm")]
        for k in range(0, len(cc), 6):
            grid(ctx, "confirm", [("ref", "unsteered")] + cmp_keys + [("bokeh", "BokehDiff"),
                                                                       ("posthoc", "post-hoc 3px")],
                 OUT / f"grid_confirm_{k // 6}.jpg", cc[k:k + 6])
    # ---------------- verdicts (README rule 7)
    L += ["", "## 4. Verdicts (README rule 7)", ""]
    for method in ("narfm", "tfg"):
        s, g = sel.get(method, {}), gate.get(method, {})
        if not valid_stage(method):
            v = "unresolved (implementation controls or component validation failed)"
        elif not g.get("go"):
            v = f"no advantage established (dev: {g.get('why') or 'did not pass the confirmation gate'}; " \
                f"max response {g.get('max_response')}/12, max response ∧ preserved {g.get('max_both')}/12)"
        else:
            conf = [r for r in ctx.records.values() if r["split"] == "confirm" and r["method"] == method]
            best = max((sum(r["response"] and r["preserved"] for r in conf if r["cfg_key"] == k)
                        for k in {r["cfg_key"] for r in conf}), default=0)
            v = "promising" if best >= 8 else f"no advantage established (confirm: best {best}/12 respond ∧ preserved)"
        L.append(f"- **{method}**: {v}")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    # ---------------- manifest
    mf = dict(
        model=C.MODEL, steps=C.STEPS, cfg=C.CFG, size=C.SIZE, scheduler="DDIM (checkpoint config), eta 0",
        target_px=C.T_BLUR, tol_px=C.TOL, schedules={k: [v[0], v[-1]] for k, v in C.SCHEDULES.items()},
        layers=C.LAYERS, prompts=dict(dev=PR.cases("dev"), confirm=PR.cases("confirm")),
        camedit=dict(shard="CamEdit_Focal_001", split_rule="sha1(id) mod 20: 0-13 train, 14-16 val, 17-19 test",
                     ids={s: [p["id"] for p in pairs.get("pairs", []) if p["split"] == s]
                          for s in ("train", "val", "test")}),
        checkpoints={k: sha(OUT / p) for k, p in (("directions", "narfm/directions.pt"),
                                                  ("na_filters", "narfm/na_filters.pt"), ("P", "tfg/P.pt"))},
        selection_sha256=(man["s4c_select"] or {}).get("sha256"), smoke=smoke, rules=rules.get("rules"),
        rule45=sha(C.ROOT / "bokeh" / "controller" / "results" / "rule45.json"))
    C.jdump(mf, OUT / "manifest48.json")
    print("\n".join(L))


def valid_stage(method):
    m = C.jload(C.manifest_path("s3_smoke"))
    return bool(m and m.get(f"{method}_valid"))
