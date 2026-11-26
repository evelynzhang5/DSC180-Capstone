import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
rec_path, out = sys.argv[1], sys.argv[2]
R = {}
for l in open(rec_path):
    r = json.loads(l); R[r["key"]] = r
EL = ["d01","d03","d04","d05","d06","d07","d08","d09","d10","d11"]
keys = sorted({r["cfg_key"] for r in R.values() if r["split"] == "dev" and r["method"] in ("narfm","tfg","random")
               and r.get("phase") in ("sweep","layers") and r["cfg_key"].startswith(("narfm|mid","tfg","random"))})
keys += ["bokeh", "posthoc"]
col = dict(narfm="#2a6fdb", tfg="#d1495b", random="#8d99ae", bokeh="#2e933c", posthoc="#111111")
lab = lambda k: {"bokeh": "SD + BokehDiff", "posthoc": "post-hoc 3 px blur"}.get(
    k, k.replace("narfm|mid|", "NA-RFM ").replace("random|mid|", "random ").replace("tfg|", "TFG ").replace("|it2", "").replace("|", " "))
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
for k in keys:
    rr = [R[f"dev|{c}|{k}"] for c in EL if f"dev|{c}|{k}" in R]
    if not rr: continue
    m = rr[0]["method"]; big = m in ("bokeh", "posthoc")
    sty = dict(color=col[m], s=90 if big else 34, marker="s" if big else "o", zorder=3)
    y = np.median([r["scene"] for r in rr])
    meas = [r["target_err"] for r in rr if r["measurable"]]
    if meas:
        x = np.median(meas); axs[0].scatter(x, y, **sty); axs[0].annotate(lab(k), (x, y), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    x2 = np.median([r["dog_bg"] for r in rr]); axs[1].scatter(x2, y, **sty)
    axs[1].annotate(lab(k), (x2, y), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
axs[0].set_xlabel("blur error: median |delivered blur − 3 px target|\n(disk fit; outputs it could measure)")
axs[1].set_xlabel("background detail kept: median DoG band-energy ratio\n(1 = unchanged, lower = blurrier)")
for a in axs:
    a.set_ylabel("scene change (median)\n1 − DINOv2 similarity to the blur-matched original")
    a.grid(alpha=0.3)
axs[0].set_title("Blur accuracy vs scene change")
axs[1].set_title("Background blur vs scene change")
h = [Line2D([], [], color=col[m], marker="s" if m in ("bokeh","posthoc") else "o", ls="", ms=8, label=l) for m, l in
     (("narfm", "NA-RFM (activation steering)"), ("tfg", "TFG (gradient guidance)"), ("random", "random direction (control)"),
      ("bokeh", "SD + BokehDiff"), ("posthoc", "post-hoc blur of the original"))]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=9)
fig.suptitle("Steering SD 1.5 toward background blur: 10 dev prompts, lower-left is better", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
