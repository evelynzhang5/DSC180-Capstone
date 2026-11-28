"""Full NA-RFM layer sweep (README "Full layer sweep"): learned and random directions at every configuration for the
attention, late and mid layers, plus the unsteered floor on this GPU. Dev prompts only. Resumable via records.jsonl.

usage (repo root, on a GPU pod):  P48_OUT=outputs/pareto48L python steering/pareto48/sweep48b.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import common48 as C   # noqa: E402
import run48 as RN     # noqa: E402

GRID = [0.5, 1.0, 2.0]
LAYERS = ["attn", "late", "mid"]


def main():
    ctx = RN.Ctx()
    cids = list(RN.cases("dev"))
    t0 = time.time()
    for cid in cids:                                   # floor: unsteered on this GPU, scored vs frozen refs
        RN.run_one(ctx, "dev", cid, "narfm", dict(layer="mid", m=0.0, sched="all"), phase="floor")
    C.log(f"floor done ({(time.time() - t0) / 3600:.2f} h)")
    for method in ("narfm", "random"):
        for layer in LAYERS:
            for m in GRID:
                for s in C.SCHEDULES:
                    for cid in cids:
                        RN.run_one(ctx, "dev", cid, method, dict(layer=layer, m=m, sched=s), phase="layers_full")
                    C.log(f"{method} {layer} m{m:g} {s}: done ({(time.time() - t0) / 3600:.2f} h)")
    C.log("##### SWEEP DONE")


if __name__ == "__main__":
    main()
