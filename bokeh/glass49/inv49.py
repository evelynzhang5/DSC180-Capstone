"""Round 49 stage 0: CPU-only RealBokeh glass inventory (no training, no GPU).

usage:
  python inv49.py screen  <out>   # all 1,440 f/22 sources (train, validation, test): CLIP glass score, embeddings,
                                  # near-duplicate pairs across splits, contact sheets of the top candidates
  python inv49.py align   <out> <ids.json>   # for hand-picked candidates: download every aperture, measure alignment
                                             # of each target against the f/22 source, write sheets per candidate

Independence: round 47 trained its separator on every RealBokeh *test*-split f/22 source except ids 48, 49, 51, 116,
129, 131 (train47.background_paths, a glob; no list was saved), and on round 40's rb_* inputs (also test split). A
train/validation scene is fresh only if it is not a near-duplicate of any test-split scene.
"""
import glob
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")             # transformers otherwise also imports TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from PIL import Image, ImageDraw

REPO = "timseizinger/RealBokeh_3MP"
SPLITS = ("train", "validation", "test")
R47_EXCLUDED = {48, 49, 51, 116, 129, 131}       # test ids round 47 did not train on
GLASS = ["a photo of a glass object", "a photo of a transparent glass bottle", "a photo of a wine glass",
         "a photo of a glass jar", "a photo of a glass vase", "a photo through a glass window",
         "a photo of a transparent plastic or acrylic object", "a photo of a glass display case"]
OTHER = ["a photo of an opaque object", "a photo of a plant", "a photo of a person", "a photo of a building",
         "a photo of food", "a photo of a toy", "a photo of furniture", "a photo of a street"]
DUP_COS = 0.93                                   # CLIP image cosine above which two scenes are reviewed as one


def fetch(patterns):
    """Anonymous download is rate-limited (HTTP 429): few workers, resume from the cache with growing back-off."""
    import time
    from huggingface_hub import snapshot_download
    for k in range(40):
        try:
            return snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns, max_workers=2)
        except Exception as e:                     # 429 surfaces as HfHubHTTPError or LocalEntryNotFoundError
            wait = min(300 * (k + 1), 1800)
            print(f"fetch attempt {k + 1} failed ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("RealBokeh download did not complete after 40 attempts")


def pod_cpus():
    """CPUs this pod may use (cgroup quota), not the node's core count that os.cpu_count() reports."""
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(per)))
    except Exception:
        pass
    return len(os.sched_getaffinity(0))


def load_small(path, side=448):
    """Decode a JPEG at reduced size (draft mode), keeping at least `side` px on the short edge."""
    im = Image.open(path)
    im.draft("RGB", (side * im.size[0] // min(im.size), side * im.size[1] // min(im.size)))
    return im.convert("RGB")


def thumb(path, w=256):
    im = load_small(path, w)
    return im.resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)


def sheet(items, path, cols=8, w=256):
    """items: list of (PIL image, label)."""
    h = max(im.size[1] for im, _ in items) + 18
    S = Image.new("RGB", (cols * w, h * ((len(items) + cols - 1) // cols)), "white")
    d = ImageDraw.Draw(S)
    for k, (im, lab) in enumerate(items):
        x, y = (k % cols) * w, (k // cols) * h
        S.paste(im, (x, y + 18))
        d.text((x + 3, y + 3), lab, fill="black")
    S.save(path, quality=88)


def screen(out):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    n = pod_cpus()
    torch.set_num_threads(n)
    print("threads", n, flush=True)
    root = fetch([f"{s}/metadata/*" for s in SPLITS] + [f"{s}/in/*" for s in SPLITS])
    M = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    P = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        t = M.get_text_features(**P(text=GLASS + OTHER, return_tensors="pt", padding=True))
        t = t / t.norm(dim=-1, keepdim=True)
    os.makedirs(out, exist_ok=True)
    part = f"{out}/screen_partial.jsonl"            # one line per scene; resumed after a pod limit
    done = {}
    if os.path.exists(part):
        for ln in open(part):
            d = json.loads(ln)
            done[(d["split"], d["id"])] = d
    print("resuming with", len(done), "scenes done", flush=True)
    fpart = open(part, "a")
    import time
    t0 = time.time()
    rows, embs = [], []
    for s in SPLITS:
        for mf in sorted(glob.glob(f"{root}/{s}/metadata/*.json"), key=lambda f: int(os.path.basename(f)[:-5])):
            m = json.load(open(mf))
            if (s, m["id"]) in done:
                d = done[(s, m["id"])]
                embs.append(np.array(d.pop("emb"), np.float32))
                rows.append(d)
                continue
            im = load_small(f"{root}/{s}/{m['source_image']}")
            with torch.no_grad():
                e = M.get_image_features(**P(images=im, return_tensors="pt"))
                e = (e / e.norm(dim=-1, keepdim=True))[0]
            pr = torch.softmax(100 * e @ t.T, -1)
            rows.append(dict(split=s, id=m["id"], glass=float(pr[:len(GLASS)].sum()),
                             top=(GLASS + OTHER)[int(pr.argmax())], target_avs=m["target_avs"],
                             focal_length=m["focal_length"], focus_m=m["focus_plane_distance"],
                             focus_unc_m=m["focus_plane_uncertainty"],
                             r47_trained=(s == "test" and m["id"] not in R47_EXCLUDED)))
            embs.append(e.numpy())
            fpart.write(json.dumps(dict(rows[-1], emb=[round(float(v), 5) for v in e])) + "\n")
            fpart.flush()
            if len(rows) % 100 == 0:
                print(f"screened {len(rows)} ({time.time() - t0:.0f}s this session)", flush=True)
    E = np.stack(embs)
    C = E @ E.T
    dups = [dict(a=f"{rows[i]['split']}/{rows[i]['id']}", b=f"{rows[j]['split']}/{rows[j]['id']}", cos=float(C[i, j]))
            for i in range(len(rows)) for j in range(i + 1, len(rows)) if C[i, j] >= DUP_COS]
    os.makedirs(f"{out}/sheets", exist_ok=True)
    np.save(f"{out}/clip_emb.npy", E)
    json.dump(dict(rows=rows, dup_pairs=dups, dup_cos=DUP_COS, root=root), open(f"{out}/screen.json", "w"), indent=1)
    for s, n in (("train", 160), ("validation", 48), ("test", 48)):
        cand = sorted([r for r in rows if r["split"] == s], key=lambda r: -r["glass"])[:n]
        for k in range(0, len(cand), 32):
            sheet([(thumb(f"{root}/{s}/in/{r['id']}_f22.JPG"), f"{s[0]}{r['id']}  g{r['glass']:.2f}")
                   for r in cand[k:k + 32]], f"{out}/sheets/{s}_{k // 32:02d}.jpg")
    print("screen done:", len(rows), "scenes,", len(dups), "near-duplicate pairs", flush=True)


def align(out, ids_json):
    """Alignment of each target aperture to the f/22 source: global translation by phase correlation on a 1024-px
    luminance image, and the normalised cross-correlation of gradient magnitude after that shift (a residual-
    misalignment proxy that is insensitive to defocus of smooth regions)."""
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import sobel, shift as nd_shift
    cand = json.load(open(ids_json))             # [{"split": ..., "id": ...}, ...]
    root = fetch([f"{c['split']}/gt/{c['id']}/*" for c in cand] + [f"{c['split']}/in/{c['id']}_f22.JPG" for c in cand]
                 + [f"{c['split']}/metadata/{c['id']}.json" for c in cand])
    res = []

    def lum(p):
        im = Image.open(p).convert("L")
        im = im.resize((1024, round(im.size[1] * 1024 / im.size[0])), Image.LANCZOS)
        return np.asarray(im, np.float32) / 255

    os.makedirs(f"{out}/cand", exist_ok=True)
    for c in cand:
        m = json.load(open(f"{root}/{c['split']}/metadata/{c['id']}.json"))
        src = lum(f"{root}/{c['split']}/{m['source_image']}")
        gs = np.hypot(sobel(src, 0), sobel(src, 1))
        per = []
        items = [(thumb(f"{root}/{c['split']}/{m['source_image']}", 384), "f/22 source")]
        for tp, av in zip(m["target_images"], m["target_avs"]):
            tg = lum(f"{root}/{c['split']}/{tp}")
            sh, err, _ = phase_cross_correlation(src, tg, upsample_factor=10)
            tg2 = nd_shift(tg, sh, order=1, mode="nearest")
            gt = np.hypot(sobel(tg2, 0), sobel(tg2, 1))
            b = 16
            a_, b_ = gs[b:-b, b:-b].ravel(), gt[b:-b, b:-b].ravel()
            ncc = float(np.corrcoef(a_, b_)[0, 1])
            per.append(dict(av=av, shift_px_1024=[float(sh[0]), float(sh[1])], phase_err=float(err), grad_ncc=ncc))
            items.append((thumb(f"{root}/{c['split']}/{tp}", 384), f"f/{av}  shift {np.hypot(*sh):.1f}px"))
        sheet(items, f"{out}/cand/{c['split']}_{c['id']}.jpg", cols=4, w=384)
        res.append(dict(split=c["split"], id=c["id"], focus_m=m["focus_plane_distance"],
                        focus_unc_m=m["focus_plane_uncertainty"], focal_length=m["focal_length"], targets=per))
        print("aligned", c["split"], c["id"], flush=True)
    json.dump(res, open(f"{out}/align.json", "w"), indent=1)


if __name__ == "__main__":
    {"screen": lambda: screen(sys.argv[2]), "align": lambda: align(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
