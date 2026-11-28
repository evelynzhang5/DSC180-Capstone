import json, sys, numpy as np
sys.path.insert(0, "bokeh/glass49"); import inv49
O = "outputs/glass49"
d = json.load(open(f"{O}/screen.json")); R = d["rows"]; E = np.load(f"{O}/clip_emb.npy"); root = d["root"]
P = {"train": "tr", "validation": "va", "test": "te"}
key = [f"{P[r['split']]}{r['id']}" for r in R]; idx = {k: i for i, k in enumerate(key)}
cand = sys.argv[1].split(",")
tr47 = [i for i, r in enumerate(R) if r["r47_trained"]]
items, out = [], []
for c in cand:
    i = idx[c]; s = E[tr47] @ E[i]; o = np.argsort(-s)[:3]
    items.append((inv49.thumb(f"{root}/{R[i]['split']}/in/{R[i]['id']}_f22.JPG"), f"{c} CAND"))
    for k in o:
        j = tr47[k]; items.append((inv49.thumb(f"{root}/test/in/{R[j]['id']}_f22.JPG"), f"{key[j]} cos{s[k]:.2f}"))
    out.append(dict(cand=c, nn=[(key[tr47[k]], float(s[k])) for k in o]))
for k in range(0, len(items), 32):
    inv49.sheet(items[k:k + 32], f"{O}/sheets/nn_{k // 32:02d}.jpg", cols=4)
json.dump(out, open(f"{O}/nn.json", "w"), indent=1)
print("ok", len(out))
