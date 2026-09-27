"""End-to-end entity-resolution pipeline for Amazon ML Challenge 2026.

    python src/pipeline.py train     # blocking + features + LightGBM + threshold tuning
    python src/pipeline.py predict   # writes output/submission.tsv for the test set

Run from the project root (the folder that contains dataset/).
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd

import blocking as B
import features as F
from normalize import prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "output")
MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(ROOT, "cache")

TOP_K = 8                # candidates kept per (S1 record, source) after the quick score
TRAIN_S1_SAMPLE = 400_000  # S1 records whose pairs are used to fit the model
BETA = 0.5
SEED = 42

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- data
def _path(split, name):
    for ext in (".tsv", ".tsv.gz"):
        p = os.path.join(DATA, split, f"{split}_{name}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{split}_{name}.tsv(.gz) not found in {DATA}/{split}")


def load_prepared(split, name):
    """Read a source file and return the normalised frame (cached as parquet)."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"{split}_{name}.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    parts = []
    for chunk in pd.read_csv(_path(split, name), sep="\t", dtype=str, keep_default_na=False,
                             chunksize=500_000, quoting=3):
        parts.append(prepare(chunk))
    p = pd.concat(parts, ignore_index=True)
    p.to_parquet(cp, index=False)
    return p


def load_gt():
    gt = pd.read_csv(_path("train", "ground_truth"), sep="\t", dtype=str, keep_default_na=False)
    return {s: [m for m in ms.split(",") if m] for s, ms in
            zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values)}


# --------------------------------------------------------------------------- candidates
def build_candidates(split):
    """Blocking + quick score + global context features + top-K pruning.
    Returns (p1, {2: p2, 3: p3}, pairs, context_features)."""
    log(f"[{split}] loading + normalising sources")
    p1 = load_prepared(split, "source1")
    pts = {2: load_prepared(split, "source2"), 3: load_prepared(split, "source3")}
    log(f"[{split}] S1={len(p1):,} S2={len(pts[2]):,} S3={len(pts[3]):,}")

    name_df = B.token_df(p1["name_core"])
    addr_df = B.token_df(p1["addr_sorted"])
    k1 = B.make_keys(p1, name_df, addr_df)
    log(f"[{split}] S1 keys: {len(k1):,}")

    all_pairs = []
    for s, pt in pts.items():
        kt = B.make_keys(pt, name_df, addr_df)
        a, b = B.filter_keys(k1, kt)
        del kt
        # merge in S1 slices to bound memory
        rows = a["row"].values
        step = 300_000
        for lo in range(0, len(p1), step):
            sub = a[(rows >= lo) & (rows < lo + step)]
            pr = B.candidate_pairs(sub, b)
            pr["src"] = np.int8(s)
            all_pairs.append(pr)
        del a, b
        gc.collect()
        log(f"[{split}] S{s}: {sum(len(x) for x in all_pairs if x['src'].iat[0] == s):,} raw pairs")
    pairs = pd.concat(all_pairs, ignore_index=True)
    del all_pairs
    gc.collect()

    # quick score for every pair (cheap), used for context features and pruning
    quick = np.empty(len(pairs), dtype=np.float32)
    for s, pt in pts.items():
        idx = np.where(pairs["src"].values == s)[0]
        for lo in range(0, len(idx), 2_000_000):
            ii = idx[lo:lo + 2_000_000]
            i, j = pairs["i"].values[ii], pairs["j"].values[ii]
            quick[ii] = (F._cp(p1["name_core"].values[i].tolist(), pt["name_core"].values[j].tolist(),
                               F.fuzz.token_set_ratio)
                         + F._cp(p1["addr_sorted"].values[i].tolist(), pt["addr_sorted"].values[j].tolist(),
                                 F.fuzz.token_set_ratio))
    ctx = pd.DataFrame({"quick": quick})
    ctx = F.add_context(ctx, pairs)
    keep = ctx["q_rank"].values <= TOP_K
    pairs = pairs[keep].reset_index(drop=True)
    ctx = ctx[keep].reset_index(drop=True)
    log(f"[{split}] pairs after top-{TOP_K} pruning: {len(pairs):,}")
    return p1, pts, pairs, ctx


def full_features(p1, pts, pairs, ctx):
    out = []
    for s, pt in pts.items():
        m = pairs["src"].values == s
        sub = pairs[m]
        for lo in range(0, len(sub), 1_000_000):
            part = sub.iloc[lo:lo + 1_000_000]
            f = F.pair_features(p1, pt, part)
            f.index = part.index
            out.append(f)
    f = pd.concat(out).sort_index()
    for c in ["q_rank", "q_gap", "n_cands", "q_rank_t", "n_cands_t"]:
        f[c] = ctx[c].values
    return f


# --------------------------------------------------------------------------- matching
def resolve(pairs, prob, thr, one_s1_per_target=True):
    """Pairs whose probability passes `thr`; optionally each target record is
    assigned to at most one S1 record (the highest-scoring one)."""
    d = pd.DataFrame({"i": pairs["i"].values, "j": pairs["j"].values,
                      "src": pairs["src"].values, "p": prob})
    d = d[d["p"] >= thr]
    if one_s1_per_target:
        d = d.sort_values("p", ascending=False).drop_duplicates(["j", "src"])
    return d


def fbeta(tp, fp, fn, beta=BETA):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    b2 = beta * beta
    return (1 + b2) * p * r / max(b2 * p + r, 1e-9), p, r


def evaluate(d, p1, pts, gt, s1_subset):
    ids1 = p1["entity_id"].values
    pred = set()
    for s, pt in pts.items():
        x = d[d["src"] == s]
        pred |= set(zip(ids1[x["i"].values], pt["entity_id"].values[x["j"].values]))
    true = {(s, m) for s in s1_subset for m in gt.get(s, [])}
    tp = len(pred & true)
    return fbeta(tp, len(pred) - tp, len(true) - tp)


# --------------------------------------------------------------------------- train
def train():
    import lightgbm as lgb
    os.makedirs(MODELS, exist_ok=True)
    gt = load_gt()
    p1, pts, pairs, ctx = build_candidates("train")

    ids1 = p1["entity_id"].values
    for s, pt in pts.items():
        m = pairs["src"].values == s
        r = B.blocking_recall(pairs[m], ids1, pt["entity_id"].values, gt, f"S{s}-")
        log(f"blocking recall S{s}: {r[0]:.4f} ({r[1]:,}/{r[2]:,})")

    # sample S1 records, split into fit / validation by S1 record
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(p1), size=min(TRAIN_S1_SAMPLE, len(p1)), replace=False)
    is_val = np.zeros(len(p1), dtype=np.int8) - 1
    is_val[samp] = (rng.random(len(samp)) < 0.2).astype(np.int8)
    role = is_val[pairs["i"].values]
    sel = role >= 0
    sp, sc = pairs[sel].reset_index(drop=True), ctx[sel].reset_index(drop=True)
    role = role[sel]
    log(f"computing full features for {len(sp):,} sampled pairs")
    X = full_features(p1, pts, sp, sc)

    true = {(s, m) for s, ms in gt.items() for m in ms}
    y = np.fromiter(((ids1[i], pts[s]["entity_id"].values[j]) in true
                     for i, j, s in zip(sp["i"].values, sp["j"].values, sp["src"].values)),
                    dtype=np.int8, count=len(sp))
    log(f"positives: {y.mean():.3f}")

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=0)
    tr, va = role == 0, role == 1
    dtr = lgb.Dataset(X[tr], y[tr])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    model = lgb.train(params, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    prob = model.predict(X[va], num_iteration=model.best_iteration)

    val_s1 = set(ids1[np.where(is_val == 1)[0]])
    vp = sp[va].reset_index(drop=True)
    best = (0, 0.5, True)
    for one in (True, False):
        for thr in np.arange(0.05, 0.99, 0.025):
            f, p, r = evaluate(resolve(vp, prob, thr, one), p1, pts, gt, val_s1)
            if f > best[0]:
                best = (f, float(thr), one, p, r)
    log(f"VALIDATION F0.5={best[0]:.4f}  thr={best[1]:.3f}  one_s1_per_target={best[2]}  "
        f"P={best[3]:.4f} R={best[4]:.4f}")

    model.save_model(os.path.join(MODELS, "lgbm.txt"), num_iteration=model.best_iteration)
    json.dump({"threshold": best[1], "one_s1_per_target": best[2], "val_f05": best[0],
               "features": list(X.columns)}, open(os.path.join(MODELS, "config.json"), "w"), indent=2)
    imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    log("feature importance (gain):\n" + imp.round(0).to_string())


# --------------------------------------------------------------------------- predict
def predict():
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(MODELS, "config.json")))
    model = lgb.Booster(model_file=os.path.join(MODELS, "lgbm.txt"))
    p1, pts, pairs, ctx = build_candidates("test")

    prob = np.empty(len(pairs), dtype=np.float32)
    step = 3_000_000
    for lo in range(0, len(pairs), step):
        part, pc = pairs.iloc[lo:lo + step].reset_index(drop=True), ctx.iloc[lo:lo + step].reset_index(drop=True)
        X = full_features(p1, pts, part, pc)[cfg["features"]]
        prob[lo:lo + step] = model.predict(X)
        log(f"scored {min(lo + step, len(pairs)):,}/{len(pairs):,}")

    d = resolve(pairs, prob, cfg["threshold"], cfg["one_s1_per_target"])
    ids1 = p1["entity_id"].values
    tgt = np.empty(len(d), dtype=object)
    for s, pt in pts.items():
        m = d["src"].values == s
        tgt[m] = pt["entity_id"].values[d["j"].values[m]]
    d = pd.DataFrame({"s1": ids1[d["i"].values], "src": d["src"].values, "t": tgt})
    d = d.sort_values(["s1", "src", "t"])
    agg = d.groupby("s1")["t"].agg(",".join)
    sub = pd.DataFrame({"source1_entity_id": ids1})
    sub["matched_entity_ids"] = sub["source1_entity_id"].map(agg).fillna("")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "submission.tsv")
    sub.to_csv(path, sep="\t", index=False)
    log(f"wrote {path}: {len(sub):,} rows, {len(d):,} links, "
        f"{(sub['matched_entity_ids'] != '').mean():.3f} of S1 with ≥1 match")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "all"])
    a = ap.parse_args()
    if a.mode in ("train", "all"):
        train()
    if a.mode in ("predict", "all"):
        predict()
