"""Pairwise features for (S1 record, candidate record) pairs.

Uses rapidfuzz.process.cpdist which computes element-wise similarities for two
aligned lists in C++ with all CPU cores - fast enough for tens of millions of pairs.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

W = -1  # use all cores


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=W).astype(np.float32)


def _jacc(a, b):
    out = np.zeros(len(a), dtype=np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx and sy:
            out[k] = len(sx & sy) / len(sx | sy)
    return out


def _num_feats(a, b):
    """overlap of address numbers + whether the longest number agrees."""
    ov = np.zeros(len(a), dtype=np.float32)
    first = np.zeros(len(a), dtype=np.int8)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy:
            first[k] = -1
            continue
        ov[k] = len(sx & sy) / len(sx | sy)
        lx = max(sx, key=lambda n: (len(n), n))
        ly = max(sy, key=lambda n: (len(n), n))
        first[k] = int(lx == ly)
    return ov, first


def pair_features(p1: pd.DataFrame, pt: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """p1/pt = prepared frames (normalize.prepare); pairs has i, j, n_keys, src."""
    i, j = pairs["i"].values, pairs["j"].values
    g = lambda df, col, idx: df[col].values[idx].tolist()

    n1, nt = g(p1, "name", i), g(pt, "name", j)
    c1, ct = g(p1, "name_core", i), g(pt, "name_core", j)
    a1, at = g(p1, "addr_sorted", i), g(pt, "addr_sorted", j)
    r1, rt = g(p1, "addr", i), g(pt, "addr", j)
    u1, ut = g(p1, "addr_nums", i), g(pt, "addr_nums", j)

    f = pd.DataFrame(index=pairs.index)
    f["src3"] = (pairs["src"].values == 3).astype(np.int8)
    f["n_keys"] = pairs["n_keys"].values
    # name
    f["n_ratio"] = _cp(n1, nt, fuzz.ratio)
    f["n_tset"] = _cp(n1, nt, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(n1, nt, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(n1, nt, fuzz.partial_ratio)
    f["c_ratio"] = _cp(c1, ct, fuzz.ratio)
    f["c_tset"] = _cp(c1, ct, fuzz.token_set_ratio)
    f["c_jw"] = _cp(c1, ct, JaroWinkler.normalized_similarity)
    f["c_jacc"] = _jacc(c1, ct)
    f["c_len1"] = np.fromiter((len(x) for x in c1), np.int16, len(c1))
    f["c_lent"] = np.fromiter((len(x) for x in ct), np.int16, len(ct))
    # address
    f["a_ratio"] = _cp(a1, at, fuzz.ratio)
    f["a_tset"] = _cp(a1, at, fuzz.token_set_ratio)
    f["a_partial"] = _cp(r1, rt, fuzz.partial_token_set_ratio)
    f["a_jacc"] = _jacc(a1, at)
    f["num_ov"], f["num_first"] = _num_feats(u1, ut)
    f["cty_eq"] = (p1["country"].values[i] == pt["country"].values[j]).astype(np.int8)

    f["quick"] = (f["c_tset"] + f["a_tset"]).values
    return f


def add_context(f: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Context features over ALL pairs: how a pair ranks among its S1's candidates
    and among the S1s competing for the same target record."""
    quick = f["quick"].values
    grp = pd.Series(pairs["i"].values * 4 + pairs["src"].values)
    q = pd.Series(quick)
    f["q_rank"] = q.groupby(grp.values).rank(ascending=False, method="min").values.astype(np.float32)
    f["q_gap"] = (q.groupby(grp.values).transform("max") - q).values.astype(np.float32)
    f["n_cands"] = q.groupby(grp.values).transform("size").values.astype(np.int32)
    # and the reverse direction: is this S1 the best S1 for that target record?
    grp_t = pd.Series(pairs["j"].values.astype(np.int64) * 4 + pairs["src"].values)
    f["q_rank_t"] = q.groupby(grp_t.values).rank(ascending=False, method="min").values.astype(np.float32)
    f["n_cands_t"] = q.groupby(grp_t.values).transform("size").values.astype(np.int32)
    return f

