"""Candidate generation (blocking) for S1 -> S2/S3 matching.

Idea: each record emits a handful of hashed "blocking keys". Two records become a
candidate pair if they share at least one key. Keys are built so that a true match
survives typical noise (name abbreviations, shuffled address order, missing units):

  K1  country | two rarest name tokens              -> same name, any address
  K2  country | rare name token | address number    -> similar name + same house no.
  K3  country | rarest street word | address number -> same address, renamed business
  K4  country | first 5 letters of squashed name | address number -> typos / spacing

Keys that are too common (huge blocks) are dropped. Everything is uint64 hashes
and int32 row indices, so it runs on millions of rows within a few GB of RAM.
"""
from collections import Counter
import numpy as np
import pandas as pd

MAX_BLOCK = 60          # drop keys that occur more than this many times on either side
NAME_MIN_LEN = 2


def token_df(series: pd.Series) -> Counter:
    c = Counter()
    for x in series:
        c.update(set(x.split()))
    return c


def _rare(tokens, df, k):
    toks = [t for t in set(tokens) if len(t) >= NAME_MIN_LEN and not t.isdigit()]
    toks.sort(key=lambda t: (df.get(t, 0), t))
    return toks[:k]


def _nums(nums_str, k=2):
    ns = [n for n in nums_str.split() if n != "0"]
    ns.sort(key=lambda n: (-len(n), n))
    return ns[:k]


def make_keys(p: pd.DataFrame, name_df: Counter, addr_df: Counter) -> pd.DataFrame:
    """Return DataFrame(key uint64, row int32) for prepared records `p`."""
    keys, rows = [], []
    for i, (c, nc, asrt, nums) in enumerate(zip(p["country"].values, p["name_core"].values,
                                                 p["addr_sorted"].values, p["addr_nums"].values)):
        rn = _rare(nc.split(), name_df, 2)
        ns = _nums(nums)
        street = _rare([t for t in asrt.split() if t.isalpha()], addr_df, 1)
        squash = "".join(nc.split())[:5]
        ks = []
        if len(rn) == 2:
            ks.append(f"a|{c}|{rn[0]}|{rn[1]}")
        elif len(rn) == 1:
            ks.append(f"a|{c}|{rn[0]}")
        for n in ns:
            for t in rn:
                ks.append(f"b|{c}|{t}|{n}")
            for t in street:
                ks.append(f"c|{c}|{t}|{n}")
            if squash:
                ks.append(f"d|{c}|{squash}|{n}")
        keys.extend(ks)
        rows.extend([i] * len(ks))
    h = pd.util.hash_array(np.asarray(keys, dtype=object), categorize=False)
    return pd.DataFrame({"key": h, "row": np.asarray(rows, dtype=np.int32)})


def filter_keys(k1: pd.DataFrame, kt: pd.DataFrame, max_block: int = MAX_BLOCK):
    """Keep only keys present on both sides with block size <= max_block."""
    vc_t = kt["key"].value_counts()
    vc_1 = k1["key"].value_counts()
    ok = np.intersect1d(vc_t.index.values[vc_t.values <= max_block],
                        vc_1.index.values[vc_1.values <= max_block])
    return (k1[np.isin(k1["key"].values, ok)].reset_index(drop=True),
            kt[np.isin(kt["key"].values, ok)].reset_index(drop=True))


def candidate_pairs(k1: pd.DataFrame, kt: pd.DataFrame) -> pd.DataFrame:
    """Join (already filtered) S1 keys with target keys -> (i, j, n_keys)."""
    m = k1.merge(kt, on="key", suffixes=("_1", "_t"))
    out = (m.groupby(["row_1", "row_t"], sort=False).size()
             .reset_index(name="n_keys"))
    out.columns = ["i", "j", "n_keys"]
    out["i"] = out["i"].astype(np.int32)
    out["j"] = out["j"].astype(np.int32)
    out["n_keys"] = out["n_keys"].astype(np.int16)
    return out


def blocking_recall(pairs, s1_ids, t_ids, gt_map, source_prefix):
    """Share of true S1->target links (for one target source) present in `pairs`."""
    found = set(zip(s1_ids[pairs["i"].values], t_ids[pairs["j"].values]))
    tot = hit = 0
    for s, ms in gt_map.items():
        for m in ms:
            if m.startswith(source_prefix):
                tot += 1
                hit += (s, m) in found
    return hit / max(tot, 1), hit, tot
