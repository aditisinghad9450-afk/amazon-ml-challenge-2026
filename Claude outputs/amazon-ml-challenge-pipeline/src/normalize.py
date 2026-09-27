"""Text normalisation for business names and addresses.

Everything here is vectorised over pandas Series so it scales to millions of rows.
"""
import re
import pandas as pd

# Legal / generic suffixes that carry little identity signal.
LEGAL = {
    "llc", "inc", "incorporated", "ltd", "limited", "pvt", "private", "co", "corp",
    "corporation", "company", "pc", "plc", "llp", "lp", "pllc", "lc", "pa", "the",
    "and", "of", "india", "opc", "pte", "gmbh", "sa", "nv", "bv", "dba",
}

# Canonical forms for common address abbreviations.
ADDR_MAP = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd",
    "drive": "dr", "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "circle": "cir", "highway": "hwy", "parkway": "pkwy", "terrace": "ter",
    "square": "sq", "suite": "ste", "apartment": "apt", "unit": "unit", "floor": "fl",
    "building": "bldg", "number": "no", "north": "n", "south": "s", "east": "e",
    "west": "w", "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "ground": "gnd", "opposite": "opp", "near": "nr", "sector": "sec", "nagar": "ngr",
    "mount": "mt", "saint": "st", "fort": "ft", "city": "",
}
# Tokens that are pure noise inside addresses.
ADDR_STOP = {"unit", "apt", "ste", "fl", "no", "bldg", "nr", "opp", "flat", "house",
             "plot", "shop", "c/o", "co", "the", "of", "and", "h", "hno", "#", ""}

_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def basic(s: pd.Series) -> pd.Series:
    """lowercase, '&'->' and ', strip punctuation, collapse whitespace."""
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"(?<=\b\w)\.(?=\w\b)", "", regex=True)  # p.c. -> pc, l.l.c. -> llc
    s = s.str.replace(_PUNCT, " ", regex=True)
    return s.str.replace(_SPACES, " ", regex=True).str.strip()


def name_core(name_basic: pd.Series) -> pd.Series:
    """Name without legal suffixes / connector words."""
    def f(x):
        toks = [t for t in x.split() if t not in LEGAL]
        return " ".join(toks) if toks else x
    return name_basic.map(f)


def addr_norm(addr: pd.Series) -> pd.Series:
    """Canonicalise abbreviations; drop noise tokens; sort tokens (addresses come shuffled)."""
    a = basic(addr)

    def f(x):
        out = []
        for t in x.split():
            t = ADDR_MAP.get(t, t)
            if t in ADDR_STOP:
                continue
            out.append(t)
        return " ".join(sorted(set(out)))
    return a.map(f)


def addr_numbers(addr_basic: pd.Series) -> pd.Series:
    """Space-joined sorted set of numbers in the address (house numbers etc.)."""
    return addr_basic.map(lambda x: " ".join(sorted(set(_NUM.findall(x)))))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the normalised columns used by blocking + features."""
    out = pd.DataFrame({"entity_id": df["entity_id"].astype(str).values})
    out["country"] = df.get("country", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().values
    nb = basic(df["business_name"])
    out["name"] = nb.values
    out["name_core"] = name_core(nb).values
    ab = basic(df["business_address"])
    out["addr"] = ab.values
    out["addr_sorted"] = addr_norm(df["business_address"]).values
    out["addr_nums"] = addr_numbers(ab).values
    return out
