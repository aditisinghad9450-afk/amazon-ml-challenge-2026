"""Text normalisation for business names and addresses.

Everything here is vectorised over pandas Series so it scales to millions of rows.
"""
import re
import unicodedata
import pandas as pd
from unidecode import unidecode

# Legal / generic suffixes that carry little identity signal.
LEGAL = {
    "llc", "inc", "incorporated", "ltd", "limited", "pvt", "private", "co", "corp",
    "corporation", "company", "pc", "plc", "llp", "lp", "pllc", "lc", "pa", "the",
    "and", "of", "india", "opc", "pte", "gmbh", "sa", "nv", "bv", "dba",
    "lnc", "com", "www", "pvl", "llc.", "ltd.",
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
    "mount": "mt", "saint": "st", "fort": "ft", "city": "", "trail": "trl", "way": "wy",
}
# Tokens that are pure noise inside addresses.
ADDR_STOP = {"unit", "apt", "ste", "fl", "no", "bldg", "nr", "opp", "flat", "house",
             "plot", "shop", "c/o", "co", "the", "of", "and", "h", "hno", "#", ""}

_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def basic(s: pd.Series) -> pd.Series:
    """lowercase, '&'->' and ', strip punctuation, collapse whitespace."""
    s = s.fillna("").astype(str)
    nonascii = s.str.contains(r"[^\x00-\x7f]", regex=True)
    if nonascii.any():
        s = s.copy()
        s[nonascii] = s[nonascii].map(unidecode)
    s = s.str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"(?<=\b\w)\.(?=\w\b)", "", regex=True)  # p.c. -> pc, l.l.c. -> llc
    s = s.str.replace(_PUNCT, " ", regex=True)
    return s.str.replace(_SPACES, " ", regex=True).str.strip()


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "@": "a", "8": "b", "4": "a", "7": "t"})
_WEB = re.compile(r"^(www)?(.+?)(com|in|net|org|co|biz|info|us)?$")


def _fix_token(t):
    # "br0oks" -> "brooks", "de1i" -> "deli" (digits inside words are OCR-style noise)
    if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
        if not t[0].isdigit() or (len(t) >= 4 and t[1:].isalpha()):
            return t.translate(_LEET)
    return t


def name_core(name_basic: pd.Series) -> pd.Series:
    """Name without legal suffixes / connector words, with digit-for-letter noise fixed."""
    def f(x):
        toks = [_fix_token(t) for t in x.split()]
        core = [t for t in toks if t not in LEGAL and skeleton(t) not in LEGAL_SKEL]
        return " ".join(core) if core else " ".join(toks)
    return name_basic.map(f)


_SX = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")


def skeleton(tok: str) -> str:
    """Phonetic skeleton (Soundex classes, no truncation, vowels dropped, repeats collapsed).
    'engineering' and 'injiiniyring' -> '525652'; 'maharashtra' and 'mhaaraassttr' -> '56236'.
    Robust to Hindi/Telugu/Gujarati -> Latin transliteration."""
    if len(tok) <= 2 or tok.isdigit():
        return tok
    out = []
    prev = ""
    for ch in tok.translate(_SX):
        if ch.isdigit():
            if ch != prev:
                out.append(ch)
            prev = ch
        elif ch not in "hwy":
            prev = ""  # a vowel separates repeats, like Soundex
    return "".join(out) or tok


LEGAL_SKEL = {skeleton(w) for w in ("private", "limited", "incorporated", "corporation", "company",
                                     "praaivett", "limittedd", "elelpii", "praiveett")}


def name_skel(core: pd.Series) -> pd.Series:
    return core.map(lambda x: " ".join(skeleton(t) for t in x.split()))


def name_squash(core: pd.Series) -> pd.Series:
    """All letters glued, web suffix removed: 'shivaminfrastructure.com' ~ 'shivam infrastructure'."""
    def f(x):
        z = "".join(ch for ch in x if ch.isalpha())
        m = _WEB.match(z)
        return m.group(2) if m and len(m.group(2)) >= 4 else z
    return core.map(f)


def addr_norm(addr: pd.Series) -> pd.Series:
    """Canonicalise abbreviations; drop noise tokens; sort tokens (addresses come shuffled)."""
    a = basic(addr)

    def f(x):
        out = []
        for t in x.split():
            t = ADDR_MAP.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            if t in ADDR_STOP:
                continue
            out.append(t)
        return " ".join(sorted(set(out)))
    return a.map(f)


def addr_numbers(addr_basic: pd.Series) -> pd.Series:
    """Space-joined sorted set of numbers in the address, leading zeros removed ('01179' == '1179')."""
    return addr_basic.map(lambda x: " ".join(sorted({n.lstrip("0") or "0" for n in _NUM.findall(x)})))


US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "puducherry": "py", "chandigarh": "ch",
}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")


_STATE_SX = {}
for _n, _ab in US_STATES.items():
    _k = "".join(skeleton(w) for w in _n.split())
    if len(_k) >= 3:
        _STATE_SX.setdefault(_k, _ab)
_STATE_ABBR = set(US_STATES.values())
_TRANSLIT_EXTRA = {"bihaar": "br", "raajsthaan": "rj", "pshcimbngg": "wb", "dillii": "dl",
                   "gov": "ga", "asm": "as", "punjaab": "pb", "hriyaannaa": "hr", "keerlm": "kl",
                   "tmilnaadd": "tn", "odishaa": "od", "jhaarkhndd": "jh"}


def _translit_states(x: str) -> str:
    """'mhaaraassttr' -> 'mh', 'uttr prdesh' -> 'up' (states written in Indic script)."""
    toks = x.split()
    if not toks:
        return x
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        if k + 1 < len(toks) and len(t) >= 3:
            two = skeleton(t) + skeleton(toks[k + 1])
            if two in _STATE_SX and len(two) >= 4:
                out.append(_STATE_SX[two]); k += 2; continue
        if t in _TRANSLIT_EXTRA:
            out.append(_TRANSLIT_EXTRA[t]); k += 1; continue
        if len(t) >= 5 and t not in _STATE_ABBR:
            sk = skeleton(t)
            if len(sk) >= 4 and sk in _STATE_SX:
                out.append(_STATE_SX[sk]); k += 1; continue
        out.append(t); k += 1
    return " ".join(out)


def _states(a: pd.Series, translit_mask=None) -> pd.Series:
    a = a.str.replace(_STATE_RE, lambda m: US_STATES[m.group(1)], regex=True)
    if translit_mask is not None and translit_mask.any():
        a = a.copy()
        a[translit_mask] = a[translit_mask].map(_translit_states)
    return a


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the normalised columns used by blocking + features."""
    out = pd.DataFrame({"entity_id": df["entity_id"].astype(str).values})
    out["country"] = df.get("country", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().values
    nb = basic(df["business_name"])
    out["name"] = nb.values
    out["name_core"] = name_core(nb).values
    out["name_skel"] = name_skel(out["name_core"]).values
    out["name_squash"] = name_squash(out["name_core"]).values
    raw_addr = df["business_address"].fillna("").astype(str)
    had_indic = raw_addr.str.contains(r"[^\x00-\x7f]", regex=True).values
    ab = _states(basic(raw_addr), had_indic)
    out["addr"] = ab.values
    out["addr_sorted"] = addr_norm(ab).values
    out["addr_nums"] = addr_numbers(ab).values
    return out
