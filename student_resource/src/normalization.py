"""
normalization.py — Text normalization for business names, addresses, and countries.

Conservative normalization preserves information while reducing noise from
punctuation, casing, abbreviation variants, and whitespace.
"""

import re
import unicodedata

import pandas as pd


# ── Legal suffix / abbreviation mappings ──────────────────────────────────────
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "intl": "international",
    "natl": "national",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "tech": "technologies",
    "assoc": "associates",
    "grp": "group",
    "hldgs": "holdings",
    "hldg": "holding",
    "enterp": "enterprises",
    "ent": "enterprises",
    "soln": "solutions",
    "solns": "solutions",
    "mgmt": "management",
    "sys": "systems",
    "inds": "industries",
    "ind": "industries",
    "infra": "infrastructure",
    "engr": "engineering",
    "engg": "engineering",
    "telecom": "telecommunications",
    "pharma": "pharmaceuticals",
    "govt": "government",
    "dept": "department",
    "univ": "university",
    "hosp": "hospital",
    "mkt": "market",
    "fin": "financial",
    "prop": "properties",
}

# ── Address abbreviation mappings ─────────────────────────────────────────────
ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "cir": "circle",
    "hwy": "highway",
    "sq": "square",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "opp": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "dist": "district",
    "sec": "sector",
    "ph": "phase",
    "ext": "extension",
    "nagar": "nagar",
    "marg": "marg",
    "chowk": "chowk",
    "gali": "gali",
    "mohalla": "mohalla",
}

# ── Indian city name aliases ──────────────────────────────────────────────────
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "mumbai": "bombay",
    "chennai": "madras",
    "kolkata": "calcutta",
    "thiruvananthapuram": "trivandrum",
    "kochi": "cochin",
    "pune": "poona",
    "varanasi": "benaras",
    "banaras": "benaras",
    "shimla": "simla",
    "mysuru": "mysore",
    "mangaluru": "mangalore",
    "hubballi": "hubli",
    "belagavi": "belgaum",
    "tumakuru": "tumkur",
    "vijayapura": "bijapur",
    "kalaburagi": "gulbarga",
    "raichur": "raichur",
    "gurugram": "gurgaon",
    "noida": "noida",
}

# ── Country normalization ────────────────────────────────────────────────────
COUNTRY_ALIASES = {
    "usa": "us",
    "u.s.a.": "us",
    "u.s.a": "us",
    "u.s.": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "in": "india",
    "ind": "india",
    "bharat": "india",
    "fr": "france",
    "fra": "france",
    "république française": "france",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "de": "germany",
    "deu": "germany",
    "deutschland": "germany",
    "cn": "china",
    "chn": "china",
    "jp": "japan",
    "jpn": "japan",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics / accents from unicode text."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces / tabs / newlines into single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name: str) -> str:
    """Normalize a business name for comparison.

    Steps:
        1. Unicode normalize & strip accents
        2. Lowercase
        3. Remove punctuation (keep alphanumeric, space)
        4. Expand abbreviations
        5. Collapse whitespace
    """
    if not name or not name.strip():
        return ""

    text = _strip_accents(name)
    text = text.lower()

    # Remove periods, commas, hyphens used in abbreviations
    text = re.sub(r"[.\-/\\,;:!?'\"()\[\]{}&@#$%^*_+=|~`<>]", " ", text)

    # Collapse whitespace before token processing
    text = _collapse_whitespace(text)

    # Expand abbreviations token by token
    tokens = text.split()
    expanded = []
    for token in tokens:
        expanded.append(LEGAL_SUFFIX_MAP.get(token, token))
    text = " ".join(expanded)

    return _collapse_whitespace(text)


def normalize_address(address: str) -> str:
    """Normalize a business address for comparison.

    Steps:
        1. Unicode normalize & strip accents
        2. Lowercase
        3. Remove punctuation (keep alphanumeric, space)
        4. Apply city aliases
        5. Expand address abbreviations
        6. Collapse whitespace
    """
    if not address or not address.strip():
        return ""

    text = _strip_accents(address)
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[.\-/\\,;:!?'\"()\[\]{}&@#$%^*_+=|~`<>]", " ", text)
    text = _collapse_whitespace(text)

    # Apply city aliases
    tokens = text.split()
    aliased = []
    for token in tokens:
        aliased.append(CITY_ALIASES.get(token, token))
    tokens = aliased

    # Expand address abbreviations
    expanded = []
    for token in tokens:
        expanded.append(ADDRESS_ABBREV_MAP.get(token, token))

    text = " ".join(expanded)
    return _collapse_whitespace(text)


def normalize_country(country: str) -> str:
    """Normalize country to a canonical lowercase form."""
    if not country or not country.strip():
        return ""

    text = _strip_accents(country).lower().strip()
    text = re.sub(r"[.\-]", " ", text)
    text = _collapse_whitespace(text)

    return COUNTRY_ALIASES.get(text, text)


def extract_numeric_tokens(text: str) -> set:
    """Extract all numeric substrings (house numbers, postal codes, etc.)."""
    return set(re.findall(r"\b\d+\b", text))


def extract_postal_code(address: str) -> str:
    """Try to extract a postal/ZIP code from the address.

    Handles:
        - Indian PIN codes: 6 digits
        - US ZIP codes: 5 digits or 5+4
        - French postal codes: 5 digits
        - Generic: last standalone number >= 5 digits
    """
    if not address:
        return ""

    # Indian PIN code (6 digits)
    m = re.search(r"\b(\d{6})\b", address)
    if m:
        return m.group(1)

    # US ZIP (5 or 5+4 digits)
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    if m:
        return m.group(1)

    return ""


def extract_name_tokens(name: str) -> set:
    """Extract meaningful tokens from a normalized business name.

    Removes very common / low-information tokens.
    """
    stop_words = {
        "private", "limited", "incorporated", "corporation", "company",
        "limited liability company", "limited liability partnership",
        "the", "of", "and", "for", "in", "at", "to", "a", "an",
        "group", "holdings", "enterprises", "solutions", "services",
        "industries", "international", "associates", "llc", "llp",
        "inc", "corp", "co", "pvt", "ltd",
    }
    tokens = set(name.split())
    return tokens - stop_words


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a source dataframe."""
    df = df.copy()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["country_norm"] = df["country"].apply(normalize_country)
    df["name_tokens"] = df["name_norm"].apply(lambda x: set(x.split()) if x else set())
    df["addr_tokens"] = df["addr_norm"].apply(lambda x: set(x.split()) if x else set())
    df["numeric_tokens"] = df["addr_norm"].apply(extract_numeric_tokens)
    df["postal_code"] = df["business_address"].apply(extract_postal_code)

    return df
