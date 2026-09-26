#!/usr/bin/env python3
"""
Non-Destructive Normalizer for Business Entity Resolution.

Preserves raw fields alongside multiple normalized, compact, transliterated,
and domain-derived representations. Never destructively discards discriminative information.
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional
import text_unidecode

# Comprehensive multi-jurisdiction legal entity forms
LEGAL_TERMS = {
    # US / UK / Commonwealth
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "plc", "holdings", "holding",
    "group", "enterprises", "enterprise", "solutions", "services", "service", "technologies",
    "associates", "partners", "intl", "international", "usa", "india",
    # French / European
    "sarl", "sasu", "eurl", "sa", "gmbh", "bv", "nv", "snc", "scs", "sca",
    "association", "ets", "etablissement", "cie",
    # Universal grammatical / web noise tokens that are non-discriminative
    "and", "the", "of", "in", "for", "to", "at", "by", "from", "with",
    "com", "net", "org", "co", "www", "online", "global"
}

DOMAIN_EXT_REGEX = re.compile(
    r'\.(com|in|org|net|co|io|fr|biz|info|us|gov|edu|ai|tech|co\.in|org\.in|co\.uk)\b',
    re.IGNORECASE
)

ROAD_MAPPINGS = {
    "st": "street", "ave": "avenue", "rd": "road", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "sq": "square", "ste": "suite", "apt": "apartment",
    "fl": "floor", "bldg": "building", "dept": "department", "no": "number"
}


def normalize_name_non_destructive(raw_name: Any) -> Dict[str, Any]:
    """
    Extracts comprehensive, non-destructive representations from a business name.
    
    Retains:
    - raw: exact input Unicode string
    - norm_unicode: NFKC normalized, lowercased, trimmed
    - punct_norm: punctuation replaced with spaces, single spaced
    - compact_alnum: alphanumeric only (spaces & punctuation removed)
    - domain_root: stripped domain root if a domain name pattern is detected
    - legal_stripped: name with known legal designations removed
    - legal_stripped_compact: legal-stripped alphanumeric compact form
    - transliterated: ASCII transliteration via text_unidecode
    - trans_stripped: transliterated legal-stripped form
    - trans_compact: transliterated compact form
    - tokens: all word tokens (len >= 2)
    - distinctive_tokens: non-legal tokens (len >= 3)
    """
    if raw_name is None or (isinstance(raw_name, float) and str(raw_name) == "nan"):
        raw_str = ""
    else:
        raw_str = str(raw_name).strip()
        
    if not raw_str or raw_str.lower() in {"", "nan", "<null>", "null", "none"}:
        return {
            "raw": raw_str if raw_name is not None and not (isinstance(raw_name, float) and str(raw_name) == "nan") else "",
            "norm_unicode": "",
            "punct_norm": "",
            "compact_alnum": "",
            "domain_root": "",
            "legal_stripped": "",
            "legal_stripped_compact": "",
            "transliterated": "",
            "trans_stripped": "",
            "trans_compact": "",
            "tokens": [],
            "distinctive_tokens": [],
            "trans_tokens": [],
            "trans_distinctive_tokens": [],
            "sorted_token_signature": ""
        }
        
    is_asc = raw_str.isascii()
    if is_asc:
        norm_u = raw_str.lower()
        trans = norm_u
    else:
        norm_u = unicodedata.normalize("NFKC", raw_str).lower().strip()
        trans = text_unidecode.unidecode(norm_u).strip()
    
    # 3. Domain-derived root
    domain_root = ""
    domain_match = DOMAIN_EXT_REGEX.search(norm_u)
    if domain_match:
        # e.g., "brightiheartmedia.com" or "www.brightiheartmedia.co.in"
        pre_ext = norm_u[:domain_match.start()].strip()
        pre_ext = re.sub(r'^(https?://)?(www\.)?', '', pre_ext)
        d_root = re.sub(r'[^a-z0-9]', '', pre_ext)
        if len(d_root) >= 3:
            domain_root = d_root
            
    # 4. Punctuation normalization
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', norm_u)
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    punct_norm = " ".join(tokens)
    
    # 5. Compact alphanumeric
    compact_alnum = re.sub(r'[^a-z0-9]', '', norm_u)
    
    # 6. Legal-form-stripped (core brand tokens)
    core_tokens = [t for t in tokens if t not in LEGAL_TERMS]
    legal_stripped = " ".join(core_tokens) if core_tokens else punct_norm
    legal_stripped_compact = re.sub(r'[^a-z0-9]', '', legal_stripped)
    
    # 7. Transliterated representations
    if is_asc:
        trans_stripped = legal_stripped
        trans_compact = compact_alnum
    else:
        trans_cleaned = re.sub(r'[^a-z0-9\s]', ' ', trans)
        trans_tokens = [t for t in trans_cleaned.split() if len(t) >= 2]
        trans_core = [t for t in trans_tokens if t not in LEGAL_TERMS]
        trans_stripped = " ".join(trans_core) if trans_core else " ".join(trans_tokens)
        trans_compact = re.sub(r'[^a-z0-9]', '', trans)
    
    # 8. Distinctive tokens (len >= 3 and not in legal set)
    distinctive = [t for t in core_tokens if len(t) >= 3 and not t.isdigit()]

    # 9. Transliterated tokens & distinctive tokens
    if is_asc:
        trans_tokens = tokens
        base_trans_dist = distinctive
    else:
        trans_tokens = [t for t in trans_cleaned.split() if len(t) >= 2]
        base_trans_dist = [t for t in trans_core if len(t) >= 3 and not t.isdigit()]

    # Repeat-collapsed transliterated tokens (e.g., "saaii" -> "sai") to bridge unidecode spelling variations
    trans_dist_set = set(base_trans_dist)
    for t in base_trans_dist:
        collapsed = re.sub(r'(.)\1+', r'\1', t)
        if len(collapsed) >= 3:
            trans_dist_set.add(collapsed)
    trans_distinctive = list(trans_dist_set)

    # 10. Sorted token signature
    sorted_sig = " ".join(sorted(distinctive)) if distinctive else ""

    return {
        "raw": raw_str,
        "norm_unicode": norm_u,
        "punct_norm": punct_norm,
        "compact_alnum": compact_alnum,
        "domain_root": domain_root,
        "legal_stripped": legal_stripped,
        "legal_stripped_compact": legal_stripped_compact,
        "transliterated": trans,
        "trans_stripped": trans_stripped,
        "trans_compact": trans_compact,
        "tokens": tokens,
        "distinctive_tokens": distinctive,
        "trans_tokens": trans_tokens,
        "trans_distinctive_tokens": trans_distinctive,
        "sorted_token_signature": sorted_sig
    }


COMMON_ADDR_STOP = {
    "road", "street", "st", "rd", "ave", "avenue", "lane", "ln", "dr", "drive",
    "blvd", "boulevard", "court", "ct", "place", "pl", "way", "circle", "cir",
    "unit", "suite", "ste", "apt", "apartment", "fl", "floor", "bldg", "building",
    "near", "opp", "opposite", "dist", "district", "post", "po", "box", "hwy", "highway",
    "state", "city", "county", "sector", "plot", "flat", "gali", "colony", "nagar",
    "west", "east", "north", "south", "central", "first", "second", "third",
    # Building/unit prefix labels that are not street identifiers
    "no", "num", "number", "house", "shop", "office", "room", "khasra", "survey", "gala", "cabin", "hall"
}


def normalize_address_non_destructive(raw_addr: Any) -> Dict[str, Any]:
    """
    Extracts comprehensive, non-destructive representations from a business address.
    
    Retains:
    - raw: exact input Unicode string
    - norm_unicode: NFKC normalized, lowercased, trimmed
    - compact_norm: alphanumeric characters only
    - tokens: all word/alphanumeric tokens
    - numeric_tokens: all numeric sequences (e.g., ["10018", "42"])
    - building_raw: raw building identifier
    - building_numeric: purely numeric digits of building
    - all_building_numerics: set of candidate building numbers found anywhere in address
    - postal_code: detected 5-digit US ZIP or 6-digit India PIN
    - street_tokens: non-building, non-numeric street words
    - road_standardized_tokens: tokens with abbreviations expanded
    - distinctive_tokens: non-stop, non-numeric locality/street tokens (len >= 4)
    - transliterated: ASCII transliteration
    - trans_distinctive_tokens: transliterated locality/street tokens
    """
    if raw_addr is None or (isinstance(raw_addr, float) and str(raw_addr) == "nan"):
        raw_str = ""
    else:
        raw_str = str(raw_addr).strip()
        
    if not raw_str or raw_str.lower() in {"", "nan", "<null>", "null", "none"}:
        return {
            "raw": raw_str,
            "norm_unicode": "",
            "compact_norm": "",
            "tokens": [],
            "numeric_tokens": [],
            "building_raw": "",
            "building_numeric": "",
            "all_building_numerics": [],
            "postal_code": "",
            "street_tokens": [],
            "road_standardized_tokens": [],
            "distinctive_tokens": [],
            "transliterated": "",
            "trans_distinctive_tokens": []
        }
        
    # 1. Unicode normalization & lower
    is_asc = raw_str.isascii()
    if is_asc:
        norm_u = raw_str.lower()
        trans = norm_u
    else:
        norm_u = unicodedata.normalize("NFKC", raw_str).lower().strip()
        trans = text_unidecode.unidecode(norm_u).strip()
    
    compact_norm = re.sub(r'[^a-z0-9]', '', norm_u)

    # 2. Extract Postal / PIN code directly from raw address
    postal = ""
    us_zip = re.search(r'\b(\d{5})(-\d{4})?\b', raw_str)
    in_pin = re.search(r'\b(\d{6})\b', raw_str)
    if us_zip:
        postal = us_zip.group(1)
    elif in_pin:
        postal = in_pin.group(1)
        
    # 3. Tokenize
    tokens = re.findall(r'[a-z0-9]+', norm_u)
    numeric_tokens = [t for t in tokens if t.isdigit() and len(t) <= 6]
    
    # 4. Building number extraction
    building_raw = ""
    building_numeric = ""
    for t in tokens[:4]:
        if any(c.isdigit() for c in t):
            building_raw = t
            digits = re.sub(r'[^0-9]', '', t)
            if digits:
                building_numeric = digits
            break
            
    # All building numerics: unique numeric tokens between 1 and 5 digits (excluding postal)
    all_bldg_num = [t for t in numeric_tokens if 1 <= len(t) <= 5 and t != postal]
    if building_numeric and building_numeric not in all_bldg_num:
        all_bldg_num.insert(0, building_numeric)

    # 5. Street & road tokens
    street_tokens = []
    road_std_tokens = []
    for t in tokens:
        if len(t) >= 2 and not t.isdigit() and t != building_raw:
            street_tokens.append(t)
            expanded = ROAD_MAPPINGS.get(t, t)
            road_std_tokens.append(expanded)
            
    # 6. Distinctive address tokens (locality, distinctive street name)
    distinctive_tokens = [
        t for t in tokens
        if len(t) >= 4 and not t.isdigit() and t not in COMMON_ADDR_STOP and t not in LEGAL_TERMS
    ]

    # 7. Transliterated distinctive address tokens
    if is_asc:
        trans_distinctive = distinctive_tokens
    else:
        trans_tokens = re.findall(r'[a-z0-9]+', trans)
        trans_distinctive = [
            t for t in trans_tokens
            if len(t) >= 4 and not t.isdigit() and t not in COMMON_ADDR_STOP and t not in LEGAL_TERMS
        ]

    return {
        "raw": raw_str,
        "norm_unicode": norm_u,
        "compact_norm": compact_norm,
        "tokens": tokens,
        "numeric_tokens": numeric_tokens,
        "building_raw": building_raw,
        "building_numeric": building_numeric,
        "all_building_numerics": all_bldg_num,
        "postal_code": postal,
        "street_tokens": street_tokens[:6],
        "road_standardized_tokens": road_std_tokens[:6],
        "distinctive_tokens": distinctive_tokens[:8],
        "transliterated": trans,
        "trans_distinctive_tokens": trans_distinctive[:8]
    }

