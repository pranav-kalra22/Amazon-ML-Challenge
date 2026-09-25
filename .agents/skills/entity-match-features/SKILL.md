---
name: entity-match-features
description: Extracts comprehensive, non-destructive pairwise similarity features across business names, addresses, retrieval channels, and interactions.
---

# Entity Match Features

## Goal
Provide a canonical, vectorized feature extraction engine for candidate pairs $(S_1, S_{2/3})$ that preserves both raw and normalized information, extracting discriminative string, numeric, token, and retrieval signals.

## When to Use
- During feature matrix construction for training models.
- During test inference over candidate pairs.
- For feature ablation studies.

## Non-Destructive Representations
For every name and address, generate parallel representations without overwriting raw values:
1. **Raw Unicode**: Exact case-sensitive original string.
2. **Case-folded / Punctuation-Normalized**: Lowercase, standardized spacing, basic punctuation cleaned.
3. **Alphanumeric Compact**: Whitespace and non-alphanumeric removed.
4. **Phonetic / Transliterated**: Transliteration of Indic scripts and French diacritics via `text_unidecode`.
5. **Token Bag**: Set and multiset of extracted word tokens.
6. **Character N-grams**: Character 3-grams, 4-grams, and 5-grams.
7. **Numeric Tokens**:
   - Raw number token (e.g., `"10018C"`)
   - Pure numeric digits (e.g., `"10018"`)
   - Leading zeros stripped (e.g., `"0020"` $\rightarrow$ `"20"`)

## Feature Inventory

### 1. Name Similarity Features
- `name_exact_raw`: Exact equality of raw strings.
- `name_exact_norm`: Exact equality after basic lowercasing and whitespace normalization.
- `name_exact_alphanumeric`: Equality of compact alphanumeric representations.
- `name_domain_alias`: Indicator if one name equals the other stripped of domain suffix (`.com`, `.in`, `.org`, etc.).
- `name_token_jaccard`: Jaccard similarity of name token sets.
- `name_token_dice`: Sørensen–Dice coefficient of name tokens.
- `name_token_overlap`: Overlap coefficient ($\frac{|A \cap B|}{\min(|A|, |B|)}$).
- `name_char_3gram_jaccard`: Character 3-gram Jaccard similarity.
- `name_levenshtein_ratio`: Normalized edit distance ratio.
- `name_jaro_winkler`: Jaro-Winkler string similarity.
- `name_length_ratio`: $\frac{\min(\text{len}_1, \text{len}_2)}{\max(\text{len}_1, \text{len}_2)}$.
- `name_transliteration_similarity`: String similarity computed on transliterated strings.

### 2. Address Similarity Features
- `addr_is_empty`: Boolean flag if candidate address is empty/missing.
- `addr_exact_norm`: Equality of normalized address strings.
- `addr_token_jaccard`: Jaccard similarity of address word tokens.
- `addr_char_3gram_jaccard`: Character 3-gram similarity of addresses.
- `addr_number_exact`: 1 if building numbers match, 0 if disjoint, -1 if either is missing.
- `addr_number_digit_match`: 1 if leading-zero-stripped numeric digits match.
- `addr_common_number_count`: Count of overlapping numeric tokens.
- `addr_token_overlap`: Overlap coefficient of address tokens.

### 3. Retrieval & Candidate Context Features
- `retrieval_channel_name`: Flag indicating candidate was retrieved via name index.
- `retrieval_channel_addr`: Flag indicating candidate was retrieved via address anchor index.
- `retrieval_channel_domain`: Flag indicating candidate was retrieved via domain index.
- `candidate_rank_within_s1`: Rank of candidate for this S1 entity based on composite heuristic score.
- `candidate_count_for_s1`: Total number of candidates generated for this S1 entity.

### 4. Cross & Interaction Features
- `name_sim_x_addr_sim`: Product of name token Jaccard and address token Jaccard.
- `name_high_addr_low`: Boolean indicator for high name similarity ($>0.85$) with weak/absent address.
- `addr_high_name_low`: Boolean indicator for high address similarity ($>0.85$) with low name similarity (capturing DBA/trade-name changes).
- `source_is_s2`: Binary indicator for Source 2 vs Source 3.

## Failure Conditions
- Destructively modifying raw fields in-place (e.g., stripping legal suffixes like `services` or `ltd` permanently from the primary representation).
- Dropping raw numbers or modifying coordinates without retaining original tokens.
- Introducing data leakage across train/validation splits during feature scaling or target encoding.
