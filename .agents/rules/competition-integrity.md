---
trigger: always_on
description: Strict competition integrity rules prohibiting external lookups and requiring MIT/Apache-2.0 license compliance.
---

# Competition Integrity Rule (Always-On)

In compliance with the official Amazon ML Challenge 2026 guidelines and fair play policy:

1. **Strict Prohibition on External Lookups**:
   - Do NOT search Google, Bing, Yahoo, DuckDuckGo, or any web search engine for business names, addresses, or IDs from the challenge dataset.
   - Do NOT use Google Maps, OpenStreetMap, or any map service to resolve locations or addresses.
   - Do NOT query business registries, corporate registrars (e.g., MCA, SEC EDGAR, Companies House, Infogreffe), or credit bureaus.
   - Do NOT call external geocoding APIs (e.g., Google Geocoding, Nominatim, Mapbox, Radar).
   - Do NOT call external address standardization or verification APIs.
   - Do NOT use commercial entity resolution services (e.g., Senzing, Tamr, Zingg Cloud).

2. **Data Privacy and API Safety**:
   - Never send any dataset records, business names, addresses, or identifiers to external web APIs or cloud LLM endpoints.
   - Web search is permitted ONLY for general technical documentation, machine learning papers, library APIs, and open-source licenses.
   - Never include an actual entity name, address string, or record fragment from the competition dataset in any web query.

3. **Pure Machine Learning & Local Execution**:
   - All feature extraction, candidate generation, model training, and inference must execute locally and offline using the provided dataset and approved local Python libraries.
   - Do not fabricate, hallucinate, or synthesize artificial test labels.
   - Never assume knowledge of test ground-truth labels.

4. **Model & License Compliance**:
   - All machine learning models, pre-trained weights, and software dependencies must carry permissive open-source licenses (MIT, Apache 2.0, BSD).
   - Pre-trained models must not exceed 8 Billion parameters.
   - Models with non-commercial, copyleft (GPL without exception), or restrictive licenses are prohibited.
