# Full-Data Country Consistency Audit Report

**Evidence Classification:** `[FULL-DATA MEASUREMENT]`  
**Total Links Audited:** 7,638,365 across all 2,206,821 Source 1 entities  
**Audit Execution Time:** 174.6 seconds  

## Results

- **Same-Country Links:** 7,638,365 (100.000000%)
- **Cross-Country Links:** 0 (0.0%)
- **Missing Lookup IDs:** 0

### Country Pair Breakdown

- `US -> US`: 4,578,522 links (59.94%)
- `India -> India`: 3,059,843 links (40.06%)

### Source Breakdown

- **Source 2**: Same = 3,693,619, Cross = 0
- **Source 3**: Same = 3,944,746, Cross = 0

## Policy Determination

> **FULL-DATA TRAINING MEASUREMENT**: All 7,638,365 labelled US and India training links satisfy exact country equality with zero exceptions. Exact country equality is promoted to a hard blocking constraint for the candidate-generation engine. The same generic equality rule (`country_A == country_B`) is applied dynamically to unseen country labels such as France; France ground truth is unavailable and therefore France recall cannot be directly verified.
