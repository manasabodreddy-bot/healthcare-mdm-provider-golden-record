# Data Quality Summary (Before vs After)

Generated at: **2026-02-05T15:53:03**

## Key Outcomes

- Clean source records: **4,879**

- Golden records created: **3,771**

- Clusters merged (>1): **905**

- Manual review queue (pairs): **1,885**


## Before/After Metrics Table

| metric                        | golden_providers   | providers_clean   | providers_raw_current   |
|-------------------------------|--------------------|-------------------|-------------------------|
| completeness_email            | 74.73%             | 71.61%            | 79.50%                  |
| completeness_npi              | 34.84%             | 49.64%            | 49.64%                  |
| completeness_phone            | 91.73%             | 91.88%            | 91.88%                  |
| completeness_postal_code      | 100.00%            | 100.00%           | 100.00%                 |
| completeness_state            | 98.51%             | 98.57%            | 100.00%                 |
| row_count                     | 377100.00%         | 487900.00%        | 487900.00%              |
| suspected_duplicate_row_ratio | 75.74%             | 97.36%            | 90.84%                  |
| uniqueness_ratio_email        | 65.86%             | 53.12%            | 57.46%                  |
| uniqueness_ratio_npi          | 100.00%            | 54.25%            | 54.25%                  |
| uniqueness_ratio_phone        | 58.57%             | 45.19%            | 73.97%                  |


## Notes (Stewardship Interpretation)

- **Completeness** reflects usable (non-null, non-empty) fields after standardization and survivorship.

- **Suspected duplicate ratio** is a heuristic signal; the true dedup is represented by golden record reduction.

- Manual review queue is expected in operational MDM; those pairs require steward validation.
