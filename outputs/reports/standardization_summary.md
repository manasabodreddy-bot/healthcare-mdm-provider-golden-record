# Standardization & Cleansing Summary

Generated at: **2026-02-05T00:33:18**

## Output Artifacts

- Clean providers CSV: `data\interim\providers_clean.csv`

- Clean organizations CSV: `data\interim\organizations_clean.csv`

- SQLite tables: `stg_providers_clean`, `stg_organizations_clean`, `stg_standardization_issues`


## Key Completeness Metrics (Post-Standardization)

| entity        | metric                   |    value |
|---------------|--------------------------|----------|
| providers     | completeness_email       | 0.71613  |
| providers     | completeness_phone       | 0.918836 |
| providers     | completeness_postal_code | 1        |
| providers     | completeness_state       | 0.985653 |
| organizations | completeness_postal_code | 1        |
| organizations | completeness_state       | 0.978747 |


## Standardization Issues Logged (Counts)

| entity       | issue              |   count |
|--------------|--------------------|---------|
| organization | invalid_state_code |      19 |
| provider     | invalid_email      |     385 |
| provider     | invalid_state_code |      70 |