# Raw Data Profiling Report (MDM Stewardship Baseline)

Generated at: **2026-02-05T00:10:27**

## Dataset Overview

| entity            |   row_count |   distinct_source_systems |
|-------------------|-------------|---------------------------|
| providers_raw     |        4879 |                         2 |
| organizations_raw |         894 |                         2 |
| affiliations_raw  |        6376 |                       nan |


## Provider (HCP) – Null / Missingness Profile

| column             |   null_count | null_pct   |
|--------------------|--------------|------------|
| first_name         |            0 | 0.00%      |
| last_name          |            0 | 0.00%      |
| address_line1      |            0 | 0.00%      |
| city               |            0 | 0.00%      |
| state              |            0 | 0.00%      |
| postal_code        |            0 | 0.00%      |
| country            |            0 | 0.00%      |
| provider_id_source |            0 | 0.00%      |
| npi                |         2457 | 50.36%     |
| email              |         1000 | 20.50%     |
| phone              |          396 | 8.12%      |


## Organization (HCO) – Null / Missingness Profile

| column        |   null_count | null_pct   |
|---------------|--------------|------------|
| org_name      |            0 | 0.00%      |
| org_type      |            0 | 0.00%      |
| address_line1 |            0 | 0.00%      |
| city          |            0 | 0.00%      |
| state         |            0 | 0.00%      |
| postal_code   |            0 | 0.00%      |
| country       |            0 | 0.00%      |


## Affiliations – Null / Missingness Profile

| column             |   null_count | null_pct   |
|--------------------|--------------|------------|
| provider_id_source |            0 | 0.00%      |
| org_id_source      |            0 | 0.00%      |
| affiliation_type   |            0 | 0.00%      |
| start_date         |            0 | 0.00%      |


## Baseline Validity Checks

| check                                                    | value          |
|----------------------------------------------------------|----------------|
| Provider state validity (country,state in reference)     | 98.57%         |
| Organization state validity (country,state in reference) | 97.87%         |
| Provider suspected duplicate rows (heuristic)            | 4,432 (90.84%) |
| Organization suspected duplicate rows (heuristic)        | 762 (85.23%)   |
| Provider NPI uniqueness ratio (non-null)                 | 54.25%         |
| Provider email uniqueness ratio (non-null)               | 57.46%         |
| Provider phone uniqueness ratio (non-null)               | 73.97%         |


## Quality Baseline (for Before/After comparisons)

Saved to: `outputs/reports/raw_quality_baseline.csv`


| entity            | metric                          |    value |
|-------------------|---------------------------------|----------|
| providers_raw     | completeness_first_name         | 1        |
| providers_raw     | completeness_last_name          | 1        |
| providers_raw     | completeness_address_line1      | 1        |
| providers_raw     | completeness_city               | 1        |
| providers_raw     | completeness_state              | 1        |
| providers_raw     | completeness_postal_code        | 1        |
| providers_raw     | completeness_country            | 1        |
| providers_raw     | completeness_provider_id_source | 1        |
| providers_raw     | completeness_npi                | 0.496413 |
| providers_raw     | completeness_email              | 0.79504  |
| providers_raw     | completeness_phone              | 0.918836 |
| organizations_raw | completeness_org_name           | 1        |
| organizations_raw | completeness_org_type           | 1        |
| organizations_raw | completeness_address_line1      | 1        |
| organizations_raw | completeness_city               | 1        |
| organizations_raw | completeness_state              | 1        |
| organizations_raw | completeness_postal_code        | 1        |
| organizations_raw | completeness_country            | 1        |
| providers_raw     | validity_state_ref              | 0.985653 |
| organizations_raw | validity_state_ref              | 0.978747 |
| providers_raw     | suspected_duplicate_row_ratio   | 0.908383 |
| organizations_raw | suspected_duplicate_row_ratio   | 0.852349 |
| affiliations_raw  | completeness_provider_id_source | 1        |
| affiliations_raw  | completeness_org_id_source      | 1        |
| affiliations_raw  | completeness_affiliation_type   | 1        |
| affiliations_raw  | completeness_start_date         | 1        |

