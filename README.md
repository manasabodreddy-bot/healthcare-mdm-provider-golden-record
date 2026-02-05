# Healthcare Master Data Management – Provider Golden Record

## Overview
This project demonstrates an end-to-end **Master Data Management (MDM) stewardship workflow** for healthcare provider (HCP) data.  
It simulates enterprise-scale commercial data operations commonly found in global life sciences organizations.

The solution covers:
- data ingestion and profiling
- standardization and validation
- de-duplication using match scoring
- survivorship-based golden record creation
- cross-reference (XREF) mapping
- audit logging and data quality measurement

The project is designed to mirror **real-world MDM operations**, not academic data cleaning.

---

## Business Problem
Healthcare provider data often arrives from multiple source systems with:
- inconsistent identifiers
- missing or invalid attributes
- duplicate provider records
- conflicting values across systems

Without MDM stewardship, these issues negatively impact:
- analytics and reporting
- downstream CRM systems
- compliance and governance processes

---

## Solution Architecture
Raw Provider Data
↓
Baseline Profiling & Quality Metrics
↓
Standardization & Validation
↓
Match Candidate Generation (Blocking + Scoring)
↓
Auto-Merge (Deterministic Matches)
↓
Manual Review Queue (Stewardship)
↓
Survivorship & Golden Record
↓
XREF Mapping + Audit Log

---

## Key Outputs
- **Golden Provider Records** (`golden_providers`)
- **Source-to-Golden Cross Reference** (`provider_xref`)
- **Merge Audit Log** (traceability & governance)
- **Manual Review Queue** for steward decisions
- **Before vs After Data Quality Metrics**

---

## Technology Stack
- Python (pandas, numpy)
- SQLite (simulated MDM platform)
- SQL
- Git (version control)
- Markdown documentation

---

## Key Results (Sample Run)
- Clean source records: ~1,800+
- Golden providers created: reduced dataset
- Merged clusters (>1 record): multiple
- Manual review queue: ~1,800 match pairs
- Measurable improvements in completeness and duplication metrics

---

## Repository Structure
data/
raw/ # synthetic source data
interim/ # standardized staging data
curated/ # golden records & XREF

src/
00_generate_data.py
01_profile_raw_data.py
02_standardize.py
03_match_candidates.py
04_survivorship_merge.py
05_quality_metrics.py

docs/
stewardship_sop.md
business_rules.md
data_dictionary.md

outputs/
reports/
exceptions/


---

## How to Run
```bash
python src/00_generate_data.py
python src/01_profile_raw_data.py
python src/02_standardize.py
python src/03_match_candidates.py
python src/04_survivorship_merge.py
python src/05_quality_metrics.py

## Notes
- Manual review queues are expected in real MDM operations.
- Auto-merges are restricted to deterministic, high-confidence matches.
- All merges are auditable and reversible.
- This project focuses on data stewardship discipline, not just code.
