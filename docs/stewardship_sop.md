# MDM Stewardship SOP – Provider Master Data

## 1. Purpose
To define the standard operating procedure (SOP) for maintaining high-quality provider master data through standardized stewardship, de-duplication, and governance processes.

---

## 2. Scope
This SOP applies to:
- Healthcare Provider (HCP) master data
- Source systems contributing provider records
- Golden record creation and maintenance
- Data quality monitoring and exception handling

---

## 3. Roles & Responsibilities
**Data Steward / MDM Associate**
- Execute data creation, updates, merges, and de-duplication
- Review manual match exceptions
- Validate data quality issues
- Maintain audit documentation

**Data Engineering / Platform Team**
- Maintain pipelines and infrastructure
- Support defect resolution and enhancements

---

## 4. Process Overview

### 4.1 Data Ingestion
- Load provider data from source systems into staging tables
- Preserve source identifiers for traceability

### 4.2 Baseline Profiling
- Assess completeness, validity, and duplication
- Capture baseline metrics for comparison

### 4.3 Standardization
- Normalize names, emails, phone numbers, and addresses
- Validate geographic attributes against reference data
- Log invalid or missing values

### 4.4 Matching & De-duplication
- Generate match candidates using blocking strategies
- Score candidates using weight-of-evidence logic
- Classify matches into:
  - Auto Match
  - Manual Review
  - No Match

### 4.5 Survivorship & Merge
- Auto-merge only deterministic matches
- Apply survivorship rules per attribute
- Generate golden provider record
- Maintain source-to-golden cross reference

### 4.6 Exception Handling
- Route borderline matches to manual review queue
- Document steward decisions

### 4.7 Audit & Governance
- Record merge rationale and conflicts
- Maintain merge audit log
- Ensure traceability for compliance

---

## 5. Controls & Quality Checks
- Field completeness thresholds
- Duplicate rate monitoring
- Manual review volume tracking
- Merge audit verification

---

## 6. Outputs
- Golden provider dataset
- XREF mappings
- Manual review queue
- Audit logs
- Quality metrics reports