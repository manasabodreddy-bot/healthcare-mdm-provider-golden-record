"""
Project 1 - Step 2: Load raw CSVs into SQLite and produce a baseline
data profiling + data quality report for MDM stewardship.

Outputs:
  - data/interim/mdm.db  (tables: stg_providers, stg_organizations, stg_affiliations, ref_country_state)
  - outputs/reports/raw_profile.md
  - outputs/reports/raw_quality_baseline.csv
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from tabulate import tabulate


RAW_DIR = os.path.join("data", "raw")
INTERIM_DIR = os.path.join("data", "interim")
DB_PATH = os.path.join(INTERIM_DIR, "mdm.db")

REPORT_DIR = os.path.join("outputs", "reports")
PROFILE_MD = os.path.join(REPORT_DIR, "raw_profile.md")
BASELINE_CSV = os.path.join(REPORT_DIR, "raw_quality_baseline.csv")

# Critical fields (typical stewardship focus)
PROVIDER_CRITICAL_FIELDS = ["first_name", "last_name", "address_line1", "city", "state", "postal_code", "country"]
PROVIDER_ID_FIELDS = ["provider_id_source", "npi", "email", "phone"]

ORG_CRITICAL_FIELDS = ["org_name", "org_type", "address_line1", "city", "state", "postal_code", "country"]
AFFIL_FIELDS = ["provider_id_source", "org_id_source", "affiliation_type", "start_date"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def to_sqlite(df: pd.DataFrame, conn: sqlite3.Connection, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def null_profile(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        nulls = df[c].isna().sum() if c in df.columns else n
        rows.append({"column": c, "null_count": int(nulls), "null_pct": (nulls / n) if n else 0.0})
    return pd.DataFrame(rows)


def basic_counts(df: pd.DataFrame, entity_name: str) -> Dict[str, object]:
    return {
        "entity": entity_name,
        "row_count": int(len(df)),
        "distinct_source_systems": int(df["source_system"].nunique()) if "source_system" in df.columns else None,
    }


def state_valid_ratio(df: pd.DataFrame, ref_states: pd.DataFrame) -> float:
    valid_pairs = set(zip(ref_states["country"], ref_states["state_code"]))
    if "country" not in df.columns or "state" not in df.columns:
        return 0.0
    valid = df.apply(lambda r: (r["country"], r["state"]) in valid_pairs, axis=1).mean()
    return float(valid)


def suspected_provider_duplicates(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Simple duplicate heuristic for raw baseline:
      duplicates on (last_name, postal_code, city) within same country
    This is not final matching; it's just a baseline signal.
    """
    cols = ["last_name", "postal_code", "city", "country"]
    for c in cols:
        if c not in df.columns:
            return 0, 0.0
    grp = df[cols].astype(str).fillna("").groupby(cols).size()
    dup_groups = grp[grp > 1]
    dup_rows = int(dup_groups.sum())  # total rows in duplicate groups
    dup_ratio = dup_rows / len(df) if len(df) else 0.0
    return dup_rows, float(dup_ratio)


def suspected_org_duplicates(df: pd.DataFrame) -> Tuple[int, float]:
    cols = ["org_name", "postal_code", "city", "country"]
    for c in cols:
        if c not in df.columns:
            return 0, 0.0
    grp = df[cols].astype(str).fillna("").groupby(cols).size()
    dup_groups = grp[grp > 1]
    dup_rows = int(dup_groups.sum())
    dup_ratio = dup_rows / len(df) if len(df) else 0.0
    return dup_rows, float(dup_ratio)


def uniqueness_ratio(series: pd.Series) -> float:
    s = series.dropna().astype(str)
    if len(s) == 0:
        return 0.0
    return float(s.nunique() / len(s))


def write_report(
    prov: pd.DataFrame,
    org: pd.DataFrame,
    aff: pd.DataFrame,
    ref_states: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")

    prov_counts = basic_counts(prov, "providers_raw")
    org_counts = basic_counts(org, "organizations_raw")
    aff_counts = basic_counts(aff, "affiliations_raw")

    prov_nulls = null_profile(prov, PROVIDER_CRITICAL_FIELDS + PROVIDER_ID_FIELDS)
    org_nulls = null_profile(org, ORG_CRITICAL_FIELDS)
    aff_nulls = null_profile(aff, AFFIL_FIELDS)

    prov_state_valid = state_valid_ratio(prov, ref_states)
    org_state_valid = state_valid_ratio(org, ref_states)

    prov_dup_rows, prov_dup_ratio = suspected_provider_duplicates(prov)
    org_dup_rows, org_dup_ratio = suspected_org_duplicates(org)

    # Uniqueness ratios for key identifiers (higher is better)
    npi_unique = uniqueness_ratio(prov["npi"]) if "npi" in prov.columns else 0.0
    email_unique = uniqueness_ratio(prov["email"]) if "email" in prov.columns else 0.0
    phone_unique = uniqueness_ratio(prov["phone"]) if "phone" in prov.columns else 0.0

    md = []
    md.append(f"# Raw Data Profiling Report (MDM Stewardship Baseline)\n")
    md.append(f"Generated at: **{now}**\n")

    md.append("## Dataset Overview\n")
    overview = pd.DataFrame([prov_counts, org_counts, aff_counts])
    md.append(tabulate(overview, headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    md.append("## Provider (HCP) – Null / Missingness Profile\n")
    prov_nulls_fmt = prov_nulls.copy()
    prov_nulls_fmt["null_pct"] = prov_nulls_fmt["null_pct"].map(pct)
    md.append(tabulate(prov_nulls_fmt, headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    md.append("## Organization (HCO) – Null / Missingness Profile\n")
    org_nulls_fmt = org_nulls.copy()
    org_nulls_fmt["null_pct"] = org_nulls_fmt["null_pct"].map(pct)
    md.append(tabulate(org_nulls_fmt, headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    md.append("## Affiliations – Null / Missingness Profile\n")
    aff_nulls_fmt = aff_nulls.copy()
    aff_nulls_fmt["null_pct"] = aff_nulls_fmt["null_pct"].map(pct)
    md.append(tabulate(aff_nulls_fmt, headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    md.append("## Baseline Validity Checks\n")
    validity_rows = [
        {"check": "Provider state validity (country,state in reference)", "value": pct(prov_state_valid)},
        {"check": "Organization state validity (country,state in reference)", "value": pct(org_state_valid)},
        {"check": "Provider suspected duplicate rows (heuristic)", "value": f"{prov_dup_rows:,} ({pct(prov_dup_ratio)})"},
        {"check": "Organization suspected duplicate rows (heuristic)", "value": f"{org_dup_rows:,} ({pct(org_dup_ratio)})"},
        {"check": "Provider NPI uniqueness ratio (non-null)", "value": pct(npi_unique)},
        {"check": "Provider email uniqueness ratio (non-null)", "value": pct(email_unique)},
        {"check": "Provider phone uniqueness ratio (non-null)", "value": pct(phone_unique)},
    ]
    md.append(tabulate(pd.DataFrame(validity_rows), headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    md.append("## Quality Baseline (for Before/After comparisons)\n")
    md.append("Saved to: `outputs/reports/raw_quality_baseline.csv`\n\n")
    md.append(tabulate(baseline, headers="keys", tablefmt="github", showindex=False))
    md.append("\n")

    ensure_dir(REPORT_DIR)
    with open(PROFILE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def build_baseline_metrics(
    prov: pd.DataFrame,
    org: pd.DataFrame,
    aff: pd.DataFrame,
    ref_states: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct a tidy baseline metrics table.
    These metrics will be compared again after standardization + golden record merge.
    """
    metrics: List[Dict[str, object]] = []

    # Provider completeness for critical fields
    for c in PROVIDER_CRITICAL_FIELDS + PROVIDER_ID_FIELDS:
        if c in prov.columns:
            metrics.append({
                "entity": "providers_raw",
                "metric": f"completeness_{c}",
                "value": float(1.0 - prov[c].isna().mean()),
            })

    # Organization completeness
    for c in ORG_CRITICAL_FIELDS:
        if c in org.columns:
            metrics.append({
                "entity": "organizations_raw",
                "metric": f"completeness_{c}",
                "value": float(1.0 - org[c].isna().mean()),
            })

    # State validity
    metrics.append({
        "entity": "providers_raw",
        "metric": "validity_state_ref",
        "value": state_valid_ratio(prov, ref_states),
    })
    metrics.append({
        "entity": "organizations_raw",
        "metric": "validity_state_ref",
        "value": state_valid_ratio(org, ref_states),
    })

    # Duplicate heuristics
    prov_dup_rows, prov_dup_ratio = suspected_provider_duplicates(prov)
    org_dup_rows, org_dup_ratio = suspected_org_duplicates(org)
    metrics.append({"entity": "providers_raw", "metric": "suspected_duplicate_row_ratio", "value": prov_dup_ratio})
    metrics.append({"entity": "organizations_raw", "metric": "suspected_duplicate_row_ratio", "value": org_dup_ratio})

    # Affiliations completeness (key link fields)
    for c in AFFIL_FIELDS:
        if c in aff.columns:
            metrics.append({
                "entity": "affiliations_raw",
                "metric": f"completeness_{c}",
                "value": float(1.0 - aff[c].isna().mean()),
            })

    df = pd.DataFrame(metrics)
    # Format in report, but store as numeric in CSV
    return df


def main() -> None:
    ensure_dir(INTERIM_DIR)
    ensure_dir(REPORT_DIR)

    prov_path = os.path.join(RAW_DIR, "providers_raw.csv")
    org_path = os.path.join(RAW_DIR, "organizations_raw.csv")
    aff_path = os.path.join(RAW_DIR, "affiliations_raw.csv")
    ref_path = os.path.join(RAW_DIR, "reference_country_state.csv")

    prov = load_csv(prov_path)
    org = load_csv(org_path)
    aff = load_csv(aff_path)
    ref_states = load_csv(ref_path)

    # Load into SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        to_sqlite(prov.drop(columns=[c for c in ["entity_key"] if c in prov.columns]), conn, "stg_providers")
        to_sqlite(org.drop(columns=[c for c in ["entity_key"] if c in org.columns]), conn, "stg_organizations")
        to_sqlite(aff, conn, "stg_affiliations")
        to_sqlite(ref_states, conn, "ref_country_state")
    finally:
        conn.close()

    baseline = build_baseline_metrics(prov, org, aff, ref_states)
    baseline.to_csv(BASELINE_CSV, index=False)

    write_report(prov, org, aff, ref_states, baseline)

    print("✅ Step 2 complete")
    print(f"SQLite DB: {DB_PATH}")
    print(f"Report:    {PROFILE_MD}")
    print(f"Baseline:  {BASELINE_CSV}")
    print("Next: Step 3 standardization & cleansing.")


if __name__ == "__main__":
    main()
