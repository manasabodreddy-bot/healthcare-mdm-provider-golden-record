"""
Project 1 - Step 6: Compute before/after data quality metrics and write a summary.

Reads from SQLite:
  - stg_providers
  - stg_providers_clean
  - golden_providers
  - match_decisions

Reads baseline file:
  - outputs/reports/raw_quality_baseline.csv

Writes:
  - outputs/reports/quality_before_after.csv
  - outputs/reports/quality_summary.md
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tabulate import tabulate


DB_PATH = os.path.join("data", "interim", "mdm.db")
BASELINE_CSV = os.path.join("outputs", "reports", "raw_quality_baseline.csv")

REPORT_DIR = os.path.join("outputs", "reports")
OUT_CSV = os.path.join(REPORT_DIR, "quality_before_after.csv")
OUT_MD = os.path.join(REPORT_DIR, "quality_summary.md")

# Focus fields that matter for MDM stewardship
CRITICAL_PROVIDER_FIELDS = [
    "npi", "first_name", "last_name", "email", "phone",
    "address_line1", "city", "state", "postal_code", "country",
    "record_status"
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def completeness(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    s = df[col]
    # treat empty strings as missing too
    missing = s.isna() | (s.astype(str).str.strip() == "")
    return float(1.0 - missing.mean())


def uniqueness_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    s = df[col].dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return 0.0
    return float(s.nunique() / len(s))


def suspected_duplicate_ratio(df: pd.DataFrame) -> float:
    """
    Simple heuristic (same as Step 2 baseline logic), but applied to any table:
    duplicates on (last_name, postal_code, city, country)
    """
    needed = ["last_name", "postal_code", "city", "country"]
    for c in needed:
        if c not in df.columns:
            return 0.0
    grp = (
        df[needed]
        .astype(str)
        .fillna("")
        .groupby(needed)
        .size()
    )
    dup_groups = grp[grp > 1]
    dup_rows = int(dup_groups.sum())
    return float(dup_rows / len(df)) if len(df) else 0.0


def build_metrics(df: pd.DataFrame, entity_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rows.append({"entity": entity_name, "metric": "row_count", "value": float(len(df))})

    # completeness for critical fields
    for c in CRITICAL_PROVIDER_FIELDS:
        if c in df.columns:
            rows.append({"entity": entity_name, "metric": f"completeness_{c}", "value": completeness(df, c)})

    # uniqueness signals (not absolute “good”, but useful)
    for c in ["npi", "email", "phone"]:
        if c in df.columns:
            rows.append({"entity": entity_name, "metric": f"uniqueness_ratio_{c}", "value": uniqueness_ratio(df, c)})

    # duplicate heuristic
    rows.append({"entity": entity_name, "metric": "suspected_duplicate_row_ratio", "value": suspected_duplicate_ratio(df)})
    return rows


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main() -> None:
    ensure_dir(REPORT_DIR)

    # Load baseline (raw metrics captured earlier)
    if not os.path.exists(BASELINE_CSV):
        raise FileNotFoundError(f"Missing baseline file: {BASELINE_CSV} (run Step 2 first)")
    baseline = pd.read_csv(BASELINE_CSV)

    conn = sqlite3.connect(DB_PATH)
    try:
        raw = read_sql(conn, "SELECT * FROM stg_providers")
        clean = read_sql(conn, "SELECT * FROM stg_providers_clean")
        golden = read_sql(conn, "SELECT * FROM golden_providers")
        decisions = read_sql(conn, "SELECT * FROM match_decisions")
    finally:
        conn.close()

    # Compute new metrics for clean + golden (and raw again for consistency)
    rows = []
    rows.extend(build_metrics(raw, "providers_raw_current"))
    rows.extend(build_metrics(clean, "providers_clean"))
    rows.extend(build_metrics(golden, "golden_providers"))

    metrics = pd.DataFrame(rows)

    # Add operational merge metrics
    # - cluster_size is present in golden_providers
    merged_clusters = int((golden["cluster_size"] > 1).sum()) if "cluster_size" in golden.columns else 0
    total_clusters = int(len(golden))
    total_source_records = int(len(clean))  # clean records represent source records after standardization

    ops = [
        {"entity": "operations", "metric": "total_source_records_clean", "value": float(total_source_records)},
        {"entity": "operations", "metric": "golden_record_count", "value": float(total_clusters)},
        {"entity": "operations", "metric": "clusters_merged_gt1", "value": float(merged_clusters)},
    ]

    # manual review queue size
    if "decision" in decisions.columns:
        manual_cnt = int((decisions["decision"] == "manual_review").sum())
        auto_cnt = int((decisions["decision"] == "auto_match").sum())
        no_cnt = int((decisions["decision"] == "no_match").sum())
    else:
        manual_cnt = auto_cnt = no_cnt = 0

    ops.extend([
        {"entity": "operations", "metric": "match_edges_auto_match", "value": float(auto_cnt)},
        {"entity": "operations", "metric": "match_edges_manual_review", "value": float(manual_cnt)},
        {"entity": "operations", "metric": "match_edges_no_match", "value": float(no_cnt)},
    ])

    ops_df = pd.DataFrame(ops)

    full = pd.concat([metrics, ops_df], ignore_index=True)

    # Pivot for before/after table (only key metrics)
    key_metrics = [
        "row_count",
        "suspected_duplicate_row_ratio",
        "completeness_npi",
        "completeness_email",
        "completeness_phone",
        "completeness_state",
        "completeness_postal_code",
        "uniqueness_ratio_npi",
        "uniqueness_ratio_email",
        "uniqueness_ratio_phone",
    ]

    subset = full[full["metric"].isin(key_metrics)].copy()
    pivot = subset.pivot_table(index="metric", columns="entity", values="value", aggfunc="first").reset_index()

    # Save a single tidy CSV (good for portfolio + resume proof)
    pivot.to_csv(OUT_CSV, index=False)

    # Build summary markdown
    md = []
    md.append("# Data Quality Summary (Before vs After)\n")
    md.append(f"Generated at: **{datetime.now().isoformat(timespec='seconds')}**\n")
    md.append("## Key Outcomes\n")
    md.append(f"- Clean source records: **{total_source_records:,}**\n")
    md.append(f"- Golden records created: **{total_clusters:,}**\n")
    md.append(f"- Clusters merged (>1): **{merged_clusters:,}**\n")
    md.append(f"- Manual review queue (pairs): **{manual_cnt:,}**\n")

    md.append("\n## Before/After Metrics Table\n")
    # Format percentages for ratio metrics for readability
    pretty = pivot.copy()
    for m in pretty["metric"].tolist():
        if "completeness_" in m or "ratio" in m:
            for col in pretty.columns:
                if col != "metric" and pd.api.types.is_numeric_dtype(pretty[col]):
                    pretty[col] = pretty[col].map(lambda x: pct(x) if pd.notna(x) else x)
    md.append(tabulate(pretty, headers="keys", tablefmt="github", showindex=False))

    md.append("\n\n## Notes (Stewardship Interpretation)\n")
    md.append("- **Completeness** reflects usable (non-null, non-empty) fields after standardization and survivorship.\n")
    md.append("- **Suspected duplicate ratio** is a heuristic signal; the true dedup is represented by golden record reduction.\n")
    md.append("- Manual review queue is expected in operational MDM; those pairs require steward validation.\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("✅ Step 6 complete")
    print(f"Before/After CSV: {OUT_CSV}")
    print(f"Summary report:   {OUT_MD}")


if __name__ == "__main__":
    main()
