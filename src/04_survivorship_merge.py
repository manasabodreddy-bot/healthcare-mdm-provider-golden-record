"""
Project 1 - Step 5: Survivorship merge and golden record creation for providers.

Reads from SQLite:
  - stg_providers_clean
  - match_decisions

Writes to SQLite:
  - golden_providers
  - provider_xref
  - merge_audit_log

Writes to disk:
  - data/curated/golden_providers.csv
  - data/curated/provider_xref.csv
  - outputs/reports/merge_audit_log.csv
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


INTERIM_DIR = os.path.join("data", "interim")
CURATED_DIR = os.path.join("data", "curated")
REPORT_DIR = os.path.join("outputs", "reports")

DB_PATH = os.path.join(INTERIM_DIR, "mdm.db")

OUT_GOLDEN = os.path.join(CURATED_DIR, "golden_providers.csv")
OUT_XREF = os.path.join(CURATED_DIR, "provider_xref.csv")
OUT_AUDIT = os.path.join(REPORT_DIR, "merge_audit_log.csv")


# Trust ranking (enterprise realistic): CRM_A slightly more reliable than CRM_B
TRUST_RANK = {"CRM_A": 2, "CRM_B": 1}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def to_sql(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def parse_dt(x: object) -> pd.Timestamp:
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def non_null_count(row: pd.Series, cols: List[str]) -> int:
    return int(sum([0 if pd.isna(row.get(c)) or str(row.get(c)).strip()=="" else 1 for c in cols]))


def pick_best_record(records: pd.DataFrame) -> pd.Series:
    """
    Choose a canonical "base" record to anchor survivorship.
    Rule: highest trust rank, then most recent updated_at, then most complete address.
    """
    tmp = records.copy()
    tmp["trust_rank"] = tmp["source_system"].map(lambda s: TRUST_RANK.get(str(s), 0))
    tmp["updated_at_ts"] = tmp["updated_at"].map(parse_dt)

    addr_cols = ["address_line1", "city", "state", "postal_code", "country"]
    tmp["addr_completeness"] = tmp.apply(lambda r: non_null_count(r, addr_cols), axis=1)

    tmp = tmp.sort_values(["trust_rank", "updated_at_ts", "addr_completeness"], ascending=[False, False, False])
    return tmp.iloc[0]


def choose_field(records: pd.DataFrame, field: str) -> Tuple[object, str]:
    """
    Survivorship for one field:
      1) Prefer non-null values
      2) Prefer higher trust rank
      3) Prefer latest updated_at
    Returns: (chosen_value, rule_description)
    """
    tmp = records.copy()
    tmp["trust_rank"] = tmp["source_system"].map(lambda s: TRUST_RANK.get(str(s), 0))
    tmp["updated_at_ts"] = tmp["updated_at"].map(parse_dt)

    # Keep only candidates with a usable value
    tmp[field] = tmp[field].apply(lambda x: pd.NA if pd.isna(x) or str(x).strip()=="" else x)
    tmp2 = tmp[tmp[field].notna()].copy()

    if len(tmp2) == 0:
        return (pd.NA, f"{field}:no_non_null")

    tmp2 = tmp2.sort_values(["trust_rank", "updated_at_ts"], ascending=[False, False])
    chosen = tmp2.iloc[0][field]
    rule = f"{field}:trust_then_latest"
    return (chosen, rule)


def conflicts_for_field(records: pd.DataFrame, field: str) -> List[str]:
    vals = records[field].dropna().astype(str).map(lambda x: x.strip()).unique().tolist()
    vals = [v for v in vals if v != ""]
    # If more than one distinct non-empty value exists => conflict
    return vals if len(vals) > 1 else []


def build_cluster_graph(auto_matches: pd.DataFrame) -> Dict[str, set]:
    """
    Build an undirected graph of provider_id_source nodes connected by auto_match edges.
    Return adjacency list: node -> set(neighbors)
    """
    adj: Dict[str, set] = {}
    for _, r in auto_matches.iterrows():
        a = str(r["provider_id_source_1"])
        b = str(r["provider_id_source_2"])
        if a not in adj:
            adj[a] = set()
        if b not in adj:
            adj[b] = set()
        adj[a].add(b)
        adj[b].add(a)
    return adj


def connected_components(nodes: List[str], adj: Dict[str, set]) -> List[List[str]]:
    """
    Find connected components in the match graph.
    Each component becomes one golden record.
    """
    seen = set()
    components: List[List[str]] = []

    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp = []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for nb in adj.get(x, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)

    return components


def main() -> None:
    ensure_dir(CURATED_DIR)
    ensure_dir(REPORT_DIR)

    conn = sqlite3.connect(DB_PATH)
    try:
        prov = read_sql(conn, "SELECT * FROM stg_providers_clean")
        decisions = read_sql(conn, "SELECT * FROM match_decisions")
    finally:
        conn.close()

    # Use only auto_match edges for merges
    auto_matches = decisions[decisions["decision"] == "auto_match"].copy()

    # Build graph and components
    all_nodes = prov["provider_id_source"].astype(str).tolist()
    adj = build_cluster_graph(auto_matches)
    comps = connected_components(all_nodes, adj)

    golden_rows: List[Dict[str, object]] = []
    xref_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    # Fields to survive
    survive_fields = [
        "npi", "first_name", "last_name", "gender", "specialty",
        "email", "phone",
        "address_line1", "city", "state", "postal_code", "country",
        "record_status"
    ]

    # For each component, create golden record
    for idx, comp in enumerate(comps, start=1):
        golden_id = f"GPROV{idx:07d}"
        records = prov[prov["provider_id_source"].astype(str).isin(comp)].copy()

        # Anchor base record
        base = pick_best_record(records)

        chosen = {"golden_provider_id": golden_id}
        chosen["cluster_size"] = int(len(records))
        chosen["source_system_anchor"] = base.get("source_system")
        chosen["updated_at_anchor"] = base.get("updated_at")

        # Survivorship field-by-field with audit
        rules_applied = []
        conflicts = {}

        for f in survive_fields:
            value, rule = choose_field(records, f)
            chosen[f] = value
            rules_applied.append(rule)
            cvals = conflicts_for_field(records, f)
            if cvals:
                conflicts[f] = cvals[:5]  # cap for audit readability

        golden_rows.append(chosen)

        # XREF mapping: each source record maps to golden
        for _, r in records.iterrows():
            xref_rows.append({
                "golden_provider_id": golden_id,
                "source_system": r.get("source_system"),
                "provider_id_source": r.get("provider_id_source"),
                "npi": r.get("npi"),
                "updated_at": r.get("updated_at"),
                "mapped_at": datetime.now().isoformat(timespec="seconds"),
            })

        # Audit log per golden
        audit_rows.append({
            "golden_provider_id": golden_id,
            "cluster_size": int(len(records)),
            "source_records": "|".join(sorted(records["provider_id_source"].astype(str).tolist())),
            "rules_applied": "|".join(rules_applied),
            "conflicts_detected": str(conflicts) if conflicts else "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

    df_golden = pd.DataFrame(golden_rows)
    df_xref = pd.DataFrame(xref_rows)
    df_audit = pd.DataFrame(audit_rows)

    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        to_sql(conn, df_golden, "golden_providers")
        to_sql(conn, df_xref, "provider_xref")
        to_sql(conn, df_audit, "merge_audit_log")
    finally:
        conn.close()

    # Export to files
    df_golden.to_csv(OUT_GOLDEN, index=False)
    df_xref.to_csv(OUT_XREF, index=False)
    df_audit.to_csv(OUT_AUDIT, index=False)

    # Print summary
    merged_clusters = (df_golden["cluster_size"] > 1).sum()
    print("✅ Step 5 complete")
    print(f"Golden providers:         {len(df_golden):,}")
    print(f"Clusters merged (>1):     {int(merged_clusters):,}")
    print(f"Golden CSV:              {OUT_GOLDEN}")
    print(f"XREF CSV:                {OUT_XREF}")
    print(f"Merge audit log CSV:     {OUT_AUDIT}")
    print("Next: Step 6 quality before/after metrics + final project docs.")


if __name__ == "__main__":
    main()
