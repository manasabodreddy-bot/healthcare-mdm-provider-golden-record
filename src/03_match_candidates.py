"""
Project 1 - Step 4: Generate match candidates for provider de-duplication
using blocking + scoring (weight-of-evidence).

Reads from SQLite:
  - stg_providers_clean

Writes to SQLite:
  - match_candidates
  - match_decisions

Also writes:
  - outputs/reports/match_candidates_top.csv
  - outputs/exceptions/match_review_queue.csv
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
from tabulate import tabulate
from difflib import SequenceMatcher


INTERIM_DIR = os.path.join("data", "interim")
DB_PATH = os.path.join(INTERIM_DIR, "mdm.db")

REPORT_DIR = os.path.join("outputs", "reports")
EXC_DIR = os.path.join("outputs", "exceptions")

TOP_CSV = os.path.join(REPORT_DIR, "match_candidates_top.csv")
REVIEW_CSV = os.path.join(EXC_DIR, "match_review_queue.csv")

# Thresholds (tuneable but realistic)
AUTO_MATCH_THRESHOLD = 0.95
REVIEW_LOW = 0.65
REVIEW_HIGH = 0.95


# Blocking parameters
MAX_BLOCK_SIZE = 300  # safety to avoid blow-ups in synthetic large blocks


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def to_sql(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def sim(a: object, b: object) -> float:
    """String similarity using SequenceMatcher ratio; returns 0..1."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    a_s = str(a).strip().lower()
    b_s = str(b).strip().lower()
    if not a_s or not b_s:
        return 0.0
    return float(SequenceMatcher(None, a_s, b_s).ratio())


def exact(a: object, b: object) -> int:
    if pd.isna(a) or pd.isna(b):
        return 0
    return 1 if str(a).strip().lower() == str(b).strip().lower() else 0


def safe_concat(*parts: object) -> str:
    vals = []
    for p in parts:
        if pd.isna(p):
            vals.append("")
        else:
            vals.append(str(p).strip().lower())
    return " ".join([v for v in vals if v])


def weighted_score(row: Dict[str, object]) -> float:
    """
    Weight-of-evidence scoring tuned for MDM stewardship:
    - Strong identifiers (NPI/email) dominate when present
    - When identifiers are missing, name+address+phone can still reach review range
    """
    npi_match = row["npi_match"]
    email_match = row["email_match"]
    phone_match = row["phone_match"]

    name_sim = row["name_sim"]
    addr_sim = row["addr_sim"]
    city_sim = row["city_sim"]
    postal_match = row["postal_match"]
    state_match = row["state_match"]

    # Base weights
    score = (
        0.33 * npi_match +
        0.22 * email_match +
        0.15 * phone_match +
        0.18 * name_sim +
        0.09 * addr_sim +
        0.02 * city_sim +
        0.005 * postal_match +
        0.005 * state_match
    )

    # Boost rule: if both NPI and email are missing/non-matching,
    # but name+address are strong, push into manual review band.
    if npi_match == 0 and email_match == 0:
        if name_sim >= 0.88 and addr_sim >= 0.82:
            score += 0.06
        elif name_sim >= 0.84 and addr_sim >= 0.75 and phone_match == 1:
            score += 0.05

    return float(max(0.0, min(1.0, score)))



def decision_logic(npi_match: int, email_match: int, score: float, name_sim: float, addr_sim: float, phone_match: int) -> str:
    """
    Decision logic tuned for stewardship realism:
    - NPI exact match is safe to auto-merge
    - Email match alone is NOT auto-match (dirty/duplicate emails exist)
    - Strong name+address signals go to manual review even if score is moderate
    """
    # Deterministic safe rule
    if npi_match == 1:
        return "auto_match"

    # Stewardship review override (creates realistic review bucket)
    # Strong identity signals but missing strong identifiers => manual review
    if (name_sim >= 0.88 and addr_sim >= 0.80) or (name_sim >= 0.84 and addr_sim >= 0.74 and phone_match == 1):
        return "manual_review"

    # Score-based classification
    if score >= AUTO_MATCH_THRESHOLD:
        return "auto_match"
    if REVIEW_LOW <= score < REVIEW_HIGH:
        return "manual_review"
    return "no_match"



def build_blocks(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Blocking reduces candidate pairs dramatically.

    Blocks:
      A) email exact (if present)
      B) npi exact (if present)
      C) last_name + postal_code + country
      D) last_name + city + country  (fallback)

    Returns dict: block_key -> list of row indices
    """
    blocks: Dict[str, List[int]] = {}

    def add_block(key: str, idx: int) -> None:
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(idx)

    for idx, r in df.iterrows():
        country = r.get("country")
        last_name = r.get("last_name")
        postal = r.get("postal_code")
        city = r.get("city")
        email = r.get("email")
        npi = r.get("npi")

        if not pd.isna(email) and str(email).strip():
            add_block(f"EMAIL::{str(email).strip().lower()}", idx)

        if not pd.isna(npi) and str(npi).strip():
            add_block(f"NPI::{str(npi).strip()}", idx)

        if not pd.isna(last_name) and not pd.isna(postal) and not pd.isna(country):
            add_block(f"LNPC::{str(last_name).strip().lower()}::{str(postal).strip()}::{str(country).strip().upper()}", idx)

        if not pd.isna(last_name) and not pd.isna(city) and not pd.isna(country):
            add_block(f"LNCC::{str(last_name).strip().lower()}::{str(city).strip().lower()}::{str(country).strip().upper()}", idx)

        # First-initial + last name + postal + country
        # Helps catch cases like "M." vs "Manasa"
        first = r.get("first_name")
        if (
            not pd.isna(first)
            and not pd.isna(last_name)
            and not pd.isna(postal)
            and not pd.isna(country)
        ):
            fi = str(first).strip().lower()[0] if str(first).strip() else ""
            if fi:
                add_block(
                    f"FI_LNPC::{fi}::{str(last_name).strip().lower()}::{str(postal).strip()}::{str(country).strip().upper()}",
                    idx
                )
    return blocks



def main() -> None:
    ensure_dir(REPORT_DIR)
    ensure_dir(EXC_DIR)

    conn = sqlite3.connect(DB_PATH)
    try:
        prov = read_sql(conn, "SELECT * FROM stg_providers_clean")
    finally:
        conn.close()

    if "provider_id_source" not in prov.columns:
        raise ValueError("stg_providers_clean missing provider_id_source")

    # Only compare within the same country; this reduces false matches
    # We'll still allow blocks to generate pairs; score uses country implicitly via addr/city/postal.
    blocks = build_blocks(prov)

    pairs_seen = set()
    candidate_rows: List[Dict[str, object]] = []

    # Generate candidate pairs from blocks
    for bkey, indices in blocks.items():
        if len(indices) < 2:
            continue
        if len(indices) > MAX_BLOCK_SIZE:
            # Skip overly-large blocks to keep compute bounded (should be rare in synthetic)
            continue

        for i, j in combinations(indices, 2):
            # Unique pair id (order-independent)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in pairs_seen:
                continue
            pairs_seen.add((a, b))

            r1 = prov.loc[a]
            r2 = prov.loc[b]

            # Basic guard: if both countries exist and mismatch, skip
            c1 = r1.get("country")
            c2 = r2.get("country")
            if not pd.isna(c1) and not pd.isna(c2) and str(c1).upper() != str(c2).upper():
                continue

            # Compute evidence signals
            name1 = safe_concat(r1.get("first_name"), r1.get("last_name"))
            name2 = safe_concat(r2.get("first_name"), r2.get("last_name"))

            row = {
                "provider_id_source_1": r1["provider_id_source"],
                "provider_id_source_2": r2["provider_id_source"],
                "source_system_1": r1.get("source_system"),
                "source_system_2": r2.get("source_system"),

                "npi_1": r1.get("npi"),
                "npi_2": r2.get("npi"),
                "email_1": r1.get("email"),
                "email_2": r2.get("email"),
                "phone_1": r1.get("phone"),
                "phone_2": r2.get("phone"),

                "name_1": name1,
                "name_2": name2,
                "address_1": r1.get("address_line1"),
                "address_2": r2.get("address_line1"),
                "city_1": r1.get("city"),
                "city_2": r2.get("city"),
                "state_1": r1.get("state"),
                "state_2": r2.get("state"),
                "postal_1": r1.get("postal_code"),
                "postal_2": r2.get("postal_code"),
                "country": r1.get("country") if not pd.isna(r1.get("country")) else r2.get("country"),

                # evidence
                "npi_match": exact(r1.get("npi"), r2.get("npi")),
                "email_match": exact(r1.get("email"), r2.get("email")),
                "phone_match": exact(r1.get("phone"), r2.get("phone")),
                "postal_match": exact(r1.get("postal_code"), r2.get("postal_code")),
                "state_match": exact(r1.get("state"), r2.get("state")),
                "name_sim": sim(name1, name2),
                "addr_sim": sim(r1.get("address_line1"), r2.get("address_line1")),
                "city_sim": sim(r1.get("city"), r2.get("city")),
                "block_key": bkey,
                "scored_at": datetime.now().isoformat(timespec="seconds"),
            }

            row["match_score"] = weighted_score(row)
            row["decision"] = decision_logic(
    row["npi_match"],
    row["email_match"],
    row["match_score"],
    row["name_sim"],
    row["addr_sim"],
    row["phone_match"],
)


            candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)

    if len(candidates) == 0:
        raise RuntimeError("No match candidates generated. Check Step 1/3 data quality or blocking rules.")

    # Save detailed candidates (can be large)
    # Keep only key columns in match_candidates table
    match_candidates = candidates[[
        "provider_id_source_1","provider_id_source_2",
        "source_system_1","source_system_2",
        "npi_1","npi_2","email_1","email_2","phone_1","phone_2",
        "name_sim","addr_sim","city_sim",
        "npi_match","email_match","phone_match","postal_match","state_match",
        "match_score","decision","block_key","scored_at",
    ]].copy()

    # Decisions table for workflow (manual review queue etc.)
    match_decisions = match_candidates[[
        "provider_id_source_1","provider_id_source_2","match_score","decision","block_key","scored_at"
    ]].copy()

    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        to_sql(conn, match_candidates, "match_candidates")
        to_sql(conn, match_decisions, "match_decisions")
    finally:
        conn.close()

    # Export top scored candidates for readability
    top = match_candidates.sort_values("match_score", ascending=False).head(250)
    top.to_csv(TOP_CSV, index=False)

    # Export manual review queue
    review = match_candidates[match_candidates["decision"] == "manual_review"].sort_values("match_score", ascending=False)
    review.to_csv(REVIEW_CSV, index=False)

    # Print summary
    summary = match_candidates["decision"].value_counts().reset_index()
    summary.columns = ["decision", "count"]

    print("✅ Step 4 complete")
    print(f"Candidates generated: {len(match_candidates):,}")
    print(tabulate(summary, headers="keys", tablefmt="github", showindex=False))
    print(f"Top candidates:      {TOP_CSV}")
    print(f"Review queue:        {REVIEW_CSV}")
    print("Next: Step 5 survivorship merge + golden record creation.")


if __name__ == "__main__":
    main()
