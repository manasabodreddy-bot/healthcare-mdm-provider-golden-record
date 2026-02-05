"""
Project 1 - Step 3: Standardize & cleanse raw master data into clean staging.

Reads from SQLite (data/interim/mdm.db):
  - stg_providers
  - stg_organizations
  - ref_country_state

Writes back to SQLite:
  - stg_providers_clean
  - stg_organizations_clean
  - stg_standardization_issues (optional logging table)

Also writes:
  - data/interim/providers_clean.csv
  - data/interim/organizations_clean.csv
  - outputs/reports/standardization_summary.md
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from tabulate import tabulate


INTERIM_DIR = os.path.join("data", "interim")
DB_PATH = os.path.join(INTERIM_DIR, "mdm.db")

OUT_PROV_CSV = os.path.join(INTERIM_DIR, "providers_clean.csv")
OUT_ORG_CSV = os.path.join(INTERIM_DIR, "organizations_clean.csv")

REPORT_DIR = os.path.join("outputs", "reports")
SUMMARY_MD = os.path.join(REPORT_DIR, "standardization_summary.md")


# -----------------------------
# Business Rules (Standardization)
# -----------------------------
ADDRESS_TOKEN_MAP = [
    (r"\bst\b\.?", "street"),
    (r"\brd\b\.?", "road"),
    (r"\bave\b\.?", "avenue"),
    (r"\bblvd\b\.?", "boulevard"),
    (r"\bln\b\.?", "lane"),
    (r"\bdr\b\.?", "drive"),
    (r"\bn\b\.?", "north"),
    (r"\bs\b\.?", "south"),
    (r"\be\b\.?", "east"),
    (r"\bw\b\.?", "west"),
]

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def to_sql(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def clean_whitespace(x: object) -> object:
    if pd.isna(x):
        return x
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(x: object) -> object:
    if pd.isna(x):
        return x
    s = clean_whitespace(x)
    # Title case, but keep initials like "M."
    if len(str(s)) == 2 and str(s).endswith("."):
        return str(s).upper()
    return str(s).title()


def normalize_email(x: object) -> object:
    if pd.isna(x):
        return x
    s = clean_whitespace(x).lower()
    # If it doesn't match email pattern, mark as null (invalid)
    if not EMAIL_REGEX.match(s):
        return pd.NA
    return s


def normalize_phone(x: object) -> object:
    if pd.isna(x):
        return x
    s = str(x)
    digits = re.sub(r"\D+", "", s)
    # Keep last 10 digits if longer (handles +1 formats)
    if len(digits) >= 10:
        digits = digits[-10:]
    # If too short, treat as missing
    if len(digits) < 10:
        return pd.NA
    return digits


def normalize_postal_code(country: object, postal: object) -> object:
    if pd.isna(postal) or pd.isna(country):
        return pd.NA if pd.isna(postal) else clean_whitespace(postal)
    c = str(country).upper()
    p = clean_whitespace(postal)
    if p is pd.NA or p is None:
        return pd.NA
    p = str(p)
    if c == "US":
        # allow 5-digit or ZIP+4; normalize to 5-digit base as a common practice for matching
        m = re.match(r"^(\d{5})", p)
        return m.group(1) if m else pd.NA
    if c == "IN":
        m = re.match(r"^(\d{6})$", p)
        return m.group(1) if m else pd.NA
    return p


def normalize_address(x: object) -> object:
    if pd.isna(x):
        return x
    s = clean_whitespace(x).lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, repl in ADDRESS_TOKEN_MAP:
        s = re.sub(pattern, repl, s)
    return s


def normalize_country(x: object) -> object:
    if pd.isna(x):
        return x
    s = clean_whitespace(x).upper()
    # basic normalization
    if s in ["USA", "US", "UNITED STATES"]:
        return "US"
    if s in ["INDIA", "IN"]:
        return "IN"
    return s


def normalize_state(country: object, state: object, ref_states: pd.DataFrame) -> Tuple[object, str | None]:
    """
    Returns (normalized_state, issue_reason_if_any)
    - If invalid state for that country: returns NA and logs issue
    """
    if pd.isna(country) or pd.isna(state):
        return (pd.NA, "missing_country_or_state")
    c = str(country).upper()
    s = clean_whitespace(state).upper()

    valid = ref_states[(ref_states["country"] == c) & (ref_states["state_code"] == s)]
    if len(valid) == 0:
        return (pd.NA, "invalid_state_code")
    return (s, None)


def build_issue_row(entity: str, record_id: str, field: str, issue: str, raw_value: object) -> Dict[str, object]:
    return {
        "entity": entity,
        "record_id": record_id,
        "field": field,
        "issue": issue,
        "raw_value": None if pd.isna(raw_value) else str(raw_value),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ensure_dir(REPORT_DIR)

    conn = sqlite3.connect(DB_PATH)
    try:
        prov = read_sql(conn, "SELECT * FROM stg_providers")
        org = read_sql(conn, "SELECT * FROM stg_organizations")
        ref = read_sql(conn, "SELECT * FROM ref_country_state")
    finally:
        conn.close()

    issues: List[Dict[str, object]] = []

    # -----------------------------
    # Providers standardization
    # -----------------------------
    prov_clean = prov.copy()

    # Basic normalization
    prov_clean["source_system"] = prov_clean["source_system"].map(clean_whitespace)
    prov_clean["provider_id_source"] = prov_clean["provider_id_source"].map(clean_whitespace)
    prov_clean["country"] = prov_clean["country"].map(normalize_country)

    prov_clean["first_name"] = prov_clean["first_name"].map(normalize_name)
    prov_clean["last_name"] = prov_clean["last_name"].map(normalize_name)

    prov_clean["email_raw"] = prov_clean["email"]
    prov_clean["email"] = prov_clean["email"].map(normalize_email)
    prov_clean["phone_raw"] = prov_clean["phone"]
    prov_clean["phone"] = prov_clean["phone"].map(normalize_phone)

    # Address
    prov_clean["address_raw"] = prov_clean["address_line1"]
    prov_clean["address_line1"] = prov_clean["address_line1"].map(normalize_address)
    prov_clean["city"] = prov_clean["city"].map(clean_whitespace).map(lambda x: x.title() if not pd.isna(x) else x)

    # Postal depends on country
    prov_clean["postal_raw"] = prov_clean["postal_code"]
    prov_clean["postal_code"] = prov_clean.apply(lambda r: normalize_postal_code(r["country"], r["postal_raw"]), axis=1)

    # State depends on country and ref
    state_norm = prov_clean.apply(lambda r: normalize_state(r["country"], r["state"], ref), axis=1)
    prov_clean["state_issue"] = [x[1] for x in state_norm]
    prov_clean["state"] = [x[0] for x in state_norm]

    # Log provider issues
    for idx, r in prov_clean.iterrows():
        rid = str(r["provider_id_source"])
        if pd.isna(r["email"]) and not pd.isna(r["email_raw"]):
            issues.append(build_issue_row("provider", rid, "email", "invalid_email", r["email_raw"]))
        if pd.isna(r["phone"]) and not pd.isna(r["phone_raw"]):
            issues.append(build_issue_row("provider", rid, "phone", "invalid_phone", r["phone_raw"]))
        if pd.isna(r["postal_code"]) and not pd.isna(r["postal_raw"]):
            issues.append(build_issue_row("provider", rid, "postal_code", "invalid_postal_code", r["postal_raw"]))
        if r.get("state_issue") is not None:
            if r["state_issue"] == "invalid_state_code":
                issues.append(build_issue_row("provider", rid, "state", "invalid_state_code", r["state"]))
            elif r["state_issue"] == "missing_country_or_state":
                issues.append(build_issue_row("provider", rid, "state", "missing_country_or_state", r["state"]))

    # Drop helper issue column
    prov_clean = prov_clean.drop(columns=["state_issue"])

    # -----------------------------
    # Organizations standardization
    # -----------------------------
    org_clean = org.copy()
    org_clean["source_system"] = org_clean["source_system"].map(clean_whitespace)
    org_clean["org_id_source"] = org_clean["org_id_source"].map(clean_whitespace)
    org_clean["country"] = org_clean["country"].map(normalize_country)

    org_clean["org_name_raw"] = org_clean["org_name"]
    org_clean["org_name"] = org_clean["org_name"].map(clean_whitespace).map(lambda x: x.title() if not pd.isna(x) else x)

    org_clean["address_raw"] = org_clean["address_line1"]
    org_clean["address_line1"] = org_clean["address_line1"].map(normalize_address)
    org_clean["city"] = org_clean["city"].map(clean_whitespace).map(lambda x: x.title() if not pd.isna(x) else x)

    org_clean["postal_raw"] = org_clean["postal_code"]
    org_clean["postal_code"] = org_clean.apply(lambda r: normalize_postal_code(r["country"], r["postal_raw"]), axis=1)

    state_norm_o = org_clean.apply(lambda r: normalize_state(r["country"], r["state"], ref), axis=1)
    org_clean["state_issue"] = [x[1] for x in state_norm_o]
    org_clean["state"] = [x[0] for x in state_norm_o]

    # Log org issues
    for idx, r in org_clean.iterrows():
        rid = str(r["org_id_source"])
        if pd.isna(r["postal_code"]) and not pd.isna(r["postal_raw"]):
            issues.append(build_issue_row("organization", rid, "postal_code", "invalid_postal_code", r["postal_raw"]))
        if r.get("state_issue") is not None:
            if r["state_issue"] == "invalid_state_code":
                issues.append(build_issue_row("organization", rid, "state", "invalid_state_code", r["state"]))
            elif r["state_issue"] == "missing_country_or_state":
                issues.append(build_issue_row("organization", rid, "state", "missing_country_or_state", r["state"]))

    org_clean = org_clean.drop(columns=["state_issue"])

    # -----------------------------
    # Save outputs
    # -----------------------------
    # Save to CSV
    prov_clean.to_csv(OUT_PROV_CSV, index=False)
    org_clean.to_csv(OUT_ORG_CSV, index=False)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        to_sql(conn, prov_clean, "stg_providers_clean")
        to_sql(conn, org_clean, "stg_organizations_clean")

        issues_df = pd.DataFrame(issues)
        if len(issues_df) > 0:
            to_sql(conn, issues_df, "stg_standardization_issues")
        else:
            # create empty table for consistency
            to_sql(conn, pd.DataFrame(columns=["entity","record_id","field","issue","raw_value","logged_at"]), "stg_standardization_issues")
    finally:
        conn.close()

    # -----------------------------
    # Summary report
    # -----------------------------
    def completeness(df: pd.DataFrame, col: str) -> float:
        return float(1.0 - df[col].isna().mean()) if col in df.columns else 0.0

    summary_rows = []

    # Providers: key improvements signals
    summary_rows.append({"entity": "providers", "metric": "completeness_email", "value": completeness(prov_clean, "email")})
    summary_rows.append({"entity": "providers", "metric": "completeness_phone", "value": completeness(prov_clean, "phone")})
    summary_rows.append({"entity": "providers", "metric": "completeness_postal_code", "value": completeness(prov_clean, "postal_code")})
    summary_rows.append({"entity": "providers", "metric": "completeness_state", "value": completeness(prov_clean, "state")})

    # Orgs
    summary_rows.append({"entity": "organizations", "metric": "completeness_postal_code", "value": completeness(org_clean, "postal_code")})
    summary_rows.append({"entity": "organizations", "metric": "completeness_state", "value": completeness(org_clean, "state")})

    # Issues count
    issues_df = pd.DataFrame(issues)
    issue_counts = issues_df.groupby(["entity", "issue"]).size().reset_index(name="count") if len(issues_df) else pd.DataFrame(columns=["entity","issue","count"])

    md = []
    md.append("# Standardization & Cleansing Summary\n")
    md.append(f"Generated at: **{datetime.now().isoformat(timespec='seconds')}**\n")
    md.append("## Output Artifacts\n")
    md.append(f"- Clean providers CSV: `{OUT_PROV_CSV}`\n")
    md.append(f"- Clean organizations CSV: `{OUT_ORG_CSV}`\n")
    md.append(f"- SQLite tables: `stg_providers_clean`, `stg_organizations_clean`, `stg_standardization_issues`\n")

    md.append("\n## Key Completeness Metrics (Post-Standardization)\n")
    md.append(tabulate(pd.DataFrame(summary_rows), headers="keys", tablefmt="github", showindex=False))

    md.append("\n\n## Standardization Issues Logged (Counts)\n")
    if len(issue_counts):
        md.append(tabulate(issue_counts, headers="keys", tablefmt="github", showindex=False))
    else:
        md.append("_No issues logged._")

    ensure_dir(REPORT_DIR)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("✅ Step 3 complete")
    print(f"Clean providers: {OUT_PROV_CSV}")
    print(f"Clean orgs:      {OUT_ORG_CSV}")
    print(f"Summary report:  {SUMMARY_MD}")
    print("Next: Step 4 match candidate generation (blocking + scoring).")


if __name__ == "__main__":
    main()
