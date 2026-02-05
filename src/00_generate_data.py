
"""
Project 1 - Step 1: Generate realistic MDM-style synthetic datasets.

Outputs (CSV) into data/raw/:
  - providers_raw.csv
  - organizations_raw.csv
  - affiliations_raw.csv
  - reference_country_state.csv

Design goals:
  - Two source systems with conflicting data (CRM_A, CRM_B)
  - Controlled duplicates (same person/org appears multiple times)
  - Missing identifiers (NPI missing in one system sometimes)
  - Inconsistent formatting (names, addresses, phones, emails)
  - Timestamp differences (updated_at)
  - Affiliations linking providers to organizations (primary/secondary)
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker


# -----------------------------
# Config
# -----------------------------
SEED = 42

RAW_DIR = os.path.join("data", "raw")

N_UNIQUE_PROVIDERS = 2200      # unique "real-world" entities
N_UNIQUE_ORGS = 420
DUPLICATE_RATE_PROVIDER = 0.28 # fraction of providers that will get a duplicate record across sources
DUPLICATE_RATE_ORG = 0.22

# Some providers will have multiple duplicates / variants (rare but realistic)
MULTI_DUPLICATE_RATE_PROVIDER = 0.05

# Percent missing NPI in CRM_B for those that do have NPI overall
MISSING_NPI_IN_ONE_SOURCE_RATE = 0.35

# Fraction of emails that are missing/invalid in one of the sources
EMAIL_ISSUE_RATE = 0.18

# Fraction of phones formatted differently
PHONE_FORMAT_VARIATION_RATE = 0.45

# A small fraction of invalid state codes to create quality issues
INVALID_STATE_RATE = 0.015

SOURCE_SYSTEMS = ["CRM_A", "CRM_B"]
TRUST_RANK = {"CRM_A": 2, "CRM_B": 1}  # later used in survivorship rules


# -----------------------------
# Helpers
# -----------------------------
fake = Faker("en_IN")  # India locale for realistic names/phones, but we'll still store country as "US/IN" mix
Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class ProviderEntity:
    entity_key: str
    npi: str | None
    first_name: str
    last_name: str
    gender: str
    specialty: str
    email: str | None
    phone: str | None
    address_line1: str
    city: str
    state: str
    postal_code: str
    country: str
    record_status: str


@dataclass
class OrgEntity:
    entity_key: str
    org_name: str
    org_type: str
    address_line1: str
    city: str
    state: str
    postal_code: str
    country: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_digits(s: str) -> str:
    return re.sub(r"\D+", "", s)


def random_date_within_days(days_back: int = 120) -> datetime:
    # updated_at in the past few months
    delta = random.randint(0, days_back)
    # random time during day
    seconds = random.randint(0, 86399)
    return datetime.now() - timedelta(days=delta, seconds=seconds)


def sometimes(p: float) -> bool:
    return random.random() < p


def make_postal_code(country: str) -> str:
    if country == "US":
        # 5-digit zip, sometimes ZIP+4
        if sometimes(0.08):
            return f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
        return f"{random.randint(10000, 99999)}"
    # India PIN
    return f"{random.randint(100000, 999999)}"


def pick_country() -> str:
    # Mixed to mimic global commercial data (but mostly US-like provider domains)
    return "US" if sometimes(0.75) else "IN"


def build_reference_states() -> pd.DataFrame:
    # Minimal but enough to validate. We'll include US states + some Indian states for realism.
    us_states = [
        ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"), ("CA", "California"),
        ("CO", "Colorado"), ("CT", "Connecticut"), ("FL", "Florida"), ("GA", "Georgia"),
        ("IL", "Illinois"), ("MA", "Massachusetts"), ("MD", "Maryland"), ("MI", "Michigan"),
        ("NC", "North Carolina"), ("NJ", "New Jersey"), ("NY", "New York"), ("OH", "Ohio"),
        ("PA", "Pennsylvania"), ("TX", "Texas"), ("VA", "Virginia"), ("WA", "Washington"),
    ]
    in_states = [
        ("KA", "Karnataka"), ("TS", "Telangana"), ("TN", "Tamil Nadu"), ("MH", "Maharashtra"),
        ("DL", "Delhi"), ("UP", "Uttar Pradesh"), ("WB", "West Bengal"),
    ]
    rows = []
    for code, name in us_states:
        rows.append({"country": "US", "state_code": code, "state_name": name})
    for code, name in in_states:
        rows.append({"country": "IN", "state_code": code, "state_name": name})
    return pd.DataFrame(rows)


def pick_state_for_country(country: str, ref_states: pd.DataFrame) -> str:
    subset = ref_states[ref_states["country"] == country]["state_code"].tolist()
    if not subset:
        return "NA"
    state = random.choice(subset)
    if sometimes(INVALID_STATE_RATE):
        # create an invalid state code
        return state[:-1] + random.choice(list("XYZ"))
    return state


def standardize_org_suffix(name: str) -> str:
    # Used to introduce variations later
    replacements = [
        ("Hospital", "Hosp"),
        ("Clinic", "Clnc"),
        ("Medical", "Med"),
        ("Center", "Ctr"),
        ("Centre", "Center"),
        ("Ltd", "Limited"),
        ("Inc", "Incorporated"),
    ]
    for a, b in replacements:
        if a in name and sometimes(0.35):
            name = name.replace(a, b)
    return name


def variant_name(first: str, last: str) -> Tuple[str, str]:
    # Introduce name variations: extra spaces, casing, initials
    f, l = first, last
    r = random.random()
    if r < 0.20:
        f = f.upper()
    elif r < 0.35:
        l = l.upper()
    elif r < 0.50:
        f = f.strip() + " "
    elif r < 0.60:
        # initial
        f = f[0] + "."
    return f, l


def variant_address(addr: str) -> str:
    # Introduce address token variations
    token_map = [
        (r"\bSt\b\.?", "Street"),
        (r"\bRd\b\.?", "Road"),
        (r"\bAve\b\.?", "Avenue"),
        (r"\bN\b\.?", "North"),
        (r"\bS\b\.?", "South"),
        (r"\bE\b\.?", "East"),
        (r"\bW\b\.?", "West"),
    ]
    out = addr
    for pattern, repl in token_map:
        if re.search(pattern, out) and sometimes(0.5):
            out = re.sub(pattern, repl, out)
    # sometimes remove commas or add extra whitespace
    if sometimes(0.25):
        out = out.replace(",", "")
    if sometimes(0.20):
        out = re.sub(r"\s+", " ", out).strip()
    return out


def variant_phone(phone: str) -> str:
    digits = clean_digits(phone)
    if len(digits) < 10:
        digits = (digits + "0" * 10)[:10]
    digits = digits[-10:]
    if not sometimes(PHONE_FORMAT_VARIATION_RATE):
        return digits
    fmt = random.choice([
        f"{digits}",
        f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:10]}",
        f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}",
        f"{digits[0:3]}.{digits[3:6]}.{digits[6:10]}",
        f"{digits[0:3]} {digits[3:6]} {digits[6:10]}",
    ])
    return fmt


def maybe_break_email(email: str) -> str | None:
    if not sometimes(EMAIL_ISSUE_RATE):
        return email
    # introduce missing or invalid
    r = random.random()
    if r < 0.50:
        return None
    if r < 0.75:
        return email.replace("@", "")
    return email.replace(".", "")


def generate_npi() -> str:
    # Simulated 10-digit NPI-like ID
    return str(random.randint(1000000000, 9999999999))


def pick_specialty() -> str:
    return random.choice([
        "Cardiology", "Oncology", "Dermatology", "Pediatrics", "Neurology",
        "General Medicine", "Endocrinology", "Gastroenterology", "Orthopedics",
        "Pulmonology", "Psychiatry"
    ])


def pick_org_type() -> str:
    return random.choice(["Hospital", "Clinic", "Lab", "Pharmacy", "Diagnostic Center"])


def create_base_provider_entities(ref_states: pd.DataFrame) -> List[ProviderEntity]:
    entities: List[ProviderEntity] = []
    for i in range(N_UNIQUE_PROVIDERS):
        country = pick_country()
        state = pick_state_for_country(country, ref_states)
        first = fake.first_name()
        last = fake.last_name()
        gender = random.choice(["F", "M", "U"])
        specialty = pick_specialty()
        npi = generate_npi() if (country == "US" and sometimes(0.78)) else None

        # email sometimes missing entirely at entity level
        email = fake.email() if sometimes(0.88) else None
        phone = fake.phone_number() if sometimes(0.92) else None

        address = fake.street_address()
        city = fake.city()
        postal = make_postal_code(country)
        status = random.choice(["Active", "Active", "Active", "Inactive"])  # mostly active

        entities.append(
            ProviderEntity(
                entity_key=f"PROV_{i:05d}",
                npi=npi,
                first_name=first,
                last_name=last,
                gender=gender,
                specialty=specialty,
                email=email,
                phone=phone,
                address_line1=address,
                city=city,
                state=state,
                postal_code=postal,
                country=country,
                record_status=status,
            )
        )
    return entities


def create_base_org_entities(ref_states: pd.DataFrame) -> List[OrgEntity]:
    entities: List[OrgEntity] = []
    for i in range(N_UNIQUE_ORGS):
        country = pick_country()
        state = pick_state_for_country(country, ref_states)
        org_type = pick_org_type()

        # A reasonably stable org name, then we will introduce variants across sources
        base_name = f"{fake.company()} {org_type}"
        org_name = base_name

        address = fake.street_address()
        city = fake.city()
        postal = make_postal_code(country)

        entities.append(
            OrgEntity(
                entity_key=f"ORG_{i:04d}",
                org_name=org_name,
                org_type=org_type,
                address_line1=address,
                city=city,
                state=state,
                postal_code=postal,
                country=country,
            )
        )
    return entities


def make_provider_record(entity: ProviderEntity, source: str) -> Dict[str, object]:
    # Create a source-specific record from an entity, with variations and occasional conflicts.
    first, last = entity.first_name, entity.last_name
    if source == "CRM_B" and sometimes(0.55):
        first, last = variant_name(first, last)

    addr = entity.address_line1
    if source == "CRM_B" and sometimes(0.55):
        addr = variant_address(addr)

    phone = entity.phone
    if phone is not None:
        phone = variant_phone(phone)

    email = entity.email
    if email is not None:
        email = maybe_break_email(email)

    # NPI can be missing in one source (but present in another)
    npi = entity.npi
    if npi is not None and source == "CRM_B" and sometimes(MISSING_NPI_IN_ONE_SOURCE_RATE):
        npi = None

    # Occasionally specialty differs between systems (realistic)
    specialty = entity.specialty
    if source == "CRM_B" and sometimes(0.08):
        specialty = random.choice([s for s in [
            "Cardiology", "Oncology", "Dermatology", "Pediatrics", "Neurology",
            "General Medicine", "Endocrinology", "Gastroenterology", "Orthopedics",
            "Pulmonology", "Psychiatry"
        ] if s != specialty])

    updated_at = random_date_within_days(120)
    # CRM_A tends to be more recently updated
    if source == "CRM_A" and sometimes(0.6):
        updated_at = updated_at + timedelta(days=random.randint(0, 20))

    record = {
        "source_system": source,
        "provider_id_source": f"{source}_P_{entity.entity_key}_{random.randint(100, 999)}",
        "npi": npi,
        "first_name": first,
        "last_name": last,
        "gender": entity.gender,
        "specialty": specialty,
        "email": email,
        "phone": phone,
        "address_line1": addr,
        "city": entity.city,
        "state": entity.state,
        "postal_code": entity.postal_code,
        "country": entity.country,
        "updated_at": updated_at.isoformat(timespec="seconds"),
        "record_status": entity.record_status,
        "entity_key": entity.entity_key,  # keep for evaluation only; we will DROP it in later steps
    }
    return record


def make_org_record(entity: OrgEntity, source: str) -> Dict[str, object]:
    org_name = entity.org_name
    addr = entity.address_line1
    if source == "CRM_B":
        if sometimes(0.55):
            org_name = standardize_org_suffix(org_name)
        if sometimes(0.50):
            addr = variant_address(addr)

    updated_at = random_date_within_days(120)
    if source == "CRM_A" and sometimes(0.55):
        updated_at = updated_at + timedelta(days=random.randint(0, 20))

    record = {
        "source_system": source,
        "org_id_source": f"{source}_O_{entity.entity_key}_{random.randint(100, 999)}",
        "org_name": org_name,
        "org_type": entity.org_type,
        "address_line1": addr,
        "city": entity.city,
        "state": entity.state,
        "postal_code": entity.postal_code,
        "country": entity.country,
        "updated_at": updated_at.isoformat(timespec="seconds"),
        "entity_key": entity.entity_key,  # keep for evaluation only; DROP later
    }
    return record


def choose_duplicate_entities(base_entities: List, rate: float) -> set:
    n = int(len(base_entities) * rate)
    chosen = set(random.sample([e.entity_key for e in base_entities], n))
    return chosen


def generate_affiliations(
    provider_records: pd.DataFrame,
    org_records: pd.DataFrame
) -> pd.DataFrame:
    """
    Create affiliations based on SOURCE IDs. That’s how real systems often link.

    We'll pick 1-2 org affiliations per provider record (not per entity),
    to simulate duplication propagation.
    """
    provider_ids = provider_records["provider_id_source"].tolist()
    org_ids = org_records["org_id_source"].tolist()

    rows = []
    for pid in provider_ids:
        # Each provider record affiliates with 1 to 2 orgs
        n_aff = 1 if sometimes(0.70) else 2
        chosen_orgs = random.sample(org_ids, n_aff)
        for j, oid in enumerate(chosen_orgs):
            rows.append({
                "provider_id_source": pid,
                "org_id_source": oid,
                "affiliation_type": "Primary" if j == 0 else "Secondary",
                "start_date": (datetime.now() - timedelta(days=random.randint(30, 2000))).date().isoformat(),
            })
    return pd.DataFrame(rows)


def validate_outputs(df_prov: pd.DataFrame, df_org: pd.DataFrame, df_aff: pd.DataFrame, ref_states: pd.DataFrame) -> None:
    """
    Hard validation to avoid hidden mistakes.
    Raises ValueError with clear message if something is off.
    """
    required_prov_cols = {
        "source_system","provider_id_source","npi","first_name","last_name","gender","specialty",
        "email","phone","address_line1","city","state","postal_code","country","updated_at","record_status"
    }
    required_org_cols = {
        "source_system","org_id_source","org_name","org_type",
        "address_line1","city","state","postal_code","country","updated_at"
    }
    required_aff_cols = {"provider_id_source","org_id_source","affiliation_type","start_date"}

    if not required_prov_cols.issubset(set(df_prov.columns)):
        raise ValueError(f"providers_raw.csv missing columns: {required_prov_cols - set(df_prov.columns)}")
    if not required_org_cols.issubset(set(df_org.columns)):
        raise ValueError(f"organizations_raw.csv missing columns: {required_org_cols - set(df_org.columns)}")
    if not required_aff_cols.issubset(set(df_aff.columns)):
        raise ValueError(f"affiliations_raw.csv missing columns: {required_aff_cols - set(df_aff.columns)}")

    # Referential integrity checks (affiliations must reference existing IDs)
    prov_ids = set(df_prov["provider_id_source"].astype(str))
    org_ids = set(df_org["org_id_source"].astype(str))

    bad_prov = df_aff.loc[~df_aff["provider_id_source"].isin(prov_ids)]
    bad_org = df_aff.loc[~df_aff["org_id_source"].isin(org_ids)]
    if len(bad_prov) > 0:
        raise ValueError(f"Affiliations contain provider_id_source not present in providers_raw: {len(bad_prov)} rows")
    if len(bad_org) > 0:
        raise ValueError(f"Affiliations contain org_id_source not present in organizations_raw: {len(bad_org)} rows")

    # State reference sanity (we expect some invalids, but most should be valid)
    valid_state_pairs = set(zip(ref_states["country"], ref_states["state_code"]))
    prov_valid = df_prov.apply(lambda r: (r["country"], r["state"]) in valid_state_pairs, axis=1).mean()
    org_valid = df_org.apply(lambda r: (r["country"], r["state"]) in valid_state_pairs, axis=1).mean()

    if prov_valid < 0.90:
        raise ValueError(f"Too many invalid provider state codes: valid ratio {prov_valid:.2f} (expected >= 0.90)")
    if org_valid < 0.90:
        raise ValueError(f"Too many invalid org state codes: valid ratio {org_valid:.2f} (expected >= 0.90)")

    # Ensure data volumes are meaningful
    if len(df_prov) < N_UNIQUE_PROVIDERS:
        raise ValueError(f"providers_raw too small: {len(df_prov)}")
    if len(df_org) < N_UNIQUE_ORGS:
        raise ValueError(f"organizations_raw too small: {len(df_org)}")
    if len(df_aff) < len(df_prov):
        raise ValueError(f"affiliations_raw too small: {len(df_aff)} (expected >= providers rows)")

    # Ensure both sources exist
    if set(df_prov["source_system"].unique()) != set(SOURCE_SYSTEMS):
        raise ValueError("Expected providers from both CRM_A and CRM_B")
    if set(df_org["source_system"].unique()) != set(SOURCE_SYSTEMS):
        raise ValueError("Expected organizations from both CRM_A and CRM_B")


def main() -> None:
    ensure_dir(RAW_DIR)

    ref_states = build_reference_states()
    ref_states_path = os.path.join(RAW_DIR, "reference_country_state.csv")
    ref_states.to_csv(ref_states_path, index=False)

    providers = create_base_provider_entities(ref_states)
    orgs = create_base_org_entities(ref_states)

    # Choose which unique entities will have duplicates across systems
    dup_prov_keys = choose_duplicate_entities(providers, DUPLICATE_RATE_PROVIDER)
    dup_org_keys = choose_duplicate_entities(orgs, DUPLICATE_RATE_ORG)

    records_prov: List[Dict[str, object]] = []
    records_org: List[Dict[str, object]] = []

    # Providers: generate one record per entity per source, plus duplicates for selected entities
    for p in providers:
        # Base record in CRM_A always
        records_prov.append(make_provider_record(p, "CRM_A"))

        # CRM_B record exists for most entities (simulate multi-system integration)
        if sometimes(0.92):
            records_prov.append(make_provider_record(p, "CRM_B"))

        # Introduce duplicates: extra record(s) in one of the sources
        if p.entity_key in dup_prov_keys:
            # Usually duplicate in CRM_B, sometimes in CRM_A
            dup_source = "CRM_B" if sometimes(0.75) else "CRM_A"
            records_prov.append(make_provider_record(p, dup_source))

            # Some entities get a second duplicate record variant
            if sometimes(MULTI_DUPLICATE_RATE_PROVIDER):
                dup_source2 = "CRM_B" if dup_source == "CRM_A" else "CRM_A"
                records_prov.append(make_provider_record(p, dup_source2))

    # Orgs: similar strategy
    for o in orgs:
        records_org.append(make_org_record(o, "CRM_A"))
        if sometimes(0.90):
            records_org.append(make_org_record(o, "CRM_B"))
        if o.entity_key in dup_org_keys:
            dup_source = "CRM_B" if sometimes(0.70) else "CRM_A"
            records_org.append(make_org_record(o, dup_source))

    df_prov = pd.DataFrame(records_prov)
    df_org = pd.DataFrame(records_org)

    # Affiliations based on current raw records
    df_aff = generate_affiliations(df_prov, df_org)

    # Drop evaluation keys? Keep entity_key only in raw to later demonstrate we can remove internal keys.
    # We keep it in raw for sanity checking now; later steps should DROP it.
    prov_path = os.path.join(RAW_DIR, "providers_raw.csv")
    org_path = os.path.join(RAW_DIR, "organizations_raw.csv")
    aff_path = os.path.join(RAW_DIR, "affiliations_raw.csv")

    df_prov.to_csv(prov_path, index=False)
    df_org.to_csv(org_path, index=False)
    df_aff.to_csv(aff_path, index=False)

    # Validate: no hidden mistakes
    validate_outputs(df_prov, df_org, df_aff, ref_states)

    # Print a small summary
    def pct(x: float) -> str:
        return f"{x*100:.1f}%"

    # Duplicate roughness checks (not perfect, but indicative)
    npi_dup_rate = (df_prov["npi"].dropna().duplicated().mean()) if df_prov["npi"].notna().any() else 0.0
    email_missing = df_prov["email"].isna().mean()
    npi_missing = df_prov["npi"].isna().mean()

    print("✅ Synthetic data generated in data/raw/")
    print(f"Providers rows:      {len(df_prov):,}")
    print(f"Organizations rows:  {len(df_org):,}")
    print(f"Affiliations rows:   {len(df_aff):,}")
    print(f"NPI missing:         {pct(npi_missing)}")
    print(f"Email missing:       {pct(email_missing)}")
    print(f"NPI duplicate ratio (rough): {pct(npi_dup_rate)}")
    print("Next: run Step 2 profiling & load to SQLite.")


if __name__ == "__main__":
    main()
