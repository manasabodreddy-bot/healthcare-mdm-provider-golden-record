# MDM Business Rules – Provider Data

## 1. Standardization Rules
- Names: trimmed, title-cased
- Emails: lowercased, regex-validated
- Phone numbers: digits-only, 10-digit normalization
- Addresses: token normalization (st., rd., ave.)
- Country/State: validated against reference tables

---

## 2. Matching Rules

### Blocking
- Email exact match
- NPI exact match
- Last name + city/postal + country
- First initial + last name + postal + country

### Scoring (Weight-of-Evidence)
- NPI match: highest weight
- Email match: high weight
- Name similarity
- Address similarity
- Phone and geographic corroboration

---

## 3. Decision Thresholds
- Auto Match: deterministic or very high confidence
- Manual Review: strong signals but insufficient certainty
- No Match: weak or conflicting evidence

---

## 4. Survivorship Rules
- Prefer higher-trust source systems
- Prefer latest updated records
- Prefer non-null, complete values
- Log conflicts when multiple values exist

---

## 5. Governance
- No manual merge without audit record
- All merges must be reversible
