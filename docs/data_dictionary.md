# Provider Master Data – Data Dictionary

## Provider Attributes

| Field | Description | Critical |
|------|------------|---------|
| provider_id_source | Source system identifier | Yes |
| source_system | Originating system | Yes |
| npi | National Provider Identifier | Yes |
| first_name | Provider first name | Yes |
| last_name | Provider last name | Yes |
| email | Contact email | Medium |
| phone | Contact phone number | Medium |
| specialty | Medical specialty | Medium |
| address_line1 | Primary address | Yes |
| city | City | Yes |
| state | State code | Yes |
| postal_code | Postal code | Yes |
| country | Country code | Yes |
| record_status | Active/Inactive | Yes |

---

## Golden Record Attributes
- `golden_provider_id`
- `cluster_size`
- `source_system_anchor`
- Survivorship-selected attributes

---

## XREF Attributes
- golden_provider_id
- source_system
- provider_id_source
- mapped_at
