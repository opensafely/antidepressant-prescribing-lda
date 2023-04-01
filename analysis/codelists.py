######################################

# Some covariates used in the study are created from codelists of clinical
# conditions or numerical values available on a patient's records.
# This script fetches all of the codelists identified in codelists.txt from
# OpenCodelists.

######################################


# --- IMPORT STATEMENTS ---

# Import code building blocks from cohort extractor package
from cohortextractor import codelist_from_csv, combine_codelists


# --- CODELISTS ---


# Medication DM&D

# Selective serotonin reputake inhibitors
ssri_codes = codelist_from_csv(
    "local_codelists/opensafely-selective-serotonin-reuptake-inhibitors-dmd_new.csv",
    system="snomed",
    column="dmd_id",
)
tricyclic_codes = codelist_from_csv(
    "local_codelists/opensafely-tricyclic-and-related-antidepressants-dmd_new.csv",
    system="snomed",
    column="dmd_id",
)
maoi_codes = codelist_from_csv(
    "local_codelists/opensafely-monoamine-oxidase-inhibitors-dmd_new.csv",
    system="snomed",
    column="dmd_id",
)
other_antidepressant_codes = codelist_from_csv(
    "local_codelists/opensafely-other-antidepressants-dmd_new.csv",
    system="snomed",
    column="dmd_id",
)
maoi_or_other_codes = combine_codelists(
    maoi_codes,
    other_antidepressant_codes,
)

# Groups

# Learning disabilities
learning_disability_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-ld_cod.csv",
    system="snomed",
    column="code",
)

# Autism
autism_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-autism_cod.csv",
    system="snomed",
    column="code",
)

# Care homes
carehome_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-carehome_cod.csv",
    system="snomed",
    column="code",
)

# Depression
depression_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-depr_cod.csv",
    system="snomed",
    column="code",
)

# Depression resolved
depression_resolved_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-depres_cod.csv",
    system="snomed",
    column="code",
)

# Depression Review
depression_review_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-deprvw_cod.csv",
    system="snomed",
    column="code",
)

# Depression indicator unsuitable
depression_review_unsuitable_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-deprpcapu_cod.csv",
    system="snomed",
    column="code",
)

# Depression indicator informed dissent
depression_review_dissent_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-deprpcadec_cod.csv",
    system="snomed",
    column="code",
)


# Depression review invitation
depression_invitation_codes = codelist_from_csv(
    "codelists/nhsd-primary-care-domain-refsets-deprinvite_cod.csv",
    system="snomed",
    column="code",
)

# Anxiety codes
anxiety_codes = codelist_from_csv(
    "codelists/opensafely-anxiety-disorders.csv",
    system="snomed",
    column="code",
)

# Variables

# Ethnicity
ethnicity_codes_16 = codelist_from_csv(
    "codelists/opensafely-ethnicity-snomed-0removed.csv",
    system="snomed",
    column="snomedcode",
    category_column="Grouping_16",
)
