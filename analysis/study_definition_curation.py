######################################

# This curation script is checking for
# impossible dates

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import (
    StudyDefinition,
    patients,
    combine_codelists,
)

# Import codelists from codelist.py (which pulls them from the codelist folder)
from codelists import (
    anxiety_codes,
    ssri_codes,
    tricyclic_codes,
    maoi_codes,
    other_antidepressant_codes,
    depression_codes,
    depression_resolved_codes,
    learning_disability_codes,
    autism_codes,
    carehome_codes,
)

all_antidepressant_codes = combine_codelists(
    ssri_codes, tricyclic_codes, maoi_codes, other_antidepressant_codes
)

# Define study population and variables
study = StudyDefinition(
    index_date="2019-03-01",
    # Configure the expectations framework
    default_expectations={
        "date": {"earliest": "1900-01-01", "latest": "today"},
        "rate": "uniform",
        "incidence": 0.8,
    },
    population=patients.satisfying(
        """
        depr_earliest_date
        """,
    ),
    yob=patients.date_of_birth(date_format="YYYY"),
    sex=patients.sex(
        return_expectations={
            "rate": "universal",
            "category": {"ratios": {"M": 0.4, "F": 0.5, "I": 0.1}},
        }
    ),
    died_earliest_date=patients.died_from_any_cause(
        on_or_before="last_day_of_month(index_date)",
        returning="date_of_death",
    ),
    died_latest_date=patients.died_from_any_cause(
        on_or_after="last_day_of_month(index_date)",
        returning="date_of_death",
    ),
    ld_earliest_date=patients.with_these_clinical_events(
        learning_disability_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    ld_latest_date=patients.with_these_clinical_events(
        learning_disability_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    aut_earliest_date=patients.with_these_clinical_events(
        autism_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    aut_latest_date=patients.with_these_clinical_events(
        autism_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    ch_earliest_date=patients.with_these_clinical_events(
        carehome_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    ch_latest_date=patients.with_these_clinical_events(
        carehome_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    depr_earliest_date=patients.with_these_clinical_events(
        depression_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    depr_latest_date=patients.with_these_clinical_events(
        depression_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    depr_res_earliest_date=patients.with_these_clinical_events(
        depression_resolved_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    depr_res_latest_date=patients.with_these_clinical_events(
        depression_resolved_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    anxiety_earliest_date=patients.with_these_clinical_events(
        anxiety_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    anxiety_latest_date=patients.with_these_clinical_events(
        anxiety_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
    antidepressant_earliest_date=patients.with_these_medications(
        all_antidepressant_codes,
        on_or_before="last_day_of_month(index_date)",
        find_first_match_in_period=True,
        returning="date",
    ),
    antidepressant_latest_date=patients.with_these_medications(
        all_antidepressant_codes,
        on_or_after="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        returning="date",
    ),
)
