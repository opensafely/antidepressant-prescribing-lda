######################################

# This script provides the formal specification of the study data that will be extracted from
# the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import (
    StudyDefinition,
    patients,
    Measure,
)

# Import codelists from codelist.py (which pulls them from the codelist folder)
from codelists import (
    ssri_codes,
    learning_disability_codes,
    autism_codes,
    carehome_primis_codes,
    depression_codes,
    depression_resolved_codes,
    depression_review_codes,
)
from config import start_date, end_date, codelist_path, demographics

from demographic_variables import demographic_variables

from dep003_variables import dep003_variables

# Define study population and variables
study = StudyDefinition(
    index_date=start_date,
    # Configure the expectations framework
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.1,
    },
    # Define the study population
    population=patients.satisfying(
        """
        NOT has_died
        AND
        registered
        AND
        (sex = "M" OR sex = "F")
        AND
        (age >=0 AND age < 110)
        AND
        (learning_disability OR autism)
        """,
        has_died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
        ),
        registered=patients.satisfying(
            "registered_at_start",
            registered_at_start=patients.registered_as_of("index_date"),
        ),
        # Groups
        # Learning disabilities
        learning_disability=patients.with_these_clinical_events(
            learning_disability_codes,
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.2},
        ),
        # Autism
        autism=patients.with_these_clinical_events(
            autism_codes,
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.3},
        ),
    ),
    # Common demographic variables
    **demographic_variables,
    # QOF DEP003
    **dep003_variables,
    # Other subgroups
    # Care home
    care_home=patients.with_these_clinical_events(
        carehome_primis_codes,
        on_or_before="index_date",
        returning="binary_flag",
        return_expectations={"incidence": 0.2},
    ),
    # Depression
    depression=patients.satisfying(
        """
        new_depression_date AND
	NOT depression_resolved
        """,
        new_depression_date=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=["first_day_of_month(index_date)", "last_day_of_month(index_date)"],
            return_expectations={
                "date": {
                    "earliest": "first_day_of_month(index_date)",
                    "latest": "last_day_of_month(index_date)",
                },
                "incidence": 0.98,
            },
        ),
        depression_resolved=patients.with_these_clinical_events(
            codelist=depression_resolved_codes,
            returning="binary_flag",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=["new_depression_date", "last_day_of_month(index_date)"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    # New depression
    new_depression=patients.satisfying(
        """
        depression_date AND
	NOT depression_resolved AND
        NOT previous
        """,
        previous=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=["new_depression_date - 2 years", "new_depression_date - 1 day"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    # Antidepressant Prescriptions
    # TODO: change this to a for loop for each medication
    # 1. Number of patients who’ve been prescribed each antidepressant this month
    # 2. Number of patients with a first prescriptions of each ad this month (defined as px for AD where none issued in previous two years)
    # SSRIs
    antidepressant_ssri=patients.with_these_medications(
        ssri_codes,
        between=["index_date", "last_day_of_month(index_date)"],
        returning="binary_flag",
        return_expectations={"incidence": 0.5},
    ),
    antidepressant_ssri_first=patients.satisfying(
        """
    antidepressant_ssri_current_date
    AND
    NOT antidepressant_ssri_last_date
    """,
        return_expectations={
            "incidence": 0.01,
        },
        antidepressant_ssri_current_date=patients.with_these_medications(
            ssri_codes,
            returning="date",
            find_last_match_in_period=True,
            between=["index_date", "last_day_of_month(index_date)"],
            return_expectations={"incidence": 0.1},
        ),
        antidepressant_ssri_last_date=patients.with_these_medications(
            ssri_codes,
            returning="date",
            find_first_match_in_period=True,
            between=[
                "antidepressant_ssri_current_date - 2 year",
                "antidepressant_ssri_current_date - 1 day",
            ],
            return_expectations={"incidence": 0.5},
        ),
    ),
)


# --- DEFINE MEASURES ---

# TODO: Automate this with a for loop

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="practice_rate",
        numerator="numerator",
        denominator="denominator",
        group_by=["practice"],
    ),
]
for d in demographics:
    m = Measure(
        id="prevalence_rate_{}".format(d),
        numerator="numerator",
        denominator="denominator",
        group_by=[d],
    )
    measures.append(m)
