######################################

# This script provides the formal specification of the study data that will
# be extracted from the OpenSAFELY database.

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
    tricyclic_codes,
    maoi_or_other_codes,
    combine_codelists,
)
from config import (
    start_date,
    end_date,
)

from demographic_variables import demographic_variables

codelist_any = combine_codelists(
    ssri_codes, tricyclic_codes, maoi_or_other_codes
)

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
    # TODO: determine whether we want the sex exclusion
    population=patients.satisfying(
        """
        gms_registration_status AND
        age_band != "Unknown" AND
        (sex = "M" OR sex = "F")
        """,
    ),
    # Common demographic variables
    **demographic_variables,
    antidepressant_any=patients.with_these_medications(
        codelist=codelist_any,
        returning="binary_flag",
        between=[
            "first_day_of_month(index_date) - 5 months",
            "last_day_of_month(index_date)",
        ],
    ),
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="antidepressant_any_learning_disability_total_prevalence",
        numerator="antidepressant_any",
        denominator="population",
        group_by=["learning_disability"],
        small_number_suppression=False,
    ),
]
