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
    combine_codelists,
)

# Import codelists from codelist.py (which pulls them from the codelist folder)
from codelists import (
    ssri_codes,
    tricyclic_codes,
    maoi_or_other_codes,
)
from config import (
    start_date,
    end_date,
    lda_subgroups,
)

from demographic_variables import demographic_variables

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
        gms_registration_status
        """,
    ),
    gms_registration_status=demographic_variables["gms_registration_status"],
    age=demographic_variables["age"],
    age_band=demographic_variables["age_band"],
    sex=demographic_variables["sex"],
    autism=demographic_variables["autism"],
    learning_disability=demographic_variables["learning_disability"],
    antidepressant_any=patients.with_these_medications(
        codelist=combine_codelists(
            ssri_codes, tricyclic_codes, maoi_or_other_codes
        ),
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date)",
            "last_day_of_month(index_date)",
        ],
    ),
)

# --- DEFINE MEASURES ---

measures = [
    Measure(
        id="antidepressant_any_all_breakdown_sex_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=["sex"],
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_all_breakdown_age_band_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=["age_band"],
        small_number_suppression=False,
    ),
]
for group in lda_subgroups:
    m = Measure(
        id=f"antidepressant_any_{group}_breakdown_sex_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=[group, "sex"],
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"antidepressant_any_{group}_breakdown_age_band_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=[group, "age_band"],
        small_number_suppression=False,
    )
    measures.append(m)
