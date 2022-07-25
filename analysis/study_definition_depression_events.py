######################################

# This script provides the formal specification of the study data that will be
# extracted from the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import StudyDefinition, patients, Measure
from cohortextractor import codelistlib

from config import start_date, end_date

from codelists import depression_codes, depression_resolved_codes

from depression_variables import depression_register_variables
from demographic_variables import demographic_variables

def with_these_clinical_events_date_X(name, codelist, start_date, n):

    def var_signature(name, codelist, start_date, exclude_code=None):
        return {
            name : patients.with_these_clinical_events(
                    codelist,
                    returning="code",
                    find_first_match_in_period=True,
                    between=[start_date, "last_day_of_month(index_date)"],
                    include_date_of_match=True,
                    date_format="YYYY-MM-DD",
                    return_expectations={
                        "category": {"ratios": {"10": 0.2, "11": 0.3, "12": 0.5}}
                    },
            ),
        }

    variables = var_signature(f"{name}_1", codelist, start_date)
    for i in range(2, n+1):
        variables.update(var_signature(f"{name}_{i}", codelist, f"{name}_{i-1}_date + 1 day"))
    return variables

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
        (sex = "M" OR sex = "F" ) AND
        depression_register
        """,
    ),
    gms_registration_status=demographic_variables["gms_registration_status"],
    age=demographic_variables["age"],
    age_band=demographic_variables["age_band"],
    sex=demographic_variables["sex"],
    **depression_register_variables,
    # NOTE: not currently excluding ongoing
    **with_these_clinical_events_date_X(
        name = "depression_15mo_code",
        codelist = depression_codes,
        start_date = "first_day_of_month(index_date) - 14 months",
        n = 10,
    ),
)
