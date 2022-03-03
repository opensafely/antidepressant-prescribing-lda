######################################

# This script provides the formal specification of the study data that will be extracted from
# the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import StudyDefinition, patients, Measure

from config import start_date, end_date

from demographic_variables import *
from dep003_variables import *

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
        """,
        has_died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
        ),
        registered=patients.satisfying(
            "registered_at_start",
            registered_at_start=patients.registered_as_of("index_date"),
        ),
    ),
    **{
        "sex": demographic_variables["sex"],
        "practice": demographic_variables["practice"],
        "age_financial_year": dep003_variables["age_financial_year"],
        "depression_register": dep003_variables["depression_register"],
    },
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="prevalence_rate",
        numerator="depression_register",
        denominator="population",
        group_by=["population"],
    ),
]
