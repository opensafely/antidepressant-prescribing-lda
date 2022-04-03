######################################

# This script provides the formal specification of the study data that will be extracted from
# the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import StudyDefinition, patients, Measure

from config import start_date, end_date

from demographic_variables import *
from depression_variables import depression_register_variables


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
        # Depression patient list
        age_qof>=18 AND
        age_band_qof != "Unknown" AND

	# Extra OpenSafely parameters
        NOT has_died AND 
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
    **depression_register_variables,
    **{
        "sex": demographic_variables["sex"],
    },
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="prevalence_rate",
        numerator="depression_register",
        denominator="population",
        group_by=["population"],
        small_number_suppression=True,
    ),
]
