######################################

# This script provides the formal specification of the study data that will be
# extracted from the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import StudyDefinition, patients, Measure

from config import start_date, end_date

from codelists import depression_codes, first_new_codes, ongoing_codes

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
        depr_lat
        """,
    ),
    depr_lat=patients.with_these_clinical_events(
        between=[
            "first_day_of_month(index_date)",
            "last_day_of_month(index_date)",
        ],
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    first_new_lat=patients.with_these_clinical_events(
        between=[
            "first_day_of_month(index_date)",
            "last_day_of_month(index_date)",
        ],
        codelist=first_new_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    ongoing_lat=patients.with_these_clinical_events(
        between=[
            "first_day_of_month(index_date)",
            "last_day_of_month(index_date)",
        ],
        codelist=ongoing_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    depr_and_ongoing=patients.satisfying(
        """
        depr_lat AND
        ongoing_lat AND
        depr_lat_date = ongoing_lat_date
        """,
    ),
    depr_and_first_new=patients.satisfying(
        """
        depr_lat AND
        first_new_lat AND
        depr_lat_date = first_new_lat_date
        """,
    ),
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="first_new_rate",
        numerator="first_new_lat",
        denominator="population",
        group_by=["population"],
    ),
    Measure(
        id="ongoing_rate",
        numerator="ongoing_lat",
        denominator="population",
        group_by=["population"],
    ),
    Measure(
        id="depr_first_new_rate",
        numerator="depr_and_first_new",
        denominator="population",
        group_by=["population"],
    ),
    Measure(
        id="depr_ongoing_rate",
        numerator="depr_and_ongoing",
        denominator="population",
        group_by=["population"],
    ),
]
