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
    depression_codes,
    depression_resolved_codes,
    learning_disability_codes,
    autism_codes,
)

# Import common variables
# NOTE: I do not like that the imported variable names are opaque
# but the study definition requires a dictionary
from demographic_variables import demographic_variables
from dep003_variables import dep003_variables

from config import start_date, end_date, depr_register_date, codelist_path, demographics

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
    # Number of patients who have a current diagnosis of depression (exclude those with a depression resolved code)
    population=patients.satisfying(
        """
        # Define general population parameters
        registered AND 
        (NOT has_died) AND
        # TODO: why are we excluding intersex or blank? 
        (sex = "M" OR sex = "F") AND
        # TODO: how do we choose the upper threshold? 
        (age >=18 AND age < 110)
        # depression_date AND
        # NOT (depression_resolved)
        """,
        has_died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.1},
        ),
        registered=patients.registered_as_of(
            "index_date", return_expectations={"incidence": 0.9}
        ),
        # TODO: change between to depr_register_date
        depression_resolved=patients.with_these_clinical_events(
            codelist=depression_resolved_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "depression_date + 1 day",
                "last_day_of_month(index_date)",
            ],
            return_expectations={"incidence": 0.01},
        ),
    ),
    # QOF variables
    **dep003_variables,
    # Demographic variables
    **demographic_variables
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="practice_rate",
        numerator="event",
        denominator="population",
        group_by=["practice"],
    ),
]
for d in demographics:
    m = Measure(
        id="prevalence_rate_{}".format(d),
        numerator="event",
        denominator="population",
        group_by=[d],
    )
    measures.append(m)
