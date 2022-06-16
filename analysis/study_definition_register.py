######################################

# This script provides the formal specification of the study data that will be
# extracted from the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import StudyDefinition, patients, Measure

from config import start_date, end_date, demographics

from demographic_variables import demographic_variables
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
    # TODO: determine whether we want the sex exclusion
    population=patients.satisfying(
        """
        (sex = "M" OR sex = "F") AND
        # Depression patient list
        depression_list_type
        """,
    ),
    **demographic_variables,
    **depression_register_variables,
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="register_total_rate",
        numerator="depression_register",
        denominator="population",
        group_by=["population"],
        small_number_suppression=True,
    ),
    Measure(
        id="register_practice_rate",
        numerator="depression_register",
        denominator="population",
        group_by=["practice"],
    ),
    Measure(
        id="register_code_rate",
        numerator="depression_register",
        denominator="population",
        group_by=["depr_lat_code"],
        small_number_suppression=True,
    )
]
# Register prevalence by each demographic in the config file
for d in demographics:
    m = Measure(
        id="register_{}_rate".format(d),
        numerator="depression_register",
        denominator="population",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
