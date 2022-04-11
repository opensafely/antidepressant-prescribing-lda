######################################

# This script provides the formal specification of the study data that will be
# extracted from the OpenSAFELY database.

######################################


# IMPORT STATEMENTS ----

# Import code building blocks from cohort extractor package
from cohortextractor import (
    StudyDefinition,
    patients,
    Measure,
)

# Import common variables
# NOTE: I do not like that the imported variable names are opaque
# but the study definition requires a dictionary
from demographic_variables import demographic_variables

from depression_variables import (
    depression_register_variables,
    dep003_variables,
)


from config import start_date, end_date, demographics

# Define study population and variables
study = StudyDefinition(
    index_date=start_date,
    # Configure the expectations framework
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.1,
    },
    # TODO: determine whether we want the sex exclusion
    # TODO: could immediately restrict this to dep003_denominator
    population=patients.satisfying(
        """
        (sex = "M" OR sex = "F") AND
        depression_register
        """,
    ),
    # Demographic variables
    **demographic_variables,
    # QOF variables
    **depression_register_variables,
    **dep003_variables
)

# TODO: Small number suppression may be overly stringent for decile chart
# production
# See: https://github.com/opensafely-core/cohort-extractor/issues/759
# When running, we should check how much is redacted
# Using tested code now rather than custom decile chart redaction code

# --- DEFINE MEASURES ---
measures = [
    # QOF achievement over the population (same as denominator)
    Measure(
        id="dep003_total_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["population"],
        small_number_suppression=True,
    ),
    # QOF achievement by practice
    Measure(
        id="dep003_practice_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["practice"],
        small_number_suppression=True,
    ),
]
# QOF achievement by each demographic in the config file
for d in demographics:
    m = Measure(
        id="dep003_{}_rate".format(d),
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
