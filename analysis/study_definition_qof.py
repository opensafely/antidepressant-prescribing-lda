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

from depression_variables import (
    depression_register_variables,
    depression_indicator_variables,
)


from config import start_date, end_date, codelist_path, demographics

# Define study population and variables
study = StudyDefinition(
    index_date=start_date,
    # Configure the expectations framework
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.1,
    },
    population=patients.satisfying(
        """
        depression_register AND
        age>=18

        # Extra OpenSafely population parameters
        (NOT has_died) AND
        (sex = "M" OR sex = "F") AND
        """,
        has_died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.1},
        ),
    ),
    # QOF variables
    **depression_register_variables,
    **depression_indicator_variables,
    # Demographic variables
    **demographic_variables
)

# TODO: Small number suppression may be overly stringent for decile chart production
# See: https://github.com/opensafely-core/cohort-extractor/issues/759
# When running, we should check how much is redacted
# Using tested code now rather than custom decile chart redaction code

# --- DEFINE MEASURES ---
measures = [
    # QOF achievement by practice
    Measure(
        id="qof_practice_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["practice"],
        small_number_suppression=True,
    ),
]
# QOF achievement by each demographic in the config file
for d in demographics:
    m = Measure(
        id="qof_{}_rate".format(d),
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
