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
    depression_codes,
    depression_resolved_codes,
)
from config import (
    start_date,
    end_date,
    demographics,
)

from demographic_variables import demographic_variables

from depression_variables import (
    depression_register_variables,
    dep003_variables,
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
        (sex = "M" OR sex = "F") AND
        depression_register
        """,
    ),
    # Common demographic variables
    **demographic_variables,
    # QOF DEP003
    **depression_register_variables,
    **dep003_variables,
    # Depression
)

# --- DEFINE MEASURES ---

##  QOF Measures

measures = [
    # QOF achievement over the population
    Measure(
        id="dep003_total_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["population"],
        small_number_suppression=True,
    ),
    # QOF achievement by practice
    # Output checking for practice will happen on the decile table
    Measure(
        id="dep003_practice_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["practice"],
    ),
    # Invitation codes
    Measure(
        id="dep003_invite_1_code_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["depr_invite_1_code"],
    ),
    Measure(
        id="dep003_invite_2_code_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["depr_invite_2_code"],
    ),
]

# QOF achievement by each demographic in the config file
for d in demographics:
    m = Measure(
        id=f"dep003_{d}_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)

### Debugging
for i in range(1, 9):
    m = Measure(
        id=f"dep003_denominator_r{i}_total_rate",
        numerator=f"dep003_denominator_r{i}",
        denominator="population",
        group_by="population",
        small_number_suppression=True,
    )
    measures.append(m)

### Exclusions
exclusions = ["unsuitable_12mo", "dissent_12mo"]
for ex in exclusions:
    m = Measure(
        id=f"dep003_excl_{ex}_total_rate",
        numerator=ex,
        denominator="population",
        group_by="population",
        small_number_suppression=True,
    )
    measures.append(m)
    # Output checking for practice will happen on the decile table
    m = Measure(
        id=f"dep003_excl_{ex}_practice_rate",
        numerator=ex,
        denominator="population",
        group_by=["practice"],
    )
    measures.append(m)
    for d in demographics:
        m = Measure(
            id=f"dep003_excl_{ex}_{d}_rate",
            numerator=ex,
            denominator="population",
            group_by=[d],
            small_number_suppression=True,
        )
        measures.append(m)
