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
    ssri_codes,
    tricyclic_codes,
    maoi_or_other_codes,
    combine_codelists,
)
from config import (
    start_date,
    end_date,
)

from demographic_variables import demographic_variables

antidepressant_types = {
    "antidepressant_ssri": ssri_codes,
    "antidepressant_tricyclic": tricyclic_codes,
    "antidepressant_other": maoi_or_other_codes,
    "antidepressant_any": combine_codelists(
        ssri_codes, tricyclic_codes, maoi_or_other_codes
    ),
}


def create_antidepressant_vars():
    def var_signature(name, codelist):
        return {
            f"{name}_count": patients.with_these_medications(
                codelist=codelist,
                returning="binary_flag",
                between=[
                    "first_day_of_month(index_date)",
                    "last_day_of_month(index_date)",
                ],
            ),
            f"{name}_events": patients.with_these_medications(
                codelist=codelist,
                returning="number_of_matches_in_period",
                between=[
                    "first_day_of_month(index_date)",
                    "last_day_of_month(index_date)",
                ],
                return_expectations={
                    "int": {"distribution": "normal", "mean": 3, "stddev": 0.5}
                },
            ),
        }

    antidepressant_vars = {}
    for antidepressant_type, codelist in antidepressant_types.items():
        antidepressant_vars.update(
            var_signature(antidepressant_type, codelist)
        )
    return antidepressant_vars


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
        gms_registration_status AND
        age_band != "Unknown" AND
        (sex = "M" OR sex = "F")
        """,
    ),
    # Common demographic variables
    **demographic_variables,
    **create_antidepressant_vars(),
)

# --- DEFINE MEASURES ---
measures = [
    Measure(
        id="antidepressant_ssri_all_total_rate",
        numerator="antidepressant_ssri_count",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_tricyclic_all_total_rate",
        numerator="antidepressant_tricyclic_count",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_other_all_total_rate",
        numerator="antidepressant_other_count",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_all_total_rate",
        numerator="antidepressant_any_count",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_ssri_all_total_events_rate",
        numerator="antidepressant_ssri_events",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_tricyclic_all_total_events_rate",
        numerator="antidepressant_tricyclic_events",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_other_all_total_events_rate",
        numerator="antidepressant_other_events",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_all_total_events_rate",
        numerator="antidepressant_any_events",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
]

# Demographic trends in prescribing for each at-risk group
# Use a set difference because there are some categories that are both
# lda subgroups and demographic groups
for d in [
    "age_band",
    "sex",
    "region",
    "carehome",
    "imd",
    "autism",
    "learning_disability",
]:
    m = Measure(
        id=f"antidepressant_any_all_breakdown_{d}_rate",
        numerator="antidepressant_any_count",
        denominator="population",
        group_by=[d],
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"antidepressant_any_all_breakdown_{d}_events_rate",
        numerator="antidepressant_any_events",
        denominator="population",
        group_by=[d],
        small_number_suppression=False,
    )
    measures.append(m)
