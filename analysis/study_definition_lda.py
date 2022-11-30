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
    anxiety_codes,
    ssri_codes,
    tricyclic_codes,
    maoi_or_other_codes,
    depression_codes,
)
from config import (
    start_date,
    end_date,
    demographics,
    lda_subgroups,
)

from demographic_variables import demographic_variables
from depression_variables import (
    depression_register_variables,
)

antidepressant_types = {
    "antidepressant_ssri": ssri_codes,
    "antidepressant_tricyclic": tricyclic_codes,
    "antidepressant_other": maoi_or_other_codes,
}
antidepressant_groups = [
    "antidepressant_ssri",
    "antidepressant_tricyclic",
    "antidepressant_other",
    "antidepressant_any",
]


def create_antidepressant_vars():
    def var_signature(name, codelist):
        return {
            name: patients.with_these_medications(
                codelist=codelist,
                returning="binary_flag",
                find_last_match_in_period=True,
                date_format="YYYY-MM-DD",
                between=[
                    "first_day_of_month(index_date)",
                    "last_day_of_month(index_date)",
                ],
            ),
            f"{name}_previous": patients.with_these_medications(
                codelist=codelist,
                returning="binary_flag",
                find_last_match_in_period=True,
                between=[
                    f"first_day_of_month(index_date) - 2 years",
                    f"first_day_of_month(index_date) - 1 day",
                ],
            ),
            f"{name}_naive": patients.satisfying(
                f"""
                NOT {name}_previous
                """,
            ),
            f"{name}_new": patients.satisfying(
                f"""
                {name} AND
                {name}_naive
                """,
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
    # QOF DEP003
    **depression_register_variables,
    # New depression
    depression_naive=patients.satisfying(
        """
        depression_list_type AND
        NOT depression_2y
        """,
        depression_2y=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date) - 2 years",
                "last_day_of_month(index_date) - 1 day",
            ],
            return_expectations={"incidence": 0.01},
        ),
    ),
    depression_new=patients.satisfying(
        """
        depression_register AND
        depression_naive
        """,
        return_expectations={"incidence": 0.1},
    ),
    **create_antidepressant_vars(),
    antidepressant_any=patients.satisfying(
        """
        antidepressant_ssri OR
        antidepressant_tricyclic OR
        antidepressant_other
        """,
    ),
    antidepressant_any_naive=patients.satisfying(
        """
        antidepressant_ssri_naive AND
        antidepressant_tricyclic_naive AND
        antidepressant_other_naive
        """,

    ),
    antidepressant_any_new=patients.satisfying(
        """
        antidepressant_any_naive AND (
            antidepressant_ssri_new OR
            antidepressant_tricyclic_new OR
            antidepressant_other_new
        )
        """,
    ),
    # Subgroups variables for measures framework
    # Needed because the measures framework is a ratio, not a rate
    antidepressant_any_18=patients.satisfying(
        """
        antidepressant_any AND
        depression_list_type
        """,
    ),
    anxiety=patients.with_these_clinical_events(
        codelist=anxiety_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        on_or_before="last_day_of_month(index_date)",
        return_expectations={"incidence": 0.01},
    ),
    diagnosis=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "Depression register": "depression_register AND NOT anxiety",
            "Anxiety": "anxiety AND NOT depression_register",
            "Both": "depression_register AND anxiety",
            "Neither": "NOT anxiety AND NOT depression_register",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "Unknown": 0.0,
                    "Depression register": 0.4,
                    "Anxiety": 0.2,
                    "Both": 0.3,
                    "Neither": 0.1,
                },
            },
        },
    ),
)

# --- DEFINE MEASURES ---

measures = [
    Measure(
        id="antidepressant_any_all_breakdown_diagnosis_18+_rate",
        numerator="antidepressant_any_18",
        denominator="depression_list_type",
        group_by=["diagnosis"],
        small_number_suppression=True,
    ),
    Measure(
        id="depression_all_total_rate",
        numerator="depression_register",
        denominator="depression_list_type",
        group_by="population",
        small_number_suppression=True,
    ),
    Measure(
        id="depression_new_all_total_rate",
        numerator="depression_new",
        denominator="depression_naive",
        group_by="population",
        small_number_suppression=True,
    ),
]
for group in lda_subgroups:
    m = Measure(
        id=f"antidepressant_any_{group}_breakdown_diagnosis_18+_rate",
        numerator=f"antidepressant_any_18",
        denominator=f"depression_list_type",
        group_by=[group, "diagnosis"],
        small_number_suppression=True,
    )
    measures.append(m)
    m = Measure(
        id=f"depression_{group}_total_rate",
        numerator=f"depression_register",
        denominator=f"depression_list_type",
        group_by=[group],
        small_number_suppression=True,
    )
    measures.append(m)
    m = Measure(
        id=f"depression_new_{group}_total_rate",
        numerator=f"depression_new",
        denominator=f"depression_naive",
        group_by=[group],
        small_number_suppression=True,
    )
    measures.append(m)

for antidepressant_group in antidepressant_groups:
    m = Measure(
        id=f"{antidepressant_group}_all_total_rate",
        numerator=antidepressant_group,
        denominator="population",
        group_by="population",
        small_number_suppression=True,
    )
    measures.append(m)
    new_m = Measure(
        id=f"{antidepressant_group}_new_all_total_rate",
        numerator=f"{antidepressant_group}_new",
        denominator=f"{antidepressant_group}_naive",
        group_by="population",
        small_number_suppression=True,
    )
    measures.append(new_m)
    # Group rate by outcome
    for group in lda_subgroups:
        m = Measure(
            id=f"{antidepressant_group}_{group}_total_rate",
            numerator=f"{antidepressant_group}",
            denominator="population",
            group_by=[group],
            small_number_suppression=True,
        )
        measures.append(m)
        new_m = Measure(
            id=f"{antidepressant_group}_new_{group}_total_rate",
            numerator=f"{antidepressant_group}_new",
            denominator=f"{antidepressant_group}_naive",
            group_by=[group],
            small_number_suppression=True,
        )
        measures.append(new_m)

# Demographic trends in prescribing for each at-risk group
# Use a set difference because there are some categories that are both
# lda subgroups and demographic groups
breakdown_list = list(set(demographics) - set(lda_subgroups))
for d in breakdown_list:
    m = Measure(
        id=f"antidepressant_any_all_breakdown_{d}_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
    for group in lda_subgroups:
        m = Measure(
            id=f"antidepressant_any_{group}_breakdown_{d}_rate",
            numerator=f"antidepressant_any",
            denominator="population",
            group_by=[group, d],
            small_number_suppression=True,
        )
        measures.append(m)
