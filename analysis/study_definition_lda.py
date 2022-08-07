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
    maoi_codes,
    other_antidepressant_codes,
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
    "antidepressant_maoi": maoi_codes,
    "antidepressant_other_cod": other_antidepressant_codes,
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
                include_date_of_match=True,
                date_format="YYYY-MM-DD",
                between=[
                    "first_day_of_month(index_date)",
                    "last_day_of_month(index_date)",
                ],
            ),
            f"{name}_new": patients.with_these_medications(
                codelist=codelist,
                returning="binary_flag",
                find_last_match_in_period=True,
                include_date_of_match=True,
                date_format="YYYY-MM-DD",
                between=[
                    f"{name}_date - 2 years",
                    f"{name}_date - 1 day",
                ],
            ),
        }

    antidepressant_vars = {}
    for antidepressant_type, codelist in antidepressant_types.items():
        antidepressant_vars.update(
            var_signature(antidepressant_type, codelist)
        )
    return antidepressant_vars


def create_subgroups():
    def var_signature(name, subgroup, antidepressant):
        print(name, subgroup, antidepressant)
        return {
            name: patients.satisfying(
                f"""
                {subgroup} AND {antidepressant}
                """,
            ),
        }

    variables = {}
    for group_label, group in lda_subgroups.items():
        for antidepressant_group in antidepressant_groups:
            variables.update(
                var_signature(
                    f"{antidepressant_group}_{group_label}",
                    group,
                    antidepressant_group,
                )
            )
            variables.update(
                var_signature(
                    f"{antidepressant_group}_new_{group_label}",
                    group,
                    f"{antidepressant_group}_new",
                )
            )
    return variables


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
    depression_new=patients.satisfying(
        """
        depression_register AND
        NOT previous_depression
        """,
        previous_depression=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=["depr_lat_date - 2 years", "depr_lat_date - 1 day"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    **create_antidepressant_vars(),
    antidepressant_other=patients.satisfying(
        """
        antidepressant_maoi OR
        antidepressant_other_cod
        """
    ),
    antidepressant_other_new=patients.satisfying(
        """
        antidepressant_maoi_new OR
        antidepressant_other_cod_new
        """,
    ),
    antidepressant_any=patients.satisfying(
        """
        antidepressant_ssri OR
        antidepressant_tricyclic OR
        antidepressant_other
        """
    ),
    antidepressant_any_new=patients.satisfying(
        """
        antidepressant_ssri_new OR
        antidepressant_tricyclic_new OR
        antidepressant_other_new
        """,
    ),
    **create_subgroups(),
    antidepressant_any_18=patients.satisfying(
        """
        antidepressant_any AND
        depression_list_type
        """,
    ),
    depression_list_type_autism=patients.satisfying(
        """
        depression_list_type AND
        aut
        """,
    ),
    antidepressant_any_18_autism=patients.satisfying(
        """
        antidepressant_any AND
        depression_list_type_autism
        """,
    ),
    depression_list_type_learning_disability=patients.satisfying(
        """
        depression_list_type AND
        ld
        """,
    ),
    antidepressant_any_18_learning_disability=patients.satisfying(
        """
        antidepressant_any AND
        depression_list_type_learning_disability
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
        id="antidepressant_any_all_breakdown_diagnosis_rate",
        numerator="antidepressant_any_18",
        denominator="depression_list_type",
        group_by=["diagnosis"],
        small_number_suppression=True,
    ),
]
for group_label, group in lda_subgroups.items():
    m = Measure(
        id=f"antidepressant_any_{group_label}_breakdown_diagnosis_rate",
        numerator=f"antidepressant_any_18_{group_label}",
        denominator=f"depression_list_type_{group_label}",
        group_by=["diagnosis"],
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
        denominator="population",
        group_by="population",
        small_number_suppression=True,
    )
    measures.append(new_m)
    # Group rate by outcome
    for group_label, group in lda_subgroups.items():
        m = Measure(
            id=f"{antidepressant_group}_{group_label}_total_rate",
            numerator=f"{antidepressant_group}_{group_label}",
            denominator=group,
            group_by="population",
            small_number_suppression=True,
        )
        measures.append(m)
        new_m = Measure(
            id=f"{antidepressant_group}_new_{group_label}_total_rate",
            numerator=f"{antidepressant_group}_new_{group_label}",
            denominator=group,
            group_by="population",
            small_number_suppression=True,
        )
        measures.append(new_m)

# Demographic trends in prescribing for each at-risk group
# Use a set difference because there are some categories that are both
# lda subgroups and demographic groups
breakdown_list = list(set(demographics) - set(lda_subgroups.keys()))
for d in breakdown_list:
    m = Measure(
        id=f"antidepressant_any_all_breakdown_{d}_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
    for group_label, group in lda_subgroups.items():
        m = Measure(
            id=f"antidepressant_any_{group_label}_breakdown_{d}_rate",
            numerator=f"antidepressant_any_{group_label}",
            denominator=group,
            group_by=[d],
            small_number_suppression=True,
        )
        measures.append(m)
