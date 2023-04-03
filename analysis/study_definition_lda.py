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
                    "first_day_of_month(index_date) - 2 years",
                    "first_day_of_month(index_date) - 1 day",
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
    **create_antidepressant_vars(),
    prescription=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "multiple": "(antidepressant_ssri AND antidepressant_tricyclic) OR (antidepressant_ssri AND antidepressant_other) OR (antidepressant_tricyclic AND antidepressant_other)",
            "ssri": "antidepressant_ssri",
            "tricyclic": "antidepressant_tricyclic",
            "other": "antidepressant_other",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "Unknown": 0.8,
                    "multiple": 0.01,
                    "ssri": 0.08,
                    "tricyclic": 0.08,
                    "other": 0.03,
                },
            },
        },
    ),
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
    antidepressant_any_new_18=patients.satisfying(
        """
        antidepressant_any_new AND
        depression_list_type
        """,
    ),
    antidepressant_any_naive_18=patients.satisfying(
        """
        antidepressant_any_naive AND
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
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_new_all_breakdown_diagnosis_18+_rate",
        numerator="antidepressant_any_new_18",
        denominator="antidepressant_any_naive_18",
        group_by=["diagnosis"],
        small_number_suppression=False,
    ),
    Measure(
        id="depression_all_total_rate",
        numerator="depression_register",
        denominator="depression_list_type",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_all_total_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_new_all_total_rate",
        numerator="antidepressant_any_new",
        denominator="antidepressant_any_naive",
        group_by="population",
        small_number_suppression=False,
    ),
    Measure(
        id="antidepressant_any_all_breakdown_prescription_count",
        numerator="antidepressant_any",
        denominator="antidepressant_any",
        group_by=["prescription"],
        small_number_suppression=False,
    ),
]
for group in lda_subgroups:
    m = Measure(
        id=f"antidepressant_any_{group}_breakdown_diagnosis_18+_rate",
        numerator="antidepressant_any_18",
        denominator="depression_list_type",
        group_by=[group, "diagnosis"],
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"antidepressant_any_new_{group}_breakdown_diagnosis_18+_rate",
        numerator="antidepressant_any_new_18",
        denominator="antidepressant_any_naive_18",
        group_by=[group, "diagnosis"],
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"depression_{group}_total_rate",
        numerator="depression_register",
        denominator="depression_list_type",
        group_by=[group],
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"antidepressant_any_{group}_total_rate",
        numerator="antidepressant_any",
        denominator="population",
        group_by=[group],
        small_number_suppression=False,
    )
    measures.append(m)
    new_m = Measure(
        id=f"antidepressant_any_new_{group}_total_rate",
        numerator="antidepressant_any_new",
        denominator="antidepressant_any_naive",
        group_by=[group],
        small_number_suppression=False,
    )
    measures.append(new_m)
    m = Measure(
        id=f"antidepressant_any_{group}_breakdown_prescription_count",
        numerator="antidepressant_any",
        denominator="antidepressant_any",
        group_by=[group, "prescription"],
        small_number_suppression=False,
    )
    measures.append(m)

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
        small_number_suppression=False,
    )
    measures.append(m)
    m = Measure(
        id=f"antidepressant_any_new_all_breakdown_{d}_rate",
        numerator="antidepressant_any_new",
        denominator="antidepressant_any_naive",
        group_by=[d],
        small_number_suppression=False,
    )
    measures.append(m)
    for group in lda_subgroups:
        m = Measure(
            id=f"antidepressant_any_{group}_breakdown_{d}_rate",
            numerator="antidepressant_any",
            denominator="population",
            group_by=[group, d],
            small_number_suppression=False,
        )
        measures.append(m)
        m = Measure(
            id=f"antidepressant_any_new_{group}_breakdown_{d}_rate",
            numerator="antidepressant_any_new",
            denominator="antidepressant_any_naive",
            group_by=[group, d],
            small_number_suppression=False,
        )
        measures.append(m)
