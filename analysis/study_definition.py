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
    maoi_codes,
    other_antidepressant_codes,
    depression_codes,
    depression_resolved_codes,
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
        gms_registration_status AND
        age_band != "Unknown" AND
        (sex = "M" OR sex = "F")
        """,
    ),
    # Common demographic variables
    **demographic_variables,
    # QOF DEP003
    **depression_register_variables,
    **dep003_variables,
    # Depression
    depression=patients.satisfying(
        """
        depression_date AND
        NOT depression_resolved
        """,
        depression_date=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            on_or_before="last_day_of_month(index_date)",
            return_expectations={
                "date": {
                    "earliest": start_date,
                    "latest": "last_day_of_month(index_date)",
                },
                "incidence": 0.98,
            },
        ),
        depression_resolved=patients.with_these_clinical_events(
            codelist=depression_resolved_codes,
            returning="binary_flag",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=["depression_date", "last_day_of_month(index_date)"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    # New depression
    new_depression=patients.satisfying(
        """
        depression AND
        NOT previous_depression
        """,
        previous_depression=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=["depression_date - 2 years", "depression_date - 1 day"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    # SSRIs
    antidepressant_ssri=patients.satisfying(
        """
        antidepressant_ssri_date
        """,
        antidepressant_ssri_date=patients.with_these_medications(
            codelist=ssri_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date)",
                "last_day_of_month(index_date)",
            ],
            return_expectations={
                "date": {
                    "earliest": "first_day_of_month(index_date)",
                    "latest": "last_day_of_month(index_date)",
                },
            },
        ),
    ),
    new_antidepressant_ssri=patients.satisfying(
        """
        antidepressant_ssri AND
        NOT previous_ssri
        """,
        previous_ssri=patients.with_these_medications(
            codelist=ssri_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "antidepressant_ssri_date - 2 years",
                "antidepressant_ssri_date - 1 day",
            ],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.01},
    ),
    # Tricyclic
    antidepressant_tricyclic=patients.satisfying(
        """
        antidepressant_tricyclic_date
        """,
        antidepressant_tricyclic_date=patients.with_these_medications(
            codelist=tricyclic_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date)",
                "last_day_of_month(index_date)",
            ],
            return_expectations={
                "date": {
                    "earliest": "first_day_of_month(index_date)",
                    "latest": "last_day_of_month(index_date)",
                },
            },
        ),
    ),
    new_antidepressant_tricyclic=patients.satisfying(
        """
        antidepressant_tricyclic AND
        NOT previous_tricyclic
        """,
        previous_tricyclic=patients.with_these_medications(
            codelist=tricyclic_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "antidepressant_ssri_date - 2 years",
                "antidepressant_ssri_date - 1 day",
            ],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.01},
    ),
    # MAOI
    antidepressant_maoi=patients.satisfying(
        """
        antidepressant_maoi_date
        """,
        antidepressant_maoi_date=patients.with_these_medications(
            codelist=maoi_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date)",
                "last_day_of_month(index_date)",
            ],
            return_expectations={
                "date": {
                    "earliest": "first_day_of_month(index_date)",
                    "latest": "last_day_of_month(index_date)",
                },
            },
        ),
    ),
    new_antidepressant_maoi=patients.satisfying(
        """
        antidepressant_maoi AND
        NOT previous_maoi
        """,
        previous_maoi=patients.with_these_medications(
            codelist=maoi_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "antidepressant_maoi_date - 2 years",
                "antidepressant_maoi_date - 1 day",
            ],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.01},
    ),
    # Other antidepressant
    antidepressant_other_cod=patients.satisfying(
        """
        antidepressant_other_cod_date
        """,
        antidepressant_other_cod_date=patients.with_these_medications(
            codelist=other_antidepressant_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date)",
                "last_day_of_month(index_date)",
            ],
            return_expectations={
                "date": {
                    "earliest": "first_day_of_month(index_date)",
                    "latest": "last_day_of_month(index_date)",
                },
            },
        ),
    ),
    new_antidepressant_other_cod=patients.satisfying(
        """
        antidepressant_other_cod AND
        NOT previous_other
        """,
        previous_other=patients.with_these_medications(
            codelist=other_antidepressant_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "antidepressant_other_cod_date - 2 years",
                "antidepressant_other_cod_date - 1 day",
            ],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.01},
    ),
    antidepressant_other=patients.satisfying(
        """
        antidepressant_maoi OR
        antidepressant_other_cod
        """
    ),
    new_antidepressant_other=patients.satisfying(
        """
        new_antidepressant_maoi OR
        new_antidepressant_other_cod
        """,
    ),
    antidepressant_any=patients.satisfying(
        """
        antidepressant_ssri OR
        antidepressant_tricyclic OR
        antidepressant_other
        """
    ),
    new_antidepressant_any=patients.satisfying(
        """
        new_antidepressant_ssri OR
        new_antidepressant_tricyclic OR
        new_antidepressant_other
        """,
    ),
)

# --- DEFINE MEASURES ---

##  QOF Measures

measures = [
    # Depression register
    Measure(
        id="register_total_rate",
        numerator="depression_register",
        denominator="depression_list_type",
        group_by=["population"],
        small_number_suppression=True,
    ),
    # Depression register by practice
    # Do not need small number suppression b/c we use the decile table
    Measure(
        id="register_practice_rate",
        numerator="depression_register",
        denominator="depression_list_type",
        group_by=["practice"],
    ),
    # QOF achievement over the population
    # This will be restricted to the 18+ population
    Measure(
        id="dep003_total_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["population"],
        small_number_suppression=True,
    ),
    # QOF achievement by practice
    # Do not need small number suppression b/c we use the decile table
    Measure(
        id="dep003_practice_rate",
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=["practice"],
    ),
]

exclusions = ["dep003_denominator_r4", "dep003_denominator_r5", "dep003_denominator_r6"]
# QOF register/achievement/exclusions by each demographic in the config file
for d in demographics:
    m = Measure(
        id="register_{}_rate".format(d),
        numerator="depression_register",
        denominator="depression_list_type",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
    m = Measure(
        id="dep003_{}_rate".format(d),
        numerator="dep003_numerator",
        denominator="dep003_denominator",
        group_by=[d],
        small_number_suppression=True,
    )
    measures.append(m)
    for exclusion in exclusions:
        m = Measure(
            id="dep003_{}_{}_rate".format(exclusion, d),
            numerator=exclusion,
            denominator="depression_register",
            group_by=[d],
            small_number_suppression=True,
        )
        measures.append(m)


## Prescribing measures
outcomes = [
    "depression",
    "antidepressant_ssri",
    "antidepressant_tricyclic",
    "antidepressant_other",
    "antidepressant_any",
]

for o in outcomes:
    # Group rate by outcome
    for group_label, group in lda_subgroups.items():
        m = Measure(
            id="{}_{}_total_rate".format(o, group_label),
            numerator=o,
            denominator=group,
            group_by=["population"],
            small_number_suppression=True,
        )
        measures.append(m)
        new_m = Measure(
            id="new_{}_{}_total_rate".format(o, group_label),
            numerator="new_{}".format(o),
            denominator=group,
            group_by=["population"],
            small_number_suppression=True,
        )
        measures.append(new_m)

# Demographic trends in prescribing for each at-risk group
# Use a set difference because there are some categories that are both
# lda subgroups and demographic groups
for group_label, group in lda_subgroups.items():
    for d in list(set(demographics) - set(lda_subgroups.keys())):
        m = Measure(
            id="antidepressant_any_{}_{}_rate".format(group_label, d),
            numerator="antidepressant_any",
            denominator=group,
            group_by=[d],
            small_number_suppression=True,
        )
        measures.append(m)
