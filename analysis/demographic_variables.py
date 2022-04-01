#####################################################################

# Reusable demographic variables

# Can be imported into a study definition to apply to any population

####################################################################

from cohortextractor import patients
from codelists import learning_disability_codes, carehome_codes

demographic_variables = dict(
    # Age as of the end of the month
    # OS round dob to the first of the month, so add an extra day
    # https://docs.opensafely.org/study-def-variables/#cohortextractor.patients.age_as_of
    age=patients.age_as_of(
        "last_day_of_month(index_date) + 1 day",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
            "incidence": 0.001,
        },
    ),
    age_band=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "0-19": """ age >= 0 AND age < 20""",
            "20-29": """ age >=  20 AND age < 30""",
            "30-39": """ age >=  30 AND age < 40""",
            "40-49": """ age >=  40 AND age < 50""",
            "50-59": """ age >=  50 AND age < 60""",
            "60-69": """ age >=  60 AND age < 70""",
            "70-79": """ age >=  70 AND age < 80""",
            "80+": """ age >=  80 AND age <= 120""",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "Unknown": 0.005,
                    "0-19": 0.125,
                    "20-29": 0.125,
                    "30-39": 0.125,
                    "40-49": 0.125,
                    "50-59": 0.125,
                    "60-69": 0.125,
                    "70-79": 0.125,
                    "80+": 0.12,
                }
            },
        },
    ),
    # Sex
    sex=patients.sex(
        return_expectations={
            "rate": "universal",
            "category": {"ratios": {"M": 0.49, "F": 0.51}},
        }
    ),
    # Index of multiple deprivation
    imd=patients.categorised_as(
        {
            "0": "DEFAULT",
            "1": """index_of_multiple_deprivation >=1 AND index_of_multiple_deprivation < 32844*1/5""",
            "2": """index_of_multiple_deprivation >= 32844*1/5 AND index_of_multiple_deprivation < 32844*2/5""",
            "3": """index_of_multiple_deprivation >= 32844*2/5 AND index_of_multiple_deprivation < 32844*3/5""",
            "4": """index_of_multiple_deprivation >= 32844*3/5 AND index_of_multiple_deprivation < 32844*4/5""",
            "5": """index_of_multiple_deprivation >= 32844*4/5 """,
        },
        index_of_multiple_deprivation=patients.address_as_of(
            "index_date",
            returning="index_of_multiple_deprivation",
            round_to_nearest=100,
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "0": 0.01,
                    "1": 0.20,
                    "2": 0.20,
                    "3": 0.20,
                    "4": 0.20,
                    "5": 0.19,
                }
            },
        },
    ),
    # Region
    region=patients.registered_practice_as_of(
        "index_date",
        returning="nuts1_region_name",
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "North East": 0.1,
                    "North West": 0.1,
                    "Yorkshire and The Humber": 0.1,
                    "East Midlands": 0.1,
                    "West Midlands": 0.1,
                    "East": 0.1,
                    "London": 0.2,
                    "South East": 0.1,
                    "South West": 0.1,
                },
            },
        },
    ),
    # Learning disability
    learning_disability=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "No learning disability": "ld='0'",
            "Learning disability": "ld='1'",
        },
        ld=patients.with_these_clinical_events(
            learning_disability_codes,
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.2},
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "No learning disability": 0.8,
                    "Learning disability": 0.1,
                    "Unknown": 0.1,
                }
            },
        },
    ),
    # Care home
    carehome=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "Not in carehome": "ch='0'",
            "Carehome": "ch='1'",
        },
        ch=patients.with_these_clinical_events(
            carehome_codes,
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.2},
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "Unknown": 0.1,
                    "Not in carehome": 0.8,
                    "Carehome": 0.1,
                }
            },
        },
    ),
    # Practice
    practice=patients.registered_practice_as_of(
        "index_date",
        returning="pseudo_id",
        return_expectations={
            "int": {"distribution": "normal", "mean": 25, "stddev": 5},
            "incidence": 0.5,
        },
    ),
)
