#####################################################################

# Reusable demographic variables

# Can be imported into a study definition to apply to any population

####################################################################

from cohortextractor import patients
from codelists import learning_disability_codes, carehome_codes, autism_codes

# NOTE: Demographic variables are defined using last_day_of_month(index_date)
# rather than the first of the month in order to be QOF compliant.
# This may be different than other analytic studies

demographic_variables = dict(
    # age_as_of rounds DOB to the first of the month, so add an extra day to
    # move into the next month
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
            "Unknown": "DEFAULT",
            "1 - most deprived": """index_of_multiple_deprivation >=1 AND index_of_multiple_deprivation < 32844*1/5""",
            "2": """index_of_multiple_deprivation >= 32844*1/5 AND index_of_multiple_deprivation < 32844*2/5""",
            "3": """index_of_multiple_deprivation >= 32844*2/5 AND index_of_multiple_deprivation < 32844*3/5""",
            "4": """index_of_multiple_deprivation >= 32844*3/5 AND index_of_multiple_deprivation < 32844*4/5""",
            "5 - least deprived": """index_of_multiple_deprivation >= 32844*4/5 """,
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
                    "Unknown": 0.01,
                    "1 - most deprived": 0.20,
                    "2": 0.20,
                    "3": 0.20,
                    "4": 0.20,
                    "5 - least deprived": 0.19,
                }
            },
        },
    ),
    # Region
    region=patients.registered_practice_as_of(
        "last_day_of_month(index_date)",
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
            "No record of learning disability": "ld='0'",
            "Record of learning disability": "ld='1'",
        },
        ld=patients.with_these_clinical_events(
            learning_disability_codes,
            on_or_before="last_day_of_month(index_date)",
            returning="binary_flag",
            return_expectations={"incidence": 0.2},
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "No record of learning disability": 0.8,
                    "Record of learning disability": 0.1,
                    "Unknown": 0.1,
                }
            },
        },
    ),
    # Autism
    autism=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "No record of autism": "aut='0'",
            "Record of autism": "aut='1'",
        },
        aut=patients.with_these_clinical_events(
            autism_codes,
            on_or_before="index_date",
            returning="binary_flag",
            return_expectations={"incidence": 0.3},
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "No record of autism": 0.7,
                    "Record of autism": 0.2,
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
            on_or_before="last_day_of_month(index_date)",
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
        "last_day_of_month(index_date)",
        returning="pseudo_id",
        return_expectations={
            "int": {"distribution": "normal", "mean": 25, "stddev": 5},
            "incidence": 0.5,
        },
    ),
    # The most recent date that the patient registered for GMS, where this
    # registration occurred on or before the achievement date.
    gms_registration_status=patients.satisfying(
        """
        registered AND
        NOT has_died
        """,
        registered=patients.registered_as_of(
            "last_day_of_month(index_date)",
            return_expectations={"incidence": 0.9},
        ),
        has_died=patients.died_from_any_cause(
            on_or_before="last_day_of_month(index_date)",
            returning="binary_flag",
        ),
    ),
)
