from cohortextractor import (
    StudyDefinition,
    patients,
)

from config import start_date

from codelists import ethnicity_codes_6

study = StudyDefinition(
    default_expectations={
        "date": {"earliest": "1900-01-01", "latest": "today"},
        "rate": "uniform",
    },
    index_date=start_date,
    # Here we extract from all patients because we are only extracting
    # ethnicity at one time point. If we restrict this to our study population,
    # cohorts extracted at another time may not be included in this cohort
    population=patients.all(),
    # Categories from 2001 census
    # https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups#2001-census
    ethnicity=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "White": "eth='1'",
            "Mixed": "eth='2'",
            "Asian or Asian British": "eth='3'",
            "Black or Black British": "eth='4'",
            "Chinese or Other": "eth='5'",
        },
        eth=patients.with_these_clinical_events(
            ethnicity_codes_6,
            returning="category",
            find_last_match_in_period=True,
            include_date_of_match=False,
            return_expectations={
                "category": {
                    "ratios": {
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.2,
                        "4": 0.2,
                        "5": 0.1,
                    }
                },
                "incidence": 0.75,
            },
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "White": 0.2,
                    "Mixed": 0.2,
                    "Asian or Asian British": 0.1,
                    "Black or Black British": 0.2,
                    "Chinese or Other": 0.1,
                    "Unknown": 0.2,
                }
            },
        },
    ),
)
