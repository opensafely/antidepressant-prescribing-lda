from cohortextractor import (
    StudyDefinition,
    patients,
)

from config import start_date

from codelists import ethnicity_codes_16

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
    eth16=patients.with_these_clinical_events(
        ethnicity_codes_16,
        returning="category",
        find_last_match_in_period=True,
        include_date_of_match=False,
        return_expectations={
            "category": {
                "ratios": {
                    "1": 0.1,
                    "2": 0.1,
                    "3": 0.1,
                    "4": 0.1,
                    "5": 0.05,
                    "6": 0.05,
                    "7": 0.05,
                    "8": 0.05,
                    "9": 0.05,
                    "10": 0.05,
                    "11": 0.05,
                    "12": 0.05,
                    "13": 0.05,
                    "14": 0.05,
                    "15": 0.05,
                    "16": 0.05,
                }
            },
            "incidence": 0.75,
        },
    ),
    ethnicity=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "White": "eth16='1' OR eth16='2' OR eth16='3'",
            "Mixed": "eth16='4' OR eth16='5' OR eth16='6' OR eth16='7'",
            "Asian or Asian British": "eth16='8' OR eth16='9' OR eth16='10' OR eth16='11'",
            "Black or Black British": "eth16='12' OR eth16='13' OR eth16='14'",
            "Chinese or Other": "eth16='15' OR eth16='16'",
        },
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
    ethnicity16=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "White-British": "eth16='1'",
            "White-Irish": "eth16='2'",
            "White-Any other White background": "eth16='3'",
            "Mixed-White and Black Caribbean": "eth16='4'",
            "Mixed-White and Black African": "eth16='5'",
            "Mixed-White and Asian": "eth16='6'",
            "Mixed-Any other mixed background": "eth16='7'",
            "Asian or Asian British-Indian": "eth16='8'",
            "Asian or Asian British-Pakistani": "eth16='9'",
            "Asian or Asian British-Bangladeshi": "eth16='10'",
            "Asian or Asian British-Any other Asian background": "eth16='11'",
            "Black or Black British-Caribbean": "eth16='12'",
            "Black or Black British-African": "eth16='13'",
            "Black or Black British-Any other Black background": "eth16='14'",
            "Other Ethnic Groups-Chinese": "eth16='15'",
            "Other Ethnic Groups-Any other ethnic group": "eth16='16'",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "Unknown": 0.1,
                    "White-British": 0.1,
                    "White-Irish": 0.05,
                    "White-Any other White background": 0.1,
                    "Mixed-White and Black Caribbean": 0.05,
                    "Mixed-White and Black African": 0.05,
                    "Mixed-White and Asian": 0.05,
                    "Mixed-Any other mixed background": 0.05,
                    "Asian or Asian British-Indian": 0.05,
                    "Asian or Asian British-Pakistani": 0.05,
                    "Asian or Asian British-Bangladeshi": 0.05,
                    "Asian or Asian British-Any other Asian background": 0.05,
                    "Black or Black British-Caribbean": 0.05,
                    "Black or Black British-African": 0.05,
                    "Black or Black British-Any other Black background": 0.05,
                    "Other Ethnic Groups-Chinese": 0.05,
                    "Other Ethnic Groups-Any other ethnic group": 0.05,
                }
            },
        },
    ),
)
