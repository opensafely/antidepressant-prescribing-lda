from cohortextractor import (
    StudyDefinition,
    patients,
)

from config import end_date

from codelists import ethnicity_codes_16

study = StudyDefinition(
    default_expectations={
        "date": {"earliest": "1900-01-01", "latest": "today"},
        "rate": "uniform",
    },
    index_date=end_date,
    population=patients.satisfying(
        """
        NOT has_died
        AND
        registered
        """,
        has_died=patients.died_from_any_cause(
            on_or_before="index_date",
            returning="binary_flag",
        ),
        registered=patients.satisfying(
            "registered_at_start",
            registered_at_start=patients.registered_as_of("index_date"),
        ),
    ),
    # Categories from 2001 census
    # https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups#2001-census
    ethnicity=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "British": "eth='1'",
            "Irish": "eth='2'",
            "Any other White background": "eth='3'",
            "White and Black Caribbean": "eth='4'",
            "White and Black African": "eth='5'",
            "White and Asian": "eth='6'",
            "Any other Mixed background": "eth='7'",
            "Indian": "eth='8'",
            "Pakistani": "eth='9'",
            "Bangladeshi": "eth='10'",
            "Any other Asian background": "eth='11'",
            "Caribbean": "eth='12'",
            "African": "eth='13'",
            "Any other Black background": "eth='14'",
            "Chinese": "eth='15'",
            "Any other": "eth='16'",
        },
        eth=patients.with_these_clinical_events(
            ethnicity_codes_16,
            returning="category",
            find_last_match_in_period=True,
            include_date_of_match=False,
            return_expectations={
                "incidence": 0.75,
                "category": {
                    "ratios": {
                        "1": 0.0625,
                        "2": 0.0625,
                        "3": 0.0625,
                        "4": 0.0625,
                        "5": 0.0625,
                        "6": 0.0625,
                        "7": 0.0625,
                        "8": 0.0625,
                        "9": 0.0625,
                        "10": 0.0625,
                        "11": 0.0625,
                        "12": 0.0625,
                        "13": 0.0625,
                        "14": 0.0625,
                        "15": 0.0625,
                        "16": 0.0625,
                    },
                },
            },
        ),
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "British": 0.0625,
                    "Irish": 0.0625,
                    "Any other White background": 0.0625,
                    "White and Black Caribbean": 0.0625,
                    "White and Black African": 0.0625,
                    "White and Asian": 0.0625,
                    "Any other Mixed background": 0.0625,
                    "Indian": 0.0625,
                    "Pakistani": 0.0625,
                    "Bangladeshi": 0.0625,
                    "Any other Asian background": 0.0625,
                    "Caribbean": 0.0625,
                    "African": 0.05,
                    "Any other Black background": 0.05,
                    "Chinese": 0.05,
                    "Any other": 0.05,
                    "Unknown": 0.05,
                },
            },
        },
    ),
)
