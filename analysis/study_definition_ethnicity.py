from cohortextractor import (
    StudyDefinition,
    patients,
)

from config import end_date

from codelists import ethnicity_codes_6

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
            "White": "eth='1'",
            "Mixed": "eth='2'",
            "Asian": "eth='3'",
            "Black": "eth='4'",
            "Other": "eth='5'",
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
                        "5": 0.2,
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
                    "Asian": 0.2,
                    "Black": 0.2,
                    "Other": 0.1,
                    "Unknown": 0.1,
                }
            },
        },
    ),
)
