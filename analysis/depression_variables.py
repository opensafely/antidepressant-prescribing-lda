#####################################################################

# Reusable DEP003 QOF variables
# * Relevant QOF register
# * Indicator denominator and numerator

# Can be imported into a study definition to apply to any population

####################################################################


from cohortextractor import patients

from codelists import (
    depression_codes,
    depression_resolved_codes,
    depression_review_codes,
    depression_review_unsuitable_codes,
    depression_review_dissent_codes,
    depression_invitation_codes,
)

from config import start_date, depr_register_date

depression_register_variables = dict(
    # The most recent date that the patient registered for GMS, where this registration occurred on or before the achievement date.
    # Define depression register
    # If DEPR_DAT >= 01/04/2006 (defined in config.py)
    # AND DEPRES_DAT = Null
    # AND PAT_AGE >= 18 (defined in demographic_variables.py, handled in study definition)
    # TODO: should age/age_band be top level (sharable), or hidden?
    age_qof=patients.age_as_of(
        "last_day_of_month(index_date) + 1 day",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
            "incidence": 0.001,
        },
    ),
    age_band_qof=patients.categorised_as(
        {
            "Unknown": "DEFAULT",
            "0-19": """ age_qof >= 0 AND age_qof < 20""",
            "20-29": """ age_qof >=  20 AND age_qof < 30""",
            "30-39": """ age_qof >=  30 AND age_qof < 40""",
            "40-49": """ age_qof >=  40 AND age_qof < 50""",
            "50-59": """ age_qof >=  50 AND age_qof < 60""",
            "60-69": """ age_qof >=  60 AND age_qof < 70""",
            "70-79": """ age_qof >=  70 AND age_qof < 80""",
            "80+": """ age_qof >=  80 AND age_qof <= 120""",
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
    depression_register=patients.satisfying(
        """
        currently_registered AND
        ((depression_for_register AND (NOT depression_resolved_register)) OR
        (depression_resolved_register_date <= depression_for_register_date)) AND
        age_qof>=18 AND
        age_band_qof != "Unknown"
        """,
        currently_registered=patients.registered_as_of(
            "index_date",
            return_expectations={"incidence": 0.9},
        ),
        # Date of the latest first or new episode of depression up to and including the achievement date.
        depression_for_register=patients.with_these_clinical_events(
            between=[
                depr_register_date,
                "last_day_of_month(index_date)",
            ],
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
        ),
        depression_resolved_register=patients.with_these_clinical_events(
            on_or_before="last_day_of_month(index_date)",
            codelist=depression_resolved_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
        ),
    ),
)


depression_indicator_variables = dict(
    # Had depression within the last 15 months
    depression_15mo=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 14 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    # Review at least 12 months before index_date
    review_before_12mo=patients.satisfying(
        """
        ever_review AND (NOT review_within_12mo)
        """,
        ever_review=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[depr_register_date, "last_day_of_month(index_date)"],
        ),
        review_within_12mo=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "first_day_of_month(index_date) - 11 months",
                "last_day_of_month(index_date)",
            ],
        ),
    ),
    # Had a review within 10 to 56 days of latest depression
    review_10_to_56d=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "depression_15mo_date + 10 days",
            "depression_15mo_date + 56 days",
        ],
    ),
    unsuitable_12mo=patients.with_these_clinical_events(
        codelist=depression_review_unsuitable_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        on_or_after="first_day_of_month(index_date) - 11 months",
        return_expectations={"incidence": 0.01},
    ),
    dissent_12mo=patients.with_these_clinical_events(
        codelist=depression_review_dissent_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        on_or_after="first_day_of_month(index_date) - 11 months",
        return_expectations={"incidence": 0.01},
    ),
    # Latest depression invite date
    depr_invite_2=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    # Latest depression invite date 7 days before the last one
    depr_invite_1=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "depr_invite_2_date - 7 days",
        ],
    ),
    depression_3mo=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 2 months",
            "last_day_of_month(index_date)",
        ],
    ),
    # Reject patients passed to this rule who registered with the GP practice in the 3 month period
    # leading up to and including the payment period end date.
    # Select the remaining patients.
    registered_3mo=patients.registered_with_one_practice_between(
        start_date="index_date - 3 months",
        end_date="index_date",
        return_expectations={"incidence": 0.1},
    ),
)

dep003_variables = dict(
    dep003_denominator=patients.satisfying(
        """
        depression_register AND
        dep003_denominator_r1 AND
        dep003_denominator_r2 AND
        (
            dep003_denominator_r3 
            OR
            (
                dep003_denominator_r4 AND
                dep003_denominator_r5 AND
                dep003_denominator_r6 AND
                dep003_denominator_r7 AND
                dep003_denominator_r8
            )
        )
        """,
        **depression_indicator_variables,
        dep003_denominator_r1=patients.satisfying(
            """
            depression_15mo
            """,
        ),
        # REJECT those who had their depression review at least 12 months before PPED
        dep003_denominator_r2=patients.satisfying(
            """
            NOT review_before_12mo
            """,
        ),
        dep003_denominator_r3=patients.satisfying(
            """
            review_10_to_56d
            """,
        ),
        dep003_denominator_r4=patients.satisfying(
            """
            NOT unsuitable_12mo
            """,
        ),
        dep003_denominator_r5=patients.satisfying(
            """
            NOT dissent_12mo
            """,
        ),
        # TODO: Do we need to check that they did not respond?
        dep003_denominator_r6=patients.satisfying(
            """
            NOT (depr_invite_1 AND depr_invite_2)
            """,
        ),
        dep003_denominator_r7=patients.satisfying(
            """
            NOT depression_3mo
            """,
        ),
        dep003_denominator_r8=patients.satisfying(
            """
            NOT registered_3mo
            """,
        ),
    ),
    dep003_numerator=patients.satisfying(
        """
        dep003_denominator AND 
        dep003_denominator_r3
        """,
    ),
)
