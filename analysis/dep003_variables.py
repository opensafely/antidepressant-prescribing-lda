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

dep003_variables = dict(
    # TODO: add age back in: age >= 18 AND age <110
    # TODO: move composition of variables?
    depression_register=patients.satisfying(
        """
        depression_since_register_date AND
        NOT depression_register_resolved
        """,
        depression_since_register_date=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="date",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[depr_register_date, "last_day_of_month(index_date)"],
            return_expectations={
                "date": {"earliest": depr_register_date, "latest": "index_date"},
                "incidence": 0.98,
            },
        ),
        depression_register_resolved=patients.with_these_clinical_events(
            codelist=depression_resolved_codes,
            returning="binary_flag",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=["depression_since_register_date", "last_day_of_month(index_date)"],
            return_expectations={"incidence": 0.01},
        ),
        return_expectations={"incidence": 0.1},
    ),
    financial_year=patients.satisfying(
        """
        depression_15mo AND
        depression_register AND
        (NOT ever_review OR (ever_review AND review_12mo))
        """,
        # Had depression within the last 15 months
        depression_15mo=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "last_day_of_nhs_financial_year(index_date) - 15 months",
                "last_day_of_month(index_date)",
            ],
        ),
        ever_review=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[depr_register_date, "last_day_of_month(index_date)"],
        ),
        review_12mo=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=[
                "last_day_of_nhs_financial_year(index_date) - 12 months",
                "last_day_of_month(index_date)",
            ],
        ),
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
    # TODO: do we actually need the date for the first invite?
    depr_invite_1=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "depr_invite_2_date - 7 days",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    # TODO: how do we define "not responded"
    # They do not have a review and not dissent and not unsuitable?
    # Would have already rejected those people?
    # TODO: could handle the 7 days in here?
    dep003_denominator_6=patients.satisfying(
        """
        depr_invite_1 AND depr_invite_2
        """,
    ),
    numerator=patients.satisfying(
        """
        financial_year AND
        had_review
        """,
        had_review=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            date_format="YYYY-MM-DD",
            find_last_match_in_period=True,
            between=[
                "depression_since_register_date + 10 days",
                "depression_since_register_date + 56 days",
            ],
            return_expectations={"incidence": 0.6},
        ),
    ),
    denominator=patients.satisfying(
        """
        numerator OR
        financial_year AND
        (NOT unsuitable AND NOT dissent)
        """,
        dissent=patients.with_these_clinical_events(
            codelist=depression_review_dissent_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_after="last_day_of_nhs_financial_year(index_date) - 12 months",
            return_expectations={"incidence": 0.01},
        ),
        unsuitable=patients.with_these_clinical_events(
            codelist=depression_review_unsuitable_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_after="last_day_of_nhs_financial_year(index_date) - 12 months",
            return_expectations={"incidence": 0.01},
        ),
    ),
)
