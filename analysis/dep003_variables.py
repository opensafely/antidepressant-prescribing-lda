from cohortextractor import patients

from codelists import (
    depression_codes,
    depression_review_codes,
    depression_review_unsuitable_codes,
    depression_review_dissent_codes,
)

dep003_variables = dict(
    depression_date=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="date",
        find_last_match_in_period=True,
        between=["index_date", "last_day_of_month(index_date)"],
        return_expectations={"incidence": 0.1},
    ),
    # QOF DEP003
    denominator=patients.satisfying(
        """
    # Denominator
    # Rule 1: Reject patients who had their latest episode of depression at least 15 months before end date 
    NOT (depression_15mo)
    AND
    # Rule 2: Reject patients who had their depression review at least 12 months before end date
    NOT (review_12mo)
    AND
    # Rule 3: Select patients with a review between 10 to 56 days after late episode of depression
    # Rule 4: Reject unsuitable in the 12 months leading up to pped
    NOT (unsuitable)
    AND
    # Rule 5: Reject informed dissent in thr 12 months leading up to pped
    NOT (dissent)
    # Rule 6: Reject non-response to review invitations
    # Rule 7: Reject those with a diagnosis 3 months prior to end date
    # Rule 8: Reject those registered with the practice 3 months prior to end date

    """,
        depression_15mo=patients.with_these_clinical_events(
            codelist=depression_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_before="last_day_of_nhs_financial_year(index_date) - 15 months",
        ),
        review_12mo=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_before="last_day_of_nhs_financial_year(index_date) - 12 months",
        ),
        unsuitable=patients.with_these_clinical_events(
            codelist=depression_review_unsuitable_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_before="last_day_of_nhs_financial_year(index_date) - 12 months",
        ),
        dissent=patients.with_these_clinical_events(
            codelist=depression_review_dissent_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            on_or_before="last_day_of_nhs_financial_year(index_date) - 12 months",
        ),
    ),
    # Numerator
    # Rule 1: Select those in the denominator who had a review within 10 to 56 days after latest episode
    event=patients.satisfying(
        """
        denominator AND
        had_review
        """,
        had_review=patients.with_these_clinical_events(
            codelist=depression_review_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
            between=["depression_date + 10 days", "depression_date + 56 days"],
        ),
    ),
)
