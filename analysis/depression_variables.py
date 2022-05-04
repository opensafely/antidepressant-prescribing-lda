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

from config import depr_register_date

depression_register_variables = dict(
    # Depression register:  Patients aged at least 18 years old whose latest
    # unresolved episode of depression is since 1st April 2006
    # NOTE: dependency on age and gms_registration_status from
    # demographic_variables.py
    # Demographic variables MUST be loaded before this dictionary in the study
    # If demographics dict is not also going to be loaded, the individual
    # variables should also be loaded into the study definition
    depression_list_type=patients.satisfying(
        """
        gms_registration_status AND
        age>=18 AND
        # Excludes those with unknown age or above 120 (unrealistic)
        age_band != "Unknown"
        """,
    ),
    depression_register=patients.satisfying(
        """
        depression_list_type AND
        latest_depression_date AND
        NOT latest_depression_resolved
        """,
        # Date of the latest first or new episode of depression up to and
        # including the achievement date.
        latest_depression_date=patients.with_these_clinical_events(
            between=[
                depr_register_date,
                "last_day_of_month(index_date)",
            ],
            codelist=depression_codes,
            returning="date",
            find_last_match_in_period=True,
            date_format="YYYY-MM-DD",
        ),
        # Date of the most recent depression resolved code recorded after the
        # most recent depression diagnosis and up to and including the
        # achievement date.
        latest_depression_resolved=patients.with_these_clinical_events(
            on_or_after="latest_depression_date",
            codelist=depression_resolved_codes,
            returning="binary_flag",
            find_last_match_in_period=True,
        ),
    ),
)


depression_indicator_variables = dict(
    # Date variable: reject those with latest depression at least 15 months ago
    # Instead select those with depression in the last 15 months
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
    # Reject patients passed to this rule who had their depression review at
    # least 12 months before the payment period end date.
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
    # Date of the first depression review recorded within the period from 10 to
    # 56 days after the patients latest episode of depression up to and
    # including the achievement date.
    review_10_to_56d=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "depression_15mo_date + 10 days",
            "depression_15mo_date + 56 days",
        ],
    ),
    # The most recent date that depression quality indicator care was
    # identified as being unsuitable for the patient up to and including the
    # achievement date.
    unsuitable_12mo=patients.with_these_clinical_events(
        codelist=depression_review_unsuitable_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        on_or_after="first_day_of_month(index_date) - 11 months",
        return_expectations={"incidence": 0.01},
    ),
    # The most recent date the patient chose not to receive depression quality
    # indicator care up to and including the achievement date.
    dissent_12mo=patients.with_these_clinical_events(
        codelist=depression_review_dissent_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        on_or_after="first_day_of_month(index_date) - 11 months",
        return_expectations={"incidence": 0.01},
    ),
    # Date of the earliest invitation for a depression review recorded at least
    # 7 days after the first invitation and up to and including the achievement
    # date.
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
    # Date of the earliest invitation for a depression review on or after the
    # quality service start date and up to and including the achievement date.
    depr_invite_1=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "depr_invite_2_date - 7 days",
        ],
    ),
    # Date variable: depression diagnosis in the last 3 months
    depression_3mo=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 2 months",
            "last_day_of_month(index_date)",
        ],
    ),
    # Date variable: registered in the last 3 months
    registered_3mo=patients.registered_with_one_practice_between(
        start_date="first_day_of_month(index_date) - 2 months",
        end_date="last_day_of_month(index_date)",
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
        # Reject patients from the specified population who had their latest
        # episode of depression at least 15 months before the payment period
        # end date. Pass all remaining patients to the next rule.
        dep003_denominator_r1=patients.satisfying(
            """
            depression_15mo
            """,
        ),
        # Reject patients passed to this rule who had their depression review
        # at least 12 months before the payment period end date. Pass all
        # remaining patients to the next rule.
        dep003_denominator_r2=patients.satisfying(
            """
            NOT review_before_12mo
            """,
        ),
        # Select patients passed to this rule who had a depression review within
        # the period from 10 to 56 days after the patients latest episode of
        # depression. Pass all remaining patients to the next rule.
        dep003_denominator_r3=patients.satisfying(
            """
            review_10_to_56d
            """,
        ),
        # Reject patients passed to this rule for whom depression quality
        # indicator care has been identified as unsuitable for the patient in
        # the 12 months leading up to and including the payment period end
        # date. Pass all remaining patients to the next rule.
        dep003_denominator_r4=patients.satisfying(
            """
            NOT unsuitable_12mo
            """,
        ),
        # Reject patients passed to this rule who chose not to receive
        # depression quality indicator care in the 12 months leading up to and
        # including the payment period end date. Pass all remaining patients
        # to the next rule.
        dep003_denominator_r5=patients.satisfying(
            """
            NOT dissent_12mo
            """,
        ),
        # Reject patients passed to this rule who have not responded to at
        # least two depression care review invitations, made at least 7 days
        # apart, in the 12 months leading up to and including the payment
        # period end date. Pass all remaining patients to the next rule.
        dep003_denominator_r6=patients.satisfying(
            """
            NOT (depr_invite_1 AND depr_invite_2)
            """,
        ),
        # Reject patients passed to this rule whose depression diagnosis was in
        # the 3 months leading up to and including the payment period end date.
        # Pass all remaining patients to the next rule.
        dep003_denominator_r7=patients.satisfying(
            """
            NOT depression_3mo
            """,
        ),
        # Reject patients passed to this rule who registered with the GP practice
        # in the 3 month period leading up to and including the payment period end
        # date. Select the remaining patients.
        dep003_denominator_r8=patients.satisfying(
            """
            NOT registered_3mo
            """,
        ),
        return_expectations={"incidence": 0.8}
    ),
    dep003_numerator=patients.satisfying(
        """
        dep003_denominator AND
        dep003_denominator_r3
        """,
        return_expectations={"incidence": 0.6},
    ),
)
