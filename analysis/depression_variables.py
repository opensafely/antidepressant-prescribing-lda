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
    # Date of the latest episode of depression up to and including the
    # achievement date.
    depr_lat=patients.with_these_clinical_events(
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
    depr_lat_count=patients.with_these_clinical_events(
        between=[
            depr_register_date,
            "last_day_of_month(index_date)",
        ],
        codelist=depression_codes,
        return_number_of_matches_in_period=True,
        return_expectations={
            "int": {"distribution": "normal", "mean": 3, "stddev": 1},
            "incidence": 1,
        },
    ),
    depr_lat_code=patients.with_these_clinical_events(
        between=[
            depr_register_date,
            "last_day_of_month(index_date)",
        ],
        codelist=depression_codes,
        returning="code",
        find_last_match_in_period=True,
        return_expectations={
            "category": {"ratios": {"10": 0.2, "11": 0.3, "12": 0.5}}
        },
    ),
    # Date of the first episode of depression up to and including the
    # achievement date.
    depr=patients.with_these_clinical_events(
        between=[
            depr_register_date,
            "last_day_of_month(index_date)",
        ],
        codelist=depression_codes,
        returning="binary_flag",
        find_first_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    # Date of the most recent depression resolved code
    depr_res=patients.with_these_clinical_events(
        codelist=depression_resolved_codes,
        on_or_before="last_day_of_month(index_date)",
        find_last_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        returning="binary_flag",
    ),
    depression_register=patients.satisfying(
        """
        # Select patients from the specified population who have a diagnosis of
        # depression which has not been subsequently resolved
        depression_list_type AND
        (
            (depr AND (NOT depr_res)) OR
            (
                (depr AND depr_res) AND
                (depr_res_date < depr_lat_date)
            )
        )
        """,
    ),
)


depression_indicator_variables = dict(
    # Date variable: reject those with latest depression at least 15 months ago
    # If we use on_or_before and reject, we have no way of knowing if it is
    # the latest episode of depression
    # Instead select those with depression in the last 15 months
    # Possible because the register assumes unresolved depression
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
    depression_15mo_1=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 14 months",
            "depression_15mo_date - 1 day",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    depression_15mo_2=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        find_last_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 14 months",
            "depression_15mo_1_date - 1 day",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
    ),
    depression_15mo_count=patients.with_these_clinical_events(
        codelist=depression_codes,
        between=[
            "first_day_of_month(index_date) - 14 months",
            "last_day_of_month(index_date)",
        ],
        return_number_of_matches_in_period=True,
        return_expectations={
            "int": {"distribution": "normal", "mean": 3, "stddev": 1},
            "incidence": 1,
        },
    ),
    # Date of the first depression review recorded within the period from 10 to
    # 56 days after the patients latest episode of depression up to and
    # including the achievement date.
    review_10_to_56d=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        returning="binary_flag",
        find_first_match_in_period=True,
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        between=[
            "depression_15mo_date + 10 days",
            "depression_15mo_date + 56 days",
        ],
    ),
    # Reject patients passed to this rule who had their depression review at
    # least 12 months before the payment period end date.
    # Instead select those with a review in the last 12 months or never review
    ever_review=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        between=[depr_register_date, "last_day_of_month(index_date)"],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        returning="binary_flag",
    ),
    review_12mo=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        returning="binary_flag",
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.01},
    ),
    review_12mo_count=patients.with_these_clinical_events(
        codelist=depression_review_codes,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        return_number_of_matches_in_period=True,
        return_expectations={
            "int": {"distribution": "normal", "mean": 3, "stddev": 1},
            "incidence": 1,
        },
    ),
    # The most recent date that depression quality indicator care was
    # identified as being unsuitable for the patient up to and including the
    # achievement date.
    unsuitable_12mo=patients.with_these_clinical_events(
        codelist=depression_review_unsuitable_codes,
        returning="binary_flag",
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.01},
    ),
    # The most recent date the patient chose not to receive depression quality
    # indicator care up to and including the achievement date.
    dissent_12mo=patients.with_these_clinical_events(
        codelist=depression_review_dissent_codes,
        returning="binary_flag",
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.01},
    ),
    # Date of the earliest invitation for a depression review on or after the
    # quality service start date and up to and including the achievement date.
    depr_invite_1=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_first_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.015},
    ),
    depr_invite_1_code=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="code",
        find_first_match_in_period=True,
        between=[
            "first_day_of_month(index_date) - 11 months",
            "last_day_of_month(index_date)",
        ],
        return_expectations={
            "category": {"ratios": {"10": 0.2, "11": 0.3, "12": 0.5}}
        },
    ),
    # Date of the earliest invitation for a depression review recorded at least
    # 7 days after the first invitation and up to and including the achievement
    # date.
    depr_invite_2=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="binary_flag",
        find_first_match_in_period=True,
        between=[
            "depr_invite_1_date + 7 days",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.01},
    ),
    depr_invite_2_code=patients.with_these_clinical_events(
        codelist=depression_invitation_codes,
        returning="code",
        find_first_match_in_period=True,
        between=[
            "depr_invite_1_date + 7 days",
            "last_day_of_month(index_date)",
        ],
        return_expectations={
            "category": {"ratios": {"10": 0.2, "11": 0.3, "12": 0.5}}
        },
    ),
    # Date variable: depression diagnosis in the last 3 months
    depression_3mo=patients.with_these_clinical_events(
        codelist=depression_codes,
        returning="binary_flag",
        between=[
            "first_day_of_month(index_date) - 2 months",
            "last_day_of_month(index_date)",
        ],
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={"incidence": 0.01},
    ),
    # Date variable: registered in the last 3 months
    registered_3mo=patients.registered_with_one_practice_between(
        start_date="first_day_of_month(index_date) - 2 months",
        end_date="last_day_of_month(index_date)",
        return_expectations={"incidence": 0.1},
    ),
)

dep003_variables = dict(
    **depression_indicator_variables,
    # Reject patients from the specified population who had their latest
    # episode of depression at least 15 months before the payment period
    # end date. Pass all remaining patients to the next rule.
    # NOTE: reject changed to select
    dep003_denominator_r1=patients.satisfying(
        """
        depression_15mo
        """,
    ),
    # Reject patients passed to this rule who had their depression review
    # at least 12 months before the payment period end date. Pass all
    # remaining patients to the next rule.
    # NOTE: reject changed to select
    dep003_denominator_r2=patients.satisfying(
        """
        dep003_denominator_r1 AND
        (review_12mo OR NOT ever_review)
        """,
    ),
    # Select patients passed to this rule who had a depression review within
    # the period from 10 to 56 days after the patients latest episode of
    # depression. Pass all remaining patients to the next rule.
    # NOTE: select
    dep003_denominator_r3=patients.satisfying(
        """
        dep003_denominator_r2 AND
        review_10_to_56d
        """,
    ),
    # Reject patients passed to this rule for whom depression quality
    # indicator care has been identified as unsuitable for the patient in
    # the 12 months leading up to and including the payment period end
    # date. Pass all remaining patients to the next rule.
    dep003_denominator_r4=patients.satisfying(
        """
        dep003_denominator_r2 AND
        (NOT dep003_denominator_r3) AND
        (NOT unsuitable_12mo)
        """,
    ),
    # Reject patients passed to this rule who chose not to receive
    # depression quality indicator care in the 12 months leading up to and
    # including the payment period end date. Pass all remaining patients
    # to the next rule.
    dep003_denominator_r5=patients.satisfying(
        """
        dep003_denominator_r4 AND
        NOT dissent_12mo
        """,
    ),
    # Reject patients passed to this rule who have not responded to at
    # least two depression care review invitations, made at least 7 days
    # apart, in the 12 months leading up to and including the payment
    # period end date. Pass all remaining patients to the next rule.
    dep003_denominator_r6=patients.satisfying(
        """
        dep003_denominator_r5 AND
        NOT (depr_invite_1 AND depr_invite_2)
        """,
    ),
    # Reject patients passed to this rule whose depression diagnosis was in
    # the 3 months leading up to and including the payment period end date.
    # Pass all remaining patients to the next rule.
    # NOTE: reject changed to select
    dep003_denominator_r7=patients.satisfying(
        """
        dep003_denominator_r6 AND
        NOT depression_3mo
        """,
    ),
    # Reject patients passed to this rule who registered with the GP practice
    # in the 3 month period leading up to and including the payment period end
    # date. Select the remaining patients.
    # NOTE: reject changed to select
    dep003_denominator_r8=patients.satisfying(
        """
        dep003_denominator_r7 AND
        registered_3mo
        """,
    ),
    dep003_denominator=patients.satisfying(
        """
        dep003_denominator_r3 OR
        dep003_denominator_r8
        """,
        return_expectations={"incidence": 0.8},
    ),
    dep003_numerator=patients.satisfying(
        """
        dep003_denominator AND
        review_10_to_56d
        """,
        return_expectations={"incidence": 0.6},
    ),
)
