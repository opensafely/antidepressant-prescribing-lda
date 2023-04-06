from join_and_round import _suppress_column, round_df, redact_df_by_date
import pandas
import pytest


@pytest.fixture
def test_series():
    return pandas.Series([0, 2, 6, 6, 7, 7, 11, 17, 21])


@pytest.fixture
def test_df():
    """
    Redaction should happen per date. If there are multiple groupbys, it
    should happen per groupby, so create a dataframe in which if the
    series were treated as one, it would not need additional redaction
    """
    dates = 4 * [pandas.to_datetime("2022-01-01")] + 4 * [
        pandas.to_datetime("2022-02-01")
    ]
    df = pandas.DataFrame(
        {
            "date": dates,
            "numerator": [2, 21, 22, 23, 4, 24, 25, 26],
            "denominator": [35, 45, 82, 120, 49, 52, 129, 148],
        }
    )
    df["value"] = df["numerator"] / df["denominator"]
    return df


def test_suppress_keep_zeroes(test_series):
    """
    Check there are no numbers less than 5
    Check that all 6's get redacted as well
    But retain true zeroes
    """
    _suppress_column(test_series, redact_zeroes=False)
    assert not ((test_series > 0) & (test_series <= 6)).any()


def test_suppress_6_and_zeroes(test_series):
    """
    Check there are no numbers less than 5
    Check that all 6's get redacted as well
    Check that there are no zeroes
    """
    _suppress_column(test_series, redact_zeroes=True)
    assert not (test_series <= 6).any()


def test_suppress_only_zeroes(test_series):
    """
    Check that zeroes get redacted even if there is no other redaction
    """
    test_series.loc[1] = 9
    _suppress_column(test_series, redact_zeroes=True)
    assert not (test_series <= 5).any()


def test_round_df(test_df):
    """
    Check that all values are divisible by 10
    """
    cols = ["numerator", "denominator"]
    rounded = round_df(test_df, cols)
    assert (rounded[cols].mod(10) == 0).all(None)
    assert (test_df.value != rounded.value).all()


def test_redact_df(test_df):
    """
    Check that redaction happens per-date
    """
    redacted = redact_df_by_date(test_df, redact_zeroes=True)
    assert redacted.numerator.isna().sum() == 4


def test_redact_df_with_groups(test_df):
    """ """
    # Set all to tbe same date
    test_df["date"] = test_df.date[0]
    test_df["group_0"] = test_df.index // 4
    test_df["group_1"] = test_df.index % 4
    redacted = redact_df_by_date(test_df, redact_zeroes=True)
    assert redacted.numerator.isna().sum() == 4
