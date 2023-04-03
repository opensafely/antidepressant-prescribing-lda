from join_and_round import redact_df, round_df
import pandas
import pytest


@pytest.fixture
def test_data():
    df = pandas.DataFrame(
        {
            "numerator": [0, 2, 6, 6, 7, 7, 11, 17, 21],
            "denominator": [11, 19, 22, 28, 33, 37, 44, 46, 55],
        }
    )
    df["value"] = df.numerator / df.denominator
    return df


def test_redact_df_keep_zeroes(test_data):
    """
    Check there are no numbers less than 5
    Check that all 6's get redacted as well
    But retain true zeroes
    """
    redact_df(test_data, redact_zeroes=False)
    not_redacted = ((test_data > 0) & (test_data <= 5)).any()
    assert not not_redacted.numerator
    assert not not_redacted.denominator


def test_redact_df_and_zeroes(test_data):
    """
    Check there are no numbers less than 5
    Check that all 6's get redacted as well
    Check that there are no zeroes
    """
    redact_df(test_data, redact_zeroes=True)
    not_redacted = (test_data <= 5).any()
    assert not not_redacted.numerator
    assert not not_redacted.denominator


def test_redact_df_and_zeroes(test_data):
    """
    Check that zeroes get redacted even if there is no other redaction
    """
    test_data.loc[1, "numerator"] = 9
    redact_df(test_data, redact_zeroes=True)
    not_redacted = (test_data <= 5).any()
    assert not not_redacted.numerator
    assert not not_redacted.denominator


def test_round_df(test_data):
    """
    Check that all values are divisible by 10
    """
    rounded = round_df(test_data)
    is_rounded = (rounded.mod(10) == 0).all()
    assert is_rounded.numerator
    assert is_rounded.denominator
