from comparisons import gte, lt, lte
from cohortextractor import pandas_utils

import argparse
import glob
import pathlib
import pandas


def add_register(df):
    df["depression_register_python"] = df["depression_list_type"] & (
        (df["depr"] & ~df["depr_res"])
        | (
            df["depr"]
            & df["depr_res"]
            & lte(df["depr_res_date"], df["depr_lat_date"])
        )
    )
    return df


def add_dep003(df):
    """
    Date of the first depression review recorded within the period from 10 to 56 days after the patients latest episode of depression up to and including the achievement date
    This should be the same as combining rules 1 2 and 3
    """
    df["review_10_to_56d_python"] = (
        df["depression_15mo_date"].notnull()
        & gte(
            pandas.to_datetime(df["review_12mo_date"]),
            pandas.to_datetime(df["depression_15mo_date"])
            + pandas.Timedelta(days=10),
        )
        & lte(
            pandas.to_datetime(df["review_12mo_date"]),
            pandas.to_datetime(df["depression_15mo_date"])
            + pandas.Timedelta(days=56),
        )
    )
    """
    Reject patients passed to this rule who have not responded to at least two depression care review invitations, made at least 7 days apart, in the 12 months leading up to and including the payment period end date. Pass all remaining patients to the next rule
    """
    # TODO: Should we break this down further and get the dates a different way?
    df["dep003_denominator_r6_python"] = df["dep003_denominator_r5"] & ~(
        df["depr_invite_1"] & df["depr_invite_2"]
    )

    return df


def get_extension(path):
    return "".join(path.suffixes)


def read_dataframe(path):
    ext = get_extension(path)
    if ext == ".csv" or ext == ".csv.gz":
        return pandas.read_csv(path, parse_dates=True)
    elif ext == ".feather":
        return pandas.read_feather(path)
    elif ext == ".dta" or ext == ".dta.gz":
        return pandas.read_stata(path)
    else:
        raise ValueError(f"Cannot read '{ext}' files")


def write_dataframe(dataframe, path):
    # We refactored this function, replacing copy-and-paste code from cohort-extractor
    # with a call to cohort-extractor itself. However, the error types differed. We
    # preserved the pre-refactoring error type.
    ext = get_extension(path)
    if ext == ".feather":
        # We write feather files ourselves, because of an issue with dependencies.
        # For more information, see:
        # https://github.com/opensafely-actions/cohort-joiner/issues/25
        dataframe.to_feather(path)
    else:
        try:
            pandas_utils.dataframe_to_file(dataframe, path)
        except RuntimeError:
            raise ValueError(f"Cannot write '{ext}' files")


def get_input_table(input_files):
    for input_file in input_files:
        input_table = read_dataframe(input_file)
        input_table.attrs["fname"] = input_file.name
        yield input_table


def write_input_table(input_table, path):
    write_dataframe(input_table, path)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        required=True,
        type=match_paths,
        help="Glob pattern for matching one or more input files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=get_path,
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.input_files
    output_dir = args.output_dir

    output_dir.mkdir(exist_ok=True)
    for input_table in get_input_table(input_files):
        fname = input_table.attrs["fname"]
        # TODO: come up with a better way to do this
        if "dep003" in fname:
            python_table = add_dep003(input_table)
        elif "register" in fname:
            python_table = add_register(input_table)
        else:
            return
        write_input_table(python_table, output_dir / fname)


if __name__ == "__main__":
    main()
