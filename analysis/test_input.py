from comparisons import gt

import argparse
import glob
import pathlib
import pandas

import matplotlib.pyplot as plt


def check_register(df, depression_codes_2019):
    before_2006 = df["depression_register"] & (
        pandas.to_datetime(df["depr_lat_date"])
        < pandas.to_datetime("2006-04-01")
    )
    resolved = (
        df["depression_register"]
        & df["depr_res"]
        & df["depr_lat"]
        & (
            gt(
                pandas.to_datetime(df["depr_res_date"]),
                pandas.to_datetime(df["depr_lat_date"]),
            )
        )
    )
    under_18 = df["depression_register"] & (df["age"] < 18)
    resolved_same_day = (
        df["depression_register"]
        & df["depr_lat"]
        & df["depr_res"]
        & (df["depr_lat_date"] == df["depr_res_date"])
    )

    v42_codes = df["depression_register"] & df["depr_lat_code"].isin(
        depression_codes_2019.index
    )

    output = {
        "before_2006": before_2006.sum(),
        "resolved": resolved.sum(),
        "under_18": under_18.sum(),
        "resolved_same_day": resolved_same_day.sum(),
        "v42_codes": v42_codes.sum(),
    }
    return pandas.DataFrame(list(output.items()))


def plot_dates(df):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ever_diff = (
        pandas.to_datetime(df["ever_review_date"])
        - pandas.to_datetime(df["depression_15mo_date"])
    ).astype("timedelta64[D]")
    ever_diff.plot.hist(ax=axes[0], title="Ever Review")
    review_diff = (
        pandas.to_datetime(df["review_12mo_date"])
        - pandas.to_datetime(df["depression_15mo_date"])
    ).astype("timedelta64[D]")
    review_diff.plot.hist(ax=axes[1], title="12 Month Review")


def check_indicator(df, depression_codes_2019):
    # Population
    # Everyone on the register should be of depression list type
    # NOTE: assert=0
    population_1 = df["depression_register"] & ~df["depression_list_type"]

    # R1
    # Test that that r1 only includes those on the register
    # NOTE: assert=0
    r1_1 = ~df["depression_register"] & df["dep003_denominator_r1"]
    # Test that no one with resolved depression is in R1
    # NOTE: assert=0
    r1_2 = (
        df["dep003_denominator_r1"]
        & df["depr_res"]
        & df["depr_lat"]
        & (
            gt(
                pandas.to_datetime(df["depr_res_date"]),
                pandas.to_datetime(df["depr_lat_date"]),
            )
        )
    )
    # Test that no one with resolved depression is in the numerator
    # NOTE: assert=0
    r1_3 = (
        df["dep003_numerator"]
        & df["depr_res"]
        & df["depr_lat"]
        & (
            gt(
                pandas.to_datetime(df["depr_res_date"]),
                pandas.to_datetime(df["depr_lat_date"]),
            )
        )
    )

    # Check if you can have a diagnosis without a date
    # NOTE: assert=0
    r1_4 = df["depression_15mo"] & df["depression_15mo_date"].isnull()

    # R2
    # Test that those who have never had a review are in r2
    # NOTE: assert>0
    r2_1 = df["dep003_denominator_r2"] & ~df["ever_review"]

    # R4
    # Note: assert=0
    # Test that r4 does not have anyone meeting the numerator
    # NOTE: assert=0
    r4_1 = df["dep003_denominator_r4"] & df["review_10_to_56d"]

    # R6
    # Test that no one has invite 2 without 1
    # NOTE assert=0
    r6_1 = df["depr_invite_2"] & ~df["depr_invite_1"]
    # Test that anyone with a second invite is not in rule 6
    # NOTE assert=0
    r6_2 = df["dep003_denominator_r6"] & df["depr_invite_2"]

    # Check if you can have a diagnosis without a date
    # NOTE: assert=0
    r6_3 = df["depr_invite_1"] & df["depr_invite_1_date"].isnull()
    # Check if you can have a diagnosis without a date
    # NOTE: assert=0
    r6_4 = df["depr_invite_2"] & df["depr_invite_2_date"].isnull()

    # Denominator
    # To be in any exclusion criteria and denominator, must be in R3
    # NOTE assert=0
    denominator_1 = (
        ~df["dep003_denominator_r3"]
        & df["dep003_denominator"]
        & (
            df["unsuitable_12mo"]
            | df["dissent_12mo"]
            | df["depr_invite_2"]
            | df["depression_3mo"]
            | ~df["registered_3mo"]
        )
    )
    # Numerator must be in denominator
    # NOTE assert=0
    numerator_1 = df["dep003_numerator"] & ~df["dep003_denominator"]

    numerator_v42 = df["dep003_numerator"] & df["depr_lat_code"].isin(
        depression_codes_2019.index
    )
    denominator_v42 = df["dep003_denominator"] & df["depr_lat_code"].isin(
        depression_codes_2019.index
    )

    output = {
        "population_1": population_1.sum(),
        "r1_1": r1_1.sum(),
        "r1_2": r1_2.sum(),
        "r1_3": r1_3.sum(),
        "r1_4": r1_4.sum(),
        "r2_1": r2_1.sum(),
        "r4_1": r4_1.sum(),
        "r6_1": r6_1.sum(),
        "r6_2": r6_2.sum(),
        "r6_3": r6_3.sum(),
        "r6_4": r6_4.sum(),
        "denominator_1": denominator_1.sum(),
        "numerator_1": numerator_1.sum(),
        "numerator_v42": numerator_v42.sum(),
        "denominator_v42": denominator_v42.sum(),
    }
    return pandas.DataFrame(list(output.items()))


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
    dataframe.to_csv(path)


def get_input_table(input_files):
    for input_file in input_files:
        input_table = read_dataframe(input_file)
        input_table.attrs[
            "fname"
        ] = f"test_{input_file.name.rstrip(''.join(input_file.suffixes)).lstrip('input_')}.csv"
        input_table.attrs[
            "plot_name"
        ] = f"test_{input_file.name.rstrip(''.join(input_file.suffixes)).lstrip('input_')}.png"
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

    depression_codes_2019 = read_dataframe(
        pathlib.Path("codelists/user-ccunningham-depr_cod_qof_v042.csv")
    ).set_index("code")

    output_dir.mkdir(exist_ok=True)
    for input_table in get_input_table(input_files):
        fname = input_table.attrs["fname"]
        # TODO: come up with a better way to do this
        if "dep003" in fname:
            register_results = check_register(
                input_table, depression_codes_2019
            )
            indicator_results = check_indicator(
                input_table, depression_codes_2019
            )
            test_results = pandas.concat([register_results, indicator_results])
            plot_dates(input_table)
            plt.savefig(output_dir / input_table.attrs["plot_name"])
        elif "register" in fname:
            test_results = check_register(input_table, depression_codes_2019)
        else:
            return
        write_input_table(test_results, output_dir / fname)


if __name__ == "__main__":
    main()
