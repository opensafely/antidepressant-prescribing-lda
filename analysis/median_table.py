import pathlib
import argparse
import pandas
import fnmatch

from config import start_date, end_date

"""
Generate median table from joined measures file.
"""


# NOTE: PERCENTAGE
def get_median(df):
    sub = df[["numerator", "denominator"]]
    numeric = sub.apply(pandas.to_numeric, errors="coerce")
    rate = 100 * numeric.numerator / numeric.denominator
    q_25 = round(rate.quantile(0.25), 2)
    q_50 = round(rate.quantile(0.50), 2)
    q_75 = round(rate.quantile(0.75), 2)
    return f"{q_50} ({q_25} - {q_75})"


def get_median_table(measure_table):
    mapping = {
        "Pre-COVID": [start_date, "2020-03-01"],
        "Lockdown": ["2020-03-01", "2021-04-01"],
        "Recovery": ["2021-04-01", end_date],
    }
    columns = []
    for title, (start, end) in mapping.items():
        data = subset_table(measure_table, start, end)
        column = data.groupby(["category", "group"]).apply(
            lambda x: get_median(x)
        )
        column.name = f"{title} Median (IQR)"
        columns.append(column)
    return pandas.concat(columns, axis=1)


def get_measure_tables(input_file, measures_pattern, measures_list):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """
    measure_table = pandas.read_csv(input_file)

    if measures_pattern:
        measures_list = match_paths(measure_table["name"], measures_pattern)
        if len(measures_list) == 0:
            raise ValueError("Pattern did not match any files")

    if not measures_list:
        return measure_table
    return measure_table[measure_table["name"].isin(measures_list)]

    return measure_table


def subset_table(measure_table, start_date, end_date):
    measure_table = measure_table[
        (measure_table["date"] >= start_date)
        & (measure_table["date"] < end_date)
    ]
    return measure_table


def is_bool_as_int(series):
    """Does series have bool values but an int dtype?"""
    # numpy.nan will ensure an int series becomes a float series, so we need to
    # check for both int and float
    if not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_numeric_dtype(series):
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    elif not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_object_dtype(series):
        try:
            series = series.astype(int)
        except ValueError:
            return False
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    else:
        return False


def series_to_bool(series):
    if is_bool_as_int(series):
        return series.astype(int).astype(bool)
    else:
        return series


# NOTE: This will not work if the variable to flatten was last in the groupby
def flatten(df):
    """
    If a category has only one value and the group has boolean values, then
    filter for rows where that value is true
    Create new columns group and category with the last seen group/category
    """
    df = df.dropna(axis=1, how="all")
    df = df.apply(lambda x: series_to_bool(x) if "group" in x.name else x)
    for category in df.filter(regex="category"):
        group = f"{category.replace('category', 'group')}"
        if len(df[category].unique()) == 1 and df[group].dtype == "bool":
            df = df[df[group]]
    df["group"] = df[group]
    df["category"] = df[category]
    return df


def title_multiindex(df):
    titled = [(a.title(), b.title()) for (a, b) in df.index.to_list()]
    names = list(map(str.title, df.index.names))
    df.index = pandas.MultiIndex.from_tuples(titled, names=names)
    return df


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    measures_group = parser.add_mutually_exclusive_group(required=True)
    measures_group.add_argument(
        "--measures-pattern",
        required=False,
        help="Glob pattern matching one or more measures names for rows",
    )
    measures_group.add_argument(
        "--measures-list",
        required=False,
        action="append",
        help="Manually provide a list of one or more measure names for rows",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name for panel plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    output_dir = args.output_dir
    output_name = args.output_name

    measure_table = get_measure_tables(
        input_file, measures_pattern, measures_list
    )
    flattened = flatten(measure_table)
    median_table = get_median_table(flattened)
    titled = title_multiindex(median_table)

    titled.to_html(output_dir / output_name, index=True)


if __name__ == "__main__":
    main()
