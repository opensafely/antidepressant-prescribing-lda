import pathlib
import argparse
import pandas
import fnmatch

from config import start_date

"""
Generate table1 from joined measures file.
Group by category and group, and then further split into columns based on user
provided column names
"""


def get_measure_tables(input_file):
    measure_table = pandas.read_csv(input_file)

    return measure_table


def subset_table(measure_table, measures_pattern, measures_list, date):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """

    measure_table = measure_table[measure_table["date"] == date]

    if measures_pattern:
        measures_list = match_paths(measure_table["name"], measures_pattern)
        if len(measures_list) == 0:
            raise ValueError("Pattern did not match any files")

    if not measures_list:
        return measure_table
    return measure_table[measure_table["name"].isin(measures_list)]


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
    df = df.apply(
        lambda x: series_to_bool(x) if "group" in x.name else x
    )
    for category in df.filter(regex="category"):
        group = f"{category.replace('category', 'group')}"
        if len(df[category].unique()) == 1 and df[group].dtype == "bool":
            df = df[df[group]]
    df["group"] = df[group]
    df["category"] = df[category]
    return df



def transform_percentage(x):
    transformed = (
        x.astype(str)
        + " ("
        + (((x / x.sum()) * 100).round(0)).astype(str)
        + ")"
    )
    transformed.name = f"{x.name} (%)"
    return transformed


def get_percentages(df):
    """
    Create a new column which has count (%) of group
    After computation is complete, replace nan with "REDACTED" again
    """
    percent = df.groupby(level=0).transform(transform_percentage)
    percent.Numerator = percent.Numerator.replace("nan (nan)", "[REDACTED]")
    percent.Denominator = percent.Denominator.replace(
        "nan (nan)", "[REDACTED]"
    )
    percent = percent.rename(
        columns={
            "Numerator": "Numerator (%)",
            "Denominator": "Denominator (%)",
        }
    )
    return percent


def title_format(df):
    df.category = df.category.str.title()
    df.columns = [x.title() for x in df.columns]
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
        help="A list of one or more measure names for rows",
    )
    parser.add_argument(
        "--column-names",
        nargs="*",
        help="Split measures with these names into separate columns",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    output_dir = args.output_dir
    columns = args.column_names

    measure_table = get_measure_tables(input_file)
    subset = subset_table(
        measure_table, measures_pattern, measures_list, start_date
    )

    table1 = pandas.DataFrame()
    for column in columns:
        sub = subset[subset.name.str.contains(column)]
        sub = flatten(sub)
        sub = title_format(sub)
        sub = sub.set_index(["Category", "Group"])
        sub = sub[["Numerator", "Denominator"]]
        sub = sub.apply(pandas.to_numeric, errors="coerce")
        sub = get_percentages(sub)
        sub.columns = pandas.MultiIndex.from_product(
            [[f"{column.title()}"], sub.columns]
        )
        if table1.empty:
            table1 = sub
        else:
            table1 = table1.join(sub)

    table1.to_html(output_dir / "table1.html", index=True)


if __name__ == "__main__":
    main()
