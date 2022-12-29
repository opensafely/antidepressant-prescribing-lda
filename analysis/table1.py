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


def subset_table(measure_table, measures_pattern, date):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """

    measure_table = measure_table[measure_table["date"] == date]

    measures_list = []
    for pattern in measures_pattern:
        paths_to_add = match_paths(measure_table["name"], pattern)
        if len(paths_to_add) == 0:
            raise ValueError(f"Pattern did not match any rows: {pattern}")
        measures_list += paths_to_add

    table_subset = measure_table[measure_table["name"].isin(measures_list)]

    if table_subset.empty:
        raise ValueError("Patterns did not match any rows")

    return table_subset


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
    percent.numerator = percent.numerator.replace("nan (nan)", "[REDACTED]")
    percent.denominator = percent.denominator.replace(
        "nan (nan)", "[REDACTED]"
    )
    percent = percent.rename(
        columns={
            "numerator": "No. prescribed antidepressant (%)",
            "denominator": "No. registered patients (%)",
        }
    )
    return percent


def title_multiindex(df):
    titled = []
    # NOTE: dataframe must be sorted, otherwise new index may not match
    df = df.sort_index()
    for category, data in df.groupby(level=0):
        category = category.replace("_", " ")
        group = data.index.get_level_values(1).to_series()
        group = group.replace({"Unknown": "Missing"})
        group = group.fillna("Missing")
        group = series_to_bool(group)
        titled += [
            (str(category).title(), str(item).title())
            for item in group.to_list()
        ]
    df.index = pandas.MultiIndex.from_tuples(titled)
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
    parser.add_argument(
        "--measures-pattern",
        required=True,
        action="append",
        help="Glob pattern matching one or more measures names for rows",
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
    output_dir = args.output_dir
    output_name = args.output_name
    columns = args.column_names

    measure_table = get_measure_tables(input_file)
    subset = subset_table(measure_table, measures_pattern, start_date)

    table1 = pandas.DataFrame()
    for column in columns:
        sub = subset[subset.name.str.contains(column)]
        sub = flatten(sub)
        sub = sub.set_index(["category", "group"])
        sub = sub[["numerator", "denominator"]]
        sub = sub.apply(pandas.to_numeric, errors="coerce")
        overall = sub.loc[sub.iloc[0].name[0]].sum()
        overall.name = ("Total", "")
        sub = pandas.concat([pandas.DataFrame(overall).T, sub])
        sub = get_percentages(sub)
        sub.columns = pandas.MultiIndex.from_product(
            [[f"{column.title()}"], sub.columns]
        )
        if table1.empty:
            table1 = sub
        else:
            table1 = table1.join(sub)

    table1 = title_multiindex(table1)
    table1.to_html(output_dir / output_name, index=True)


if __name__ == "__main__":
    main()
