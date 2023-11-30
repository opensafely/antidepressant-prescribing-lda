import pathlib
import argparse
import pandas
import numpy
import fnmatch

"""
Generate table1 from joined measures file.
Group by category and group, and then further split into columns based on user
provided column names
"""


def get_measure_tables(input_file):
    measure_table = pandas.read_csv(
        input_file,
        dtype={"numerator": float, "denominator": float, "value": float},
        na_values="[REDACTED]",
    )

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


def ci_95_proportion(df, scale=1):
    # NOTE: do not assume df has value
    # See formula:
    # https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/PH717-Module6-RandomError12.html
    cis = pandas.DataFrame()
    val = df.numerator / df.denominator
    sd = numpy.sqrt(((val * (1 - val)) / df.denominator))
    cis[0] = scale * (val)
    cis[1] = scale * (val - 1.96 * sd)
    cis[2] = scale * (val + 1.96 * sd)
    return cis


def ci_to_str(ci_df, decimals=1):
    return ci_df.apply(
        lambda x: f"{x[0]:.{decimals}f} ({x[1]:.{decimals}f} to {x[2]:.{decimals}f})",
        axis=1,
    )


def transform_percentage(x):
    transformed = (
        x.map("{:.0f}".format)
        + " ("
        + (((x / x.sum()) * 100).round(1)).astype(str)
        + ")"
    )
    transformed.name = f"{x.name} (%)"
    return transformed


def get_percentages(df, include_denominator, include_rate):
    """
    Create a new column which has count (%) of group
    After computation is complete reconvert numeric to string and replace
    nan with "REDACTED" again
    """
    percent = df.groupby(level=0).transform(transform_percentage)
    percent = percent.replace("nan (nan)", "[REDACTED]")

    cis = ci_95_proportion(df, scale=1000)
    rate = ci_to_str(cis)
    rate = rate.replace("nan (nan to nan)", "[REDACTED]")
    percent["rate"] = rate
    if not include_denominator:
        percent = percent.drop("denominator", axis=1)
    if not include_rate:
        percent = percent.drop("rate", axis=1)
    percent = percent.rename(
        columns={
            "numerator": "No. prescribed antidepressant (%)",
            "denominator": "No. registered patients (%)",
            "rate": "Rate per 1,000 (95% CI)",
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


def reorder_dataframe(df):
    # Pull out Total
    total_mask = df.index.get_level_values(0).str.contains("Total")
    total_row = df[total_mask]
    remaining = df[~total_mask]
    if "Ethnicity" in remaining.index.get_level_values(
        0
    ) and "Ethnicity16" in remaining.index.get_level_values(0):
        eth_mask = remaining.index.get_level_values(0).str.contains(
            "Ethnicity"
        )
        all_eth = remaining[eth_mask]
        remaining = remaining[~eth_mask]
        all_eth_sorted = all_eth.sort_index(level=1)
        all_eth_sorted = all_eth_sorted.drop(("Ethnicity16", "Missing"))
        all_eth_sorted.index = pandas.MultiIndex.from_tuples(
            [
                ("Ethnicity", x)
                for x in list(
                    all_eth_sorted.index.get_level_values(level=1)
                    .str.split("-")
                    .map(lambda x: f"----{x[1]}" if len(x) > 1 else x[0])
                )
            ]
        )
        remaining = pandas.concat([remaining, all_eth_sorted])
    # We need a newer version of pandas to run this on the OS image
    # remaining = remaining.sort_index(key=lambda x: x=="Missing", level=1, sort_remaining=True).sort_index(level=0, sort_remaining=False)
    remaining["sorter"] = remaining.index.get_level_values(1) == "Missing"
    remaining["count"] = range(len(remaining))
    # Sort missing to the bottom of each group, then sort on category alphabetically
    remaining = remaining.sort_values(["sorter", "count"]).sort_index(
        level=0, sort_remaining=False
    )
    remaining = remaining.drop(["sorter", "count"], errors="ignore", axis=1)
    combined = pandas.concat([total_row, remaining])
    return combined


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
    parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Exclude the missing category",
    )
    parser.add_argument(
        "--include-denominator",
        action="store_true",
        help="Include denominator (%)",
    )
    parser.add_argument(
        "--include-rate",
        action="store_true",
        help="Include rate",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Date to select in YYYY-MM-DD format",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    output_dir = args.output_dir
    output_name = args.output_name
    columns = args.column_names
    exclude_missing = args.exclude_missing
    include_denominator = args.include_denominator
    include_rate = args.include_rate
    start_date = args.start_date

    measure_table = get_measure_tables(input_file)
    subset = subset_table(measure_table, measures_pattern, start_date)
    subset = subset.replace("Chinese or Other", "Other Ethnic Groups")

    table1 = pandas.DataFrame()
    for column in columns:
        sub = subset[subset.name.str.contains(column)]
        sub = flatten(sub)
        sub = sub.set_index(["category", "group"])
        sub = sub[["numerator", "denominator"]]
        overall = sub.loc[sub.iloc[0].name[0]].sum()
        overall.name = ("Total", "")
        sub = pandas.concat([pandas.DataFrame(overall).T, sub])
        sub = get_percentages(sub, include_denominator, include_rate)
        sub.columns = pandas.MultiIndex.from_product(
            [[f"{column.title()}"], sub.columns]
        )
        if table1.empty:
            table1 = sub
        else:
            table1 = table1.join(sub)

    table1 = title_multiindex(table1)
    if exclude_missing:
        table1 = table1[table1.index.get_level_values(1) != "Missing"]
    table1 = reorder_dataframe(table1)
    table1.to_html(output_dir / output_name, index=True)


if __name__ == "__main__":
    main()
