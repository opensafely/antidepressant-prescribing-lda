import argparse
import pathlib
import re
import glob
import numpy
import pandas

MEASURE_FNAME_REGEX = re.compile(r"measure_(?P<id>\S+)\.csv")


def _check_for_practice(table):
    if "practice" in table.filter(regex="category").values:
        raise (
            AssertionError("Practice-level data should not be in final output")
        )


def _round_column(column, base=10):
    """
    Rounds values to nearest multiple of base.
    Args:
        x: Value to round
        base: Base to round to
        Returns:
            Rounded value
    """
    return (column / base).round() * base


def _suppress_column(
    column, data, small_number_threshold=5, redact_zeroes=False
):
    small_value_filter = (data[column] > 0) & (
        data[column] <= small_number_threshold
    )
    large_value_filter = data[column] > small_number_threshold

    num_small_values = data.loc[small_value_filter, column].count()
    num_large_values = data.loc[large_value_filter, column].count()

    # Redact zeroes before any return
    if redact_zeroes:
        data.loc[data[column] == 0, column] = numpy.nan

    if num_small_values == 0:
        return

    small_value_total = data.loc[small_value_filter, column].sum()
    data.loc[small_value_filter, column] = numpy.nan

    if small_value_total > small_number_threshold:
        return
    if num_large_values == 0:
        return

    # If the total suppressed is small then reidentification may
    # be possible by comparing the total of the unsuppressed
    # values with the population. So we suppress further values to
    # take the total over the threshold. If there are multiple
    # rows with the next smallest value, it may be possible to
    # change the query to reorder the rows and thus reveal their
    # values; so we suppress all of them.
    next_smallest_value = data.loc[large_value_filter, column].min()
    all_next_smallest = data[column] == next_smallest_value
    data.loc[all_next_smallest, column] = numpy.nan


def redact_df(measure_table, redact_zeroes=False):
    _suppress_column("numerator", measure_table, redact_zeroes=redact_zeroes)
    _suppress_column("denominator", measure_table, redact_zeroes=redact_zeroes)


def round_df(measure_table, base=10):
    measure_table.numerator = _round_column(measure_table.numerator, base=base)
    measure_table.denominator = _round_column(
        measure_table.denominator, base=base
    )
    measure_table.value = measure_table.numerator / measure_table.denominator
    return measure_table


def _reshape_data(measure_table):
    if measure_table.date[0] != measure_table.date[1]:
        # if sequential rows have different dates, then an individual date's
        # data has not been subdivided by category, and we can assume that
        # group_by = "population"
        # Therefore, the numerator and denominator will be the first columns
        numerator = measure_table.columns[0]
        denominator = measure_table.columns[1]
        measure_table["category_0"] = "population"
        measure_table["group_0"] = "population"
        group_by = None
        measure_table["name"] = measure_table.attrs["id"]
        measure_table.rename(
            columns={
                numerator: "numerator",
                denominator: "denominator",
                group_by: "group",
            },
            inplace=True,
        )
        # Assume we only need the numerator and the denominator
        measure_table.drop(
            columns=["population"], inplace=True, errors="ignore"
        )
        return measure_table

    else:
        # No denominator, just a count
        # NOTE: denominator reconstruction may not work if already redacted
        if "count" in measure_table.attrs["id"]:
            name = measure_table.attrs["id"]
            copy = measure_table.copy()
            group_by = copy.columns[:-3]
            numerator = copy.columns[-3]
            # Assume last groupby is the count
            group_by_count = list(group_by[:-1]) + ["date"]
            total = copy.groupby(group_by_count).sum()[numerator]
            total.name = "denominator"
            copy = copy.set_index(group_by_count)
            copy = pandas.merge(copy, total, left_index=True, right_index=True)
            measure_table = copy.reset_index()
            measure_table["name"] = name
            denominator = "denominator"
        else:
            group_by = measure_table.columns[:-4]
            denominator = measure_table.columns[-3]
            numerator = measure_table.columns[-4]
            measure_table["name"] = measure_table.attrs["id"]
        for index, group in enumerate(group_by):
            measure_table[f"category_{index}"] = group
            measure_table[f"group_{index}"] = measure_table[group]

        measure_table.rename(
            columns={
                numerator: "numerator",
                denominator: "denominator",
            },
            inplace=True,
        )

        keep = [
            x
            for x in measure_table.columns
            if x in ["name", "numerator", "denominator", "date", "value"]
            or ("group" in x)
            or ("category" in x)
        ]
        return measure_table[keep]


def _join_tables(tables):
    return pandas.concat(tables)


def get_measure_tables(input_files):
    for input_file in input_files:
        measure_fname_match = re.match(MEASURE_FNAME_REGEX, input_file.name)
        if measure_fname_match is not None:
            # The `date` column is assigned by the measures framework.
            measure_table = pandas.read_csv(input_file, parse_dates=["date"])
            measure_table.attrs["id"] = measure_fname_match.group("id")
            yield measure_table


def _redacted_string(measure_table):
    """
    Replace redacted values with "[REDACTED]" string
    A group could have the name NaN, so apply to specific columns
    """
    REDACTED_STR = "[REDACTED]"
    measure_table.numerator = measure_table.numerator.replace(
        numpy.nan, REDACTED_STR
    )
    measure_table.denominator = measure_table.denominator.replace(
        numpy.nan, REDACTED_STR
    )
    measure_table.value = measure_table.value.replace(numpy.nan, REDACTED_STR)
    return measure_table


def write_table(measure_table, path, filename):
    create_dir(path)
    measure_table.to_csv(path / filename, index=False, header=True)


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_input(input_list):
    path = get_path(input_list)
    if path.exists():
        return path


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def parse_args():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-files",
        required=False,
        type=match_paths,
        help="Glob pattern for matching one or more input files",
    )
    input_group.add_argument(
        "--input-list",
        required=False,
        type=match_input,
        action="append",
        help="Manually provide a list of one or more input files",
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
        help="Name for joined measures file",
    )
    parser.add_argument(
        "--skip-redaction",
        action="store_true",
        help="Do not redact numbers below 5",
    )
    parser.add_argument(
        "--skip-round",
        action="store_true",
        help="Do not round numbers",
    )
    parser.add_argument(
        "--keep-zeroes",
        action="store_true",
        help="Do not redact zeroes",
    )
    parser.add_argument(
        "--round-to",
        required=False,
        default=10,
        type=int,
        help="Round to the nearest",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.input_files
    input_list = args.input_list
    output_dir = args.output_dir
    output_name = args.output_name
    skip_redaction = args.skip_redaction
    skip_round = args.skip_round
    keep_zeroes = args.keep_zeroes
    round_to = args.round_to

    if not input_files and not input_list:
        raise FileNotFoundError("No files matched the input pattern provided")

    tables = []
    for measure_table in get_measure_tables(input_list or input_files):
        table = _reshape_data(measure_table)
        if not skip_redaction:
            redact_df(table, redact_zeroes=(not keep_zeroes))
        if not skip_round:
            table = round_df(table, base=round_to)
        redacted_str = _redacted_string(table)
        tables.append(redacted_str)

    output = _join_tables(tables)
    _check_for_practice(output)
    write_table(output, output_dir, output_name)


if __name__ == "__main__":
    main()
