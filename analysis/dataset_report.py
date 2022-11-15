import argparse
import functools
import glob
import pathlib

import jinja2
import numpy
import pandas
import dateutil
import datetime
from pandas.api import types


# Template
# --------


@functools.singledispatch
def finalize(value):
    """Processes the value of a template variable before it is rendered."""
    # This is the default "do nothing" path.
    return value


@finalize.register
def _(value: pandas.DataFrame):
    return value.to_html()


ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader("analysis/templates"),
    finalize=finalize,
)
TEMPLATE = ENVIRONMENT.get_template("dataset_report.html")


# Application
# -----------


def get_extension(path):
    return "".join(path.suffixes)


def get_name(path):
    return path.name.split(".")[0]


def read_dataframe(path):
    from_csv = False
    if (ext := get_extension(path)) in [".csv", ".csv.gz"]:
        from_csv = True
        dataframe = pandas.read_csv(path)
    elif ext in [".feather"]:
        dataframe = pandas.read_feather(path)
    elif ext in [".dta", ".dta.gz"]:
        dataframe = pandas.read_stata(path)
    else:
        raise ValueError(f"Cannot read '{ext}' files")
    # It's useful to know whether a dataframe was read from a csv when summarizing the
    # columns later.
    dataframe.attrs["from_csv"] = from_csv
    # We give the column index a name now, because it's preserved when summaries are
    # computed later.
    dataframe.columns.name = "Column Name"
    return dataframe


def is_empty(series):
    """Does series contain only missing values?"""
    return series.isna().all()


def get_table_summary(dataframe):
    memory_usage = dataframe.memory_usage(index=False)
    memory_usage = memory_usage / 1_000 ** 2
    return pandas.DataFrame(
        {
            "Size (MB)": memory_usage,
            "Data Type": dataframe.dtypes,
            "Empty": dataframe.apply(is_empty),
        },
    )


def is_bool_as_int(series):
    """Does series have bool values but an int dtype?"""
    # numpy.nan will ensure an int series becomes a float series, so we need to check
    # for both int and float
    if not types.is_bool_dtype(series) and types.is_numeric_dtype(series):
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    else:
        return False


def is_date_as_obj(series):
    try:
        pandas.to_datetime(series)
        return True
    except dateutil.parser._parser.ParserError:
        return False


def parse_os_year(series):
    """
    OS date formats allow "YYYY" "YYYY-MM" or "YYYY-MM-DD"
    If month or date are included, the value will be a string
    But if only the year is included, it will be a float
    """
    if series.dtype == "float64" or series.dtype == "int64":
        try:
            return series.apply(
                lambda x: datetime.datetime.strptime(str(int(x)), "%Y").year
                if not numpy.isnan(x)
                else numpy.nan
            )
        except Exception:
            return pandas.Series(len(series) * [numpy.nan])
    else:
        try:
            return series.apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year
                if not pandas.isnull(x)
                else numpy.nan
            )
        except Exception:
            pass
        try:
            return series.apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m").year
                if not pandas.isnull(x)
                else numpy.nan
            )
        except Exception:
            pass
        try:
            return series.apply(
                lambda x: datetime.datetime.strptime(x, "%Y").year
                if not pandas.isnull(x)
                else numpy.nan
            )
        except Exception:
            return pandas.Series(len(series) * [numpy.nan])


def redact_round_series(series_in):
    """Redacts counts <= 7 and rounds counts to nearest 5"""
    # If we are going to have to redact the next smallest
    redact_extra = series_in[(series_in > 0) & (series_in <= 7)].sum()
    # Redact <= 7
    series_out = series_in.apply(
        lambda x: numpy.nan if x > 0 and x <= 7 else x
    )
    if redact_extra > 0 and redact_extra <= 7:
        series_out[series_out == series_out.min()] = numpy.nan
    rounded = series_out.apply(
        lambda x: 5 * round(x / 5) if not numpy.isnan(x) else x
    )
    return rounded


def round_to_nearest(series, base):
    """Rounds values in series to the nearest base."""
    # ndigits=0 ensures the return value is a whole number, but with the same type as x
    series_copy = series.apply(lambda x: base * round(x / base, ndigits=0))
    try:
        return series_copy.astype(int)
    except ValueError:
        # series contained nan
        return series_copy


def suppress(series, threshold):
    """Replaces values in series less than or equal to threshold with missing values."""
    series_copy = series.copy()
    series_copy[series_copy <= threshold] = numpy.nan  # in place
    return series_copy


def count_values(series, *, base, threshold):
    """Counts values, including missing values, in series.

    Rounds counts to the nearest base; then suppresses counts less than or equal to
    threshold.
    """
    count = series.value_counts(dropna=False)
    count = count.pipe(round_to_nearest, base).pipe(suppress, threshold)
    count = count.sort_index(na_position="first")
    return count


# NOTE: groupby(dropna=False) not supported on OS version of pandas
def _groupby(series):
    s = series.groupby(series).size()
    count_na = len(series[series.isna()])
    if count_na > 0:
        d = s.to_dict()
        d[numpy.nan] = count_na
        s = pandas.Series(d)
    return s


def get_column_summaries(dataframe):
    for name, series in dataframe.items():
        if name == "patient_id":
            continue

        is_csv_bool = dataframe.attrs["from_csv"] and is_bool_as_int(series)
        is_bool = types.is_bool_dtype(series)
        if is_csv_bool or is_bool:
            count = count_values(series, threshold=5, base=5)
            percentage = count / count.sum() * 100
            summary = pandas.DataFrame(
                {"Count": count, "Percentage": percentage}
            )
            summary.index.name = "Column Value"
            yield name, summary

        is_date = types.is_datetime64_ns_dtype(series)
        is_csv_date = dataframe.attrs["from_csv"] and "date" in name
        if is_date or is_csv_date:
            date_series = parse_os_year(series)
            redacted = redact_round_series(_groupby(date_series))
            summary = redacted.to_frame(name="Count")
            yield name, summary


# NOTE: not a general function, just for curating using yob
# TODO: switch this to using age
def count_impossible_dates(dataframe):
    dataframe = dataframe.set_index("patient_id")
    earliest = dataframe.filter(regex="earliest")
    try:
        years_since_birth = earliest.sub(dataframe["yob"], axis=0)
    except KeyError:
        return

    impossible_early = years_since_birth[years_since_birth < 0].count()

    latest = dataframe.filter(regex="latest")
    impossible_date = latest[latest > 2022].count()
    impossible = redact_round_series(
        pandas.concat([impossible_early, impossible_date])
    )
    impossible = impossible.reset_index().rename(
        {"index": "Variable", 0: "Count"}, axis=1
    )
    return impossible


def get_dataset_report(
    input_file, table_summary, column_summaries, impossible_summary
):
    return TEMPLATE.render(
        input_file=input_file,
        table_summary=table_summary,
        column_summaries=column_summaries,
        impossible_summary=impossible_summary,
    )


def write_dataset_report(output_file, dataset_report):
    with output_file.open("w", encoding="utf-8") as f:
        f.write(dataset_report)


def main():
    args = parse_args()
    input_files = args.input_files
    output_dir = args.output_dir

    for input_file in input_files:
        input_dataframe = read_dataframe(input_file)
        table_summary = get_table_summary(input_dataframe)
        column_summaries = get_column_summaries(input_dataframe)
        impossible_summary = count_impossible_dates(input_dataframe)

        output_file = output_dir / f"{get_name(input_file)}.html"
        dataset_report = get_dataset_report(
            input_file, table_summary, column_summaries, impossible_summary
        )
        write_dataset_report(output_file, dataset_report)


# Argument parsing
# ----------------


def get_path(*args):
    return pathlib.Path(*args)


def match_paths(pattern):
    yield from (get_path(x) for x in glob.iglob(pattern))


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


if __name__ == "__main__":
    main()
