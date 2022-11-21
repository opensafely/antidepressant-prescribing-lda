import argparse
import functools
import glob
import pathlib
import re

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
    # It's useful to know whether a dataframe was read from a csv when
    # summarizing the columns later.
    dataframe.attrs["from_csv"] = from_csv
    # We give the column index a name now, because it's preserved when
    # summaries are computed later.
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


def is_date_as_obj(series):
    try:
        pandas.to_datetime(series)
        return True
    except dateutil.parser._parser.ParserError:
        return False


def is_category(series):
    """Series with 10 or fewer levels"""
    return len(series.unique()) <= 10


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
    redact_extra = series_in[(series_in >= 0) & (series_in <= 7)].sum()
    # Redact <= 7
    series_out = series_in.apply(
        lambda x: "[REDACTED]" if x > 0 and x <= 7 else x
    )
    if redact_extra > 0 and redact_extra <= 7:
        next_min = series_out[series_out != "[REDACTED]"].min()
        series_out[series_out == next_min] = "[REDACTED]"
    rounded = series_out.apply(
        lambda x: 5 * round(x / 5) if (x != "[REDACTED]") else x
    )
    return rounded


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

        is_date = types.is_datetime64_ns_dtype(series)
        is_csv_date = dataframe.attrs["from_csv"] and "date" in name
        if is_date or is_csv_date:
            if is_csv_date:
                date_series = parse_os_year(series)
            else:
                date_series = series.dt.year
            redacted = redact_round_series(_groupby(date_series))
            summary = redacted.to_frame(name="Count")
            yield name, summary
        is_cat = is_category(series)
        if is_cat:
            count = series.value_counts(dropna=False)
            redacted = redact_round_series(count)
            total = redacted[redacted != "[REDACTED]"].sum()
            percentage = redacted.apply(
                lambda x: 100 * x / total
                if x != "[REDACTED]"
                else "[REDACTED]"
            )
            summary = pandas.DataFrame(
                {"Count": redacted, "Percentage": percentage}
            )
            summary.index.name = "Column Value"
            yield name, summary


# NOTE: dependent on having an age variable
def count_impossible_dates(dataframe, current_year):
    dataframe = dataframe.set_index("patient_id")
    try:
        age = dataframe["age"]
    except KeyError:
        return
    dates = dataframe.filter(regex="date")

    if dataframe.attrs["from_csv"]:
        dates = dates.apply(lambda x: parse_os_year(x))
    else:
        dates = dates.apply(lambda x: x.dt.year, axis=1)

    age_added = dates.add(age, axis=0)

    impossible_early = pandas.DataFrame(
        redact_round_series(age_added[age_added < current_year].count())
    )
    impossible_date = pandas.DataFrame(
        redact_round_series(dates[dates > current_year].count())
    )
    impossible_early = impossible_early.rename({0: "Count"}, axis=1)
    impossible_date = impossible_date.rename({0: "Count"}, axis=1)
    return (impossible_early, impossible_date)


def get_dataset_report(
    input_file,
    table_summary,
    column_summaries,
    impossible_early,
    impossible_date,
):
    return TEMPLATE.render(
        input_file=input_file,
        table_summary=table_summary,
        column_summaries=column_summaries,
        impossible_early=impossible_early,
        impossible_date=impossible_date,
    )


def write_dataset_report(output_file, dataset_report):
    with output_file.open("w", encoding="utf-8") as f:
        f.write(dataset_report)


def main():
    args = parse_args()
    input_files = args.input_files
    output_dir = args.output_dir

    for input_file in input_files:
        match_str = re.search(r"\d{4}-\d{2}-\d{2}", input_file.name)
        if match_str:
            year = datetime.datetime.strptime(
                match_str.group(), "%Y-%m-%d"
            ).year
        else:
            year = datetime.datetime.today().year
        input_dataframe = read_dataframe(input_file)
        table_summary = get_table_summary(input_dataframe)
        column_summaries = get_column_summaries(input_dataframe)
        impossible_early, impossible_date = count_impossible_dates(
            input_dataframe, year
        )

        output_file = output_dir / f"{get_name(input_file)}.html"
        dataset_report = get_dataset_report(
            input_file,
            table_summary,
            column_summaries,
            impossible_early,
            impossible_date,
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
