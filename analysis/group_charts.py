import argparse
import pathlib
import re
import glob

import pandas

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dateutil import parser

MEASURE_FNAME_REGEX = re.compile(r"measure_(?P<id>\w+)\.csv")


def _get_denominator(measure_table):
    return measure_table.columns[-3]


# WARNING: cannot always recreate groupby, so not for general case
# https://github.com/opensafely-actions/deciles-charts/issues/17
def _get_group_by(measure_table):
    return list(measure_table.columns[:-4])


def get_measure_tables(input_files):
    for input_file in input_files:
        measure_fname_match = re.match(MEASURE_FNAME_REGEX, input_file.name)
        if measure_fname_match is not None:
            # The `date` column is assigned by the measures framework.
            measure_table = pandas.read_csv(input_file, parse_dates=["date"])

            # We can reconstruct the parameters passed to `Measure` without
            # the study definition.
            measure_table.attrs["id"] = measure_fname_match.group("id")
            measure_table.attrs["denominator"] = _get_denominator(
                measure_table
            )
            measure_table.attrs["group_by"] = _get_group_by(measure_table)

            yield measure_table


def drop_zero_denominator_rows(measure_table):
    mask = measure_table[measure_table.attrs["denominator"]] > 0
    return measure_table[mask].reset_index(drop=True)


def scale_thousand(ax):
    def thousand_formatter(x, pos):
        return f"{x*1000: .0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(thousand_formatter))
    ax.set_ylabel("Rate per thousand")


def scale_hundred(ax):
    def hundred_formatter(x, pos):
        return f"{x*100: .0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(hundred_formatter))
    ax.set_ylabel("Percentage")


def get_group_chart(measure_table, date_lines=None, scale=None):
    # TODO: do not hard code date and value
    plt.figure()
    fig, ax = plt.subplots()
    measure_table.set_index("date", inplace=True)
    if (
        len(measure_table.attrs["group_by"]) == 0
        or "total" in measure_table.attrs["id"]
    ):
        measure_table.value.plot(legend=None)
    else:
        measure_table.groupby(measure_table.attrs["group_by"]).value.plot(
            legend=True
        )
        plt.legend(
            bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="small"
        )
    if date_lines:
        add_date_lines(plt, date_lines)
    if scale == "percentage":
        scale_hundred(ax)
    elif scale == "rate":
        scale_thousand(ax)
    plt.tight_layout()
    return plt


def write_group_chart(group_chart, path):
    group_chart.savefig(path)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def add_date_lines(plt, vlines):
    # TODO: Check that it is within the range?
    for date in vlines:
        try:
            plt.axvline(x=pandas.to_datetime(date), color="orange", ls="--")
        except parser._parser.ParserError:
            # TODO: add logger and print warning on exception
            # Skip any dates not in the correct format
            continue


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
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--date-lines",
        nargs="+",
        help="Vertical date lines",
    )
    choices = ["percentage", "rate"]
    parser.add_argument("--scale", default=None, choices=choices)
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.input_files
    output_dir = args.output_dir
    date_lines = args.date_lines
    scale = args.scale

    for measure_table in get_measure_tables(input_files):
        measure_table = drop_zero_denominator_rows(measure_table)
        chart = get_group_chart(
            measure_table, date_lines=date_lines, scale=scale
        )
        id_ = measure_table.attrs["id"]
        fname = f"group_chart_{id_}.png"
        write_group_chart(chart, output_dir / fname)
        chart.close()


if __name__ == "__main__":
    main()
