import argparse
import pathlib
import fnmatch
import pandas
import numpy

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dateutil import parser
from collections import Counter


def get_measure_tables(input_file):
    # The `date` column is assigned by the measures framework.
    measure_table = pandas.read_csv(input_file, parse_dates=["date"])

    return measure_table


def subset_table(measure_table, measures_pattern, measures_list):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """
    if measures_pattern:
        measures_list = match_paths(measure_table["name"], measures_pattern)
        if len(measures_list) == 0:
            raise ValueError("Pattern did not match any files")

    if not measures_list:
        return measure_table
    return measure_table[measure_table["name"].isin(measures_list)]


def coerce_numeric(table):
    """
    The denominator and value columns should contain only numeric values
    Other values, such as the REDACTED string, or values introduced by error,
    should not be plotted
    Use a copy to avoid SettingWithCopyWarning
    Leave NaN values in df so missing data are not inferred
    """
    coerced = table.copy()
    coerced["denominator"] = pandas.to_numeric(
        coerced["denominator"], errors="coerce"
    )
    coerced["value"] = pandas.to_numeric(coerced["value"], errors="coerce")
    return coerced


def scale_thousand(ax):
    """
    Scale a proportion for rate by 1000
    Used for y axis display
    """

    def thousand_formatter(x, pos):
        return f"{x*1000: .0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(thousand_formatter))
    ax.set_ylabel("Rate per thousand")


def scale_hundred(ax):
    """
    Scale a proportion for percentage
    Used for y axis display
    """

    def hundred_formatter(x, pos):
        return f"{x*100: .0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(hundred_formatter))
    ax.set_ylabel("Percentage")


def autoselect_labels(measures_list):
    measures_set = set(measures_list)
    counts = Counter(
        numpy.concatenate([item.split("_") for item in measures_set])
    )
    remove = [k for k, v in counts.items() if v == len(measures_set)]
    return remove


def translate_group(group_var, label, repeated, autolabel=False):
    """
    Translate a measure name into a plot label
    Autolabel uses the 'name' column, but removes words that appear in every
    item in the group
    The alternative labeling uses the 'group' column
    """
    if autolabel:
        return " ".join(
            [x for x in label.split("_") if x not in repeated]
        ).title()
    else:
        return filename_to_title(group_var)


def filename_to_title(filename):
    return filename.replace("_", " ").title()


def plot_cis(ax, data):
    data["ci95hi"] = data["value"] + 1.96 * (
        numpy.sqrt(data["value"] * (1 - data["value"]) / data["denominator"])
    )
    data["ci95lo"] = data["value"] - 1.96 * (
        numpy.sqrt(data["value"] * (1 - data["value"]) / data["denominator"])
    )
    ax.fill_between(
        data.index,
        (data["ci95lo"]),
        (data["ci95hi"]),
        alpha=0.1,
    )


def get_group_chart(
    measure_table,
    columns=2,
    date_lines=None,
    scale=None,
    ci=False,
    exclude_group=None,
):
    figure = plt.figure(
        figsize=(columns * 6, columns * 5), constrained_layout=True
    )
    measure_table.set_index("date", inplace=True)

    repeated = autoselect_labels(measure_table["name"])

    groups = measure_table.groupby("name")
    total_plots = len(groups)

    if total_plots > 10:
        raise Exception(f"Trying to plot more than 10 plots ({total_plots})")

    # Compute Rows required
    rows = total_plots // columns
    if total_plots % columns > 0:
        rows = rows + 1

    lgds = []
    for index, panel_group in enumerate(groups):
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.autoscale(enable=True, axis="y")
        title = translate_group(
            panel_group[1].category[0],
            panel_group[0],
            repeated,
            autolabel=True,
        )
        ax.set_title(title)
        filtered = panel_group[1][panel_group[1].group != exclude_group]
        numeric = coerce_numeric(filtered)
        for plot_group, plot_group_data in numeric.groupby("group"):
            ax.plot(
                plot_group_data.index, plot_group_data.value, label=plot_group
            )
            if ci:
                plot_cis(ax, plot_group_data)
            lgd = ax.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                fontsize="x-small",
            )
            lgds.append(lgd)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7)
        if date_lines:
            min_date = min(measure_table.index)
            max_date = max(measure_table.index)
            add_date_lines(plt, date_lines, min_date, max_date)
        if scale == "percentage":
            scale_hundred(ax)
        elif scale == "rate":
            scale_thousand(ax)
    return (plt, lgds)


def write_group_chart(group_chart, path, plot_title, lgds):
    suptitle = plt.suptitle(plot_title)
    group_chart.savefig(
        path, bbox_extra_artists=tuple(lgds) + (suptitle,), bbox_inches="tight"
    )


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


def add_date_lines(plt, vlines, min_date, max_date):
    for date in vlines:
        date_obj = pandas.to_datetime(date)
        if date_obj >= min_date and date_obj <= max_date:
            try:
                plt.axvline(x=date_obj, color="orange", ls="--")
            except parser._parser.ParserError:
                continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
    )
    measures_group = parser.add_mutually_exclusive_group(required=False)
    measures_group.add_argument(
        "--measures-pattern",
        required=False,
        help="Glob pattern for matching one or more measures names",
    )
    measures_group.add_argument(
        "--measures-list",
        required=False,
        help="A list of one or more measure names",
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
        "--date-lines",
        nargs="+",
        help="Vertical date lines",
    )
    choices = ["percentage", "rate"]
    parser.add_argument("--scale", default=None, choices=choices)
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="""
             Add confidence intervals to the plot.
             NOTE: only supported for single group",
             """,
    )
    parser.add_argument(
        "--exclude-group",
        help="Exclude group with this label from plot, e.g. Unknown",
    )
    parser.add_argument(
        "--columns", type=int, default=2, help="Number of plot columns"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    output_dir = args.output_dir
    output_name = args.output_name
    date_lines = args.date_lines
    scale = args.scale
    confidence_intervals = args.confidence_intervals
    exclude_group = args.exclude_group
    columns = args.columns

    measure_table = get_measure_tables(input_file)

    plot_title = filename_to_title(output_name)

    # Parse the names field to determine which subset to use
    subset = subset_table(measure_table, measures_pattern, measures_list)
    chart, lgds = get_group_chart(
        subset,
        columns=columns,
        date_lines=date_lines,
        scale=scale,
        ci=confidence_intervals,
        exclude_group=exclude_group,
    )
    write_group_chart(chart, output_dir / output_name, plot_title, lgds)
    chart.close()


if __name__ == "__main__":
    main()
