import pathlib
import argparse
import pandas as pd

from config import start_date, demographics, lda_subgroups

"""
Generate Table1 by demographics using measures files
Allows redaction to be handled by the measures framework
Assumes the measure has been generated over the population
And that the measure file naming follows
measure_{attribute}_{subgroup}_rate.csv
"""


def get_month(input_dir, measure_attribute, d):
    """
    Use measures small number suppression for consistency with measures results
    Get the first month
    """
    path = input_dir / "measure_{}_all_breakdown_{}_rate.csv".format(
        measure_attribute, d
    )

    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Measure file for {path} did not exist")

    try:
        data = data[data["date"] == start_date]
    except KeyError:
        raise KeyError("Measure file did not have date column")

    return data[[d, measure_attribute, "population"]].rename(
        columns={
            d: "Category",
            f"{measure_attribute}": "{}".format(
                (" ".join(measure_attribute.split("_")).title())
            ),
            "population": "Total",
        }
    )


def transform_percentage(x):
    return (
        x.astype(str)
        + " ("
        + (((x / x.sum()) * 100).round(0)).astype(str)
        + ")"
    )


def get_percentages(df):
    percent = df.groupby(level=0).transform(transform_percentage)
    return df.join(percent, rsuffix=" (%)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the input directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--measure-attribute",
        required=True,
        help="Name of the measure attribute",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    measure_attribute = args.measure_attribute

    tables = []
    demographic_subset = list(set(demographics) - set(lda_subgroups.keys()))
    for d in demographic_subset:
        data = get_month(input_dir, measure_attribute, d)
        tables.append(data)
    table1 = pd.concat(tables, keys=demographic_subset, names=["Attribute"])
    # TODO: print warning if table is empty?
    if not table1.empty:
        table1 = get_percentages(table1)
    table1.to_csv(output_dir / "table1.csv", index=True)


if __name__ == "__main__":
    main()
