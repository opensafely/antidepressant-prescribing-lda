import pathlib
import argparse
import glob
import pandas as pd

from config import start_date, end_date, marker, demographics


def get_months(input_dir):
    # Get the first month
    # Could match start_date, get multiple months...
    try:
        first_month = sorted(input_dir.glob("input_*.csv"))[0]
        return pd.read_csv(first_month)
    except IndexError:
        return None


def filter_demographics(data):
    filtered = data.filter(items=demographics)
    print(filtered.columns)
    return filtered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=pathlib.Path,
        help="Path to the input directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    data = get_months(input_dir)
    tables = []
    for d in demographics:
        counts = data[d].value_counts()
        pct = data[d].value_counts(normalize=True).mul(100).round(0)
        section = counts.astype(str) + pct.apply(lambda x: f" ({x})")
        print(section)
        df = pd.DataFrame({"Category": section.index, "Total (%)": section.values})
        tables.append(df)

    table1 = pd.concat(tables, keys=demographics, names=["Attribute"])
    table1.to_csv("table1.csv", index=True)


if __name__ == "__main__":
    main()
