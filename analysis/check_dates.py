from comparisons import gte, gt

import argparse
import glob
import pathlib
import pandas

def check_register_dates(df):
    before_2006 = df["depression_register"] & (pandas.to_datetime(df["depr_lat_date"]) < pandas.to_datetime("2006-04-01"))
    resolved = df["depression_register"] & df["depr_res"] & df["depr_lat"] & (gt(pandas.to_datetime(df["depr_res_date"]), pandas.to_datetime(df["depr_lat_date"])))
    under_18 = df["depression_register"] & (df["age"] < 18)
    resolved_same_day = df["depression_register"] & df["depr_lat"] & df["depr_res"] & (df["depr_lat_date"] == df["depr_res_date"])
    output = {"before_2006": before_2006.sum(), "resolved": resolved.sum(), "under_18": under_18.sum(), "resolved_same_day": resolved_same_day.sum()}
    return pandas.DataFrame(list(output.items()))

def get_extension(path):
    return "".join(path.suffixes)


def read_dataframe(path):
    ext = get_extension(path)
    if ext == ".csv" or ext == ".csv.gz":
        return pandas.read_csv(path, parse_dates=True)
    elif ext == ".feather":
        return pandas.read_feather(path)
    elif ext == ".dta" or ext == ".dta.gz":
        return pandas.read_stata(path)
    else:
        raise ValueError(f"Cannot read '{ext}' files")


def write_dataframe(dataframe, path):
    dataframe.to_csv(path)

def get_input_table(input_files):
    for input_file in input_files:
        input_table = read_dataframe(input_file)
        input_table.attrs[
            "fname"
        ] = f"dates_{input_file.name.lstrip('input_')}"
        yield input_table


def write_input_table(input_table, path):
    write_dataframe(input_table, path)


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


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


def main():
    args = parse_args()
    input_files = args.input_files
    output_dir = args.output_dir

    output_dir.mkdir(exist_ok=True)
    for input_table in get_input_table(input_files):
        fname = input_table.attrs["fname"]
        date_counts = check_register_dates(input_table)
        write_input_table(date_counts, output_dir / fname)


if __name__ == "__main__":
    main()
