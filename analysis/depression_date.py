import argparse
import pathlib
import re
import glob

import pandas

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

def get_latest_date(input_table):
    # NOTE: real data should not have nulls follow non-nulls
    # And a date should only be available if a code is available
    latest_date = input_table[f"depression_15mo_code_1_date"]
    latest_date.name = "depression_15mo_date"
    for i in range(1, 9):
        one = input_table[f"depression_15mo_code_{i}"]
        two = input_table[f"depression_15mo_code_{i+1}"]
        two_date = input_table[f"depression_15mo_code_{i+1}_date"]
        to_update = two_date[two.notnull() & one.ne(two)]
        latest_date.update(to_update)
    return latest_date

def filter_nones(input_table):
    return input_table.dropna(inplace=True)

def get_input_table(input_files):
    for input_file in input_files:
        input_table = read_dataframe(input_file)
        input_table.set_index("patient_id", inplace=True)
        input_table.attrs[
            "fname"
        ] = f"{input_file.name.rstrip(''.join(input_file.suffixes)).lstrip('input_')}.csv"
        yield input_table

def write_table(measure_table, path, filename):
    create_dir(path)
    measure_table.to_csv(path / filename, index=True, header=True)


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
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.input_files
    input_list = args.input_list
    output_dir = args.output_dir

    if not input_files and not input_list:
        raise FileNotFoundError("No files matched the input pattern provided")

    for input_table in get_input_table(input_list or input_files):
        output = get_latest_date(input_table)
        filter_nones(output)
        fname = input_table.attrs["fname"]
        write_table(output, output_dir, fname)


if __name__ == "__main__":
    main()
