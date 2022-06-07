from cohortextractor import pandas_utils

import argparse
import glob
import pathlib
import pandas


def count_mismatches(df):
    counts = {}
    for col in df.columns[df.columns.str.endswith("_python")]:
        name = col.rstrip("_python")
        counts[name] = (df[col] != df[name]).sum()
    return pandas.DataFrame(list(counts.items()))


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
    # We refactored this function, replacing copy-and-paste code from cohort-extractor
    # with a call to cohort-extractor itself. However, the error types differed. We
    # preserved the pre-refactoring error type.
    ext = get_extension(path)
    if ext == ".feather":
        # We write feather files ourselves, because of an issue with dependencies.
        # For more information, see:
        # https://github.com/opensafely-actions/cohort-joiner/issues/25
        dataframe.to_feather(path)
    else:
        try:
            pandas_utils.dataframe_to_file(dataframe, path)
        except RuntimeError:
            raise ValueError(f"Cannot write '{ext}' files")


def get_input_table(input_files):
    for input_file in input_files:
        input_table = read_dataframe(input_file)
        input_table.attrs[
            "fname"
        ] = f"mismatch_{input_file.name.lstrip('input_')}"
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
        mismatch_counts = count_mismatches(input_table)
        write_input_table(mismatch_counts, output_dir / fname)


if __name__ == "__main__":
    main()
