import argparse
import glob
import itertools
import pandas
import pathlib


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


def get_codelist_table(input_files):
    all_files = set(itertools.chain(*input_files))
    for input_file in all_files:
        table = pandas.read_csv(input_file, dtype="str")
        yield table


def build_url(codelists):
    def make_url(codes):
        return f"https://openprescribing.net/analyse/#org=regional_team&numIds={','.join(codes)}&denom=total_list_size&selectedTab=chart"

    codelist_tables = get_codelist_table(codelists)
    df = pandas.concat(codelist_tables)
    return make_url(df.bnf_code.unique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Build Open Prescribing Url",
        description="Take a list of codelists and build a query string for openprescribing",
    )
    parser.add_argument(
        "--codelist-path",
        action="append",
        type=match_paths,
        help="Glob pattern(s) for matching one or more input files",
    )
    args = parser.parse_args()
    codelists = args.codelist_path
    print(build_url(codelists))
