import argparse
import glob
import pathlib
import pandas


def count_any_over_time(df):
    df = df[df["antidepressant_any"] == 1]
    df = df.sort_values(["patient_id", "date"])
    df["order"] = (
        df.groupby(["patient_id"]).cumcount().apply(lambda x: f"date_{x}")
    )
    df = df.pivot(index="patient_id", columns="order", values="date")
    prescriptions_per_person = df.count(axis=1).mean()
    output = {
        "prescriptions_per_person": round(prescriptions_per_person, 1),
        **df.count(),
    }

    for previous, current in zip(df.columns, df.columns[1:]):
        output[f"{current}_{previous}"] = round(
            (df[current] - df[previous]).dt.days.mean()
        )

    return pandas.DataFrame(list(output.items()))


def count_multiple_prescriptions(df):
    multiple_prescriptions = df[
        (
            df["antidepressant_ssri"]
            + df["antidepressant_tricyclic"]
            + df["antidepressant_other"]
        )
        > 1
    ]
    count = multiple_prescriptions.groupby("date").count().antidepressant_any
    count.name = "count"
    total = (
        (df[df["antidepressant_any"] == 1])
        .groupby("date")
        .count()
        .antidepressant_any
    )
    total.name = "total"
    result = pandas.concat([count, total], axis=1)
    result["percent"] = 100 * result["count"] / result["total"]
    return pandas.DataFrame(result["percent"].to_dict().items())


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
        ] = f"test_{input_file.name.rstrip(''.join(input_file.suffixes)).lstrip('input_')}.csv"
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
    tables = []
    for input_table in get_input_table(input_files):
        fname = input_table.attrs["fname"]
        d = fname.split("_")[-1].split(".")[0]
        analysis_table = input_table[
            [
                "patient_id",
                "antidepressant_any",
                "antidepressant_ssri",
                "antidepressant_tricyclic",
                "antidepressant_other",
            ]
        ]
        analysis_table["date"] = pandas.to_datetime(d, format="%Y-%m-%d")
        tables.append(analysis_table)
    df = pandas.concat(tables)
    any_results = count_any_over_time(
        df[["antidepressant_any", "patient_id", "date"]]
    )
    multiple = count_multiple_prescriptions(df)
    all_results = pandas.concat([any_results, multiple])
    write_input_table(all_results, output_dir / "test_study_period.csv")


if __name__ == "__main__":
    main()
