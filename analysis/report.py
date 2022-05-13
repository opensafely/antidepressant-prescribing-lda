import pathlib
import argparse

from config import start_date, end_date, marker, demographics

import pkg_resources
import jinja2
from jinja2 import Template, FileSystemLoader
from collections import defaultdict

# How do we know what indicators we want to display?
# We could assume from what output files are present
# We could set in a config file
# We could parse the word docs
# We could tell from the yaml file

# Assume the indicators will have their number in the name and will therefore be ordered
# TODO: do we assume the indicators are in the same directory or separate subdirectories?
def get_indicator_names(input_dir):
    return ["register", "dep003"]


def parse_indicators(input_dir):
    """
    Create a dictionary for each indicator, which will include
    1. Title
    2. Business rules (numerator and denominator)
    3. Total population
    4. Decile charts
    5. By demographic
    """
    indicators = {}
    for i in get_indicator_names(input_dir):
        indicator_dir = {
            "description": "Description of {}".format(i),
            "business_rules": "Placeholder for the denominator and numerator rules",
        }
        # Key is the chart title
        charts = {}
        charts[
            "Patients meeting indicator {}".format(i)
        ] = "group_chart_{}_total_rate.png".format(i)
        charts[
            "{} by GP practice".format(i)
        ] = "deciles_chart_{}_practice_rate.png".format(i)
        for d in demographics:
            charts[
                "Breakdown of {} by {}".format(i, d)
            ] = "group_chart_{}_{}_rate.png".format(i, d)

        indicator_dir["charts"] = charts
        indicators[i] = indicator_dir
    return indicators


def make_report(resource_dir, input_dir, output_dir):

    # TODO: autoescaping
    loader = jinja2.FileSystemLoader(resource_dir)
    env = jinja2.Environment(loader=loader)
    template = env.get_template("report_template.html")

    reports = defaultdict(dict)
    # Make dictionary that will be passed to the template
    reports["metadata"] = {
        "start_date": start_date,
        "end_date": end_date,
        "marker": marker,
    }
    reports["indicators"] = parse_indicators(input_dir)

    html = template.render(reports=reports)

    with open(f"{output_dir}/report.html", "w", encoding="utf-8") as f:
        f.write(html)
        print(f"Created cohort report at {output_dir}/report.html")


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
        "--resource-dir",
        required=True,
        type=pathlib.Path,
        help="Path to the resource directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    resource_dir = args.resource_dir

    make_report(resource_dir, input_dir, output_dir)


if __name__ == "__main__":
    main()
