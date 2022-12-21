import argparse
import pathlib
import fnmatch
import pandas
import numpy
import scipy

import matplotlib.pyplot as plt
import dataframe_image as dfi
from collections import Counter

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from patsy.contrasts import Treatment


plt.style.use("seaborn-whitegrid")


def lrtest(smaller, bigger):
    ll_smaller = smaller.llf
    ll_bigger = bigger.llf
    stat = -2 * (ll_smaller - ll_bigger)
    return scipy.stats.chi2.sf(stat, 2)


def autoselect_labels(measures_list):
    measures_set = set(measures_list)
    counts = Counter(
        numpy.concatenate([item.split("_") for item in measures_set])
    )
    remove = [k for k, v in counts.items() if v == len(measures_set)]
    return remove


def translate_group(label, repeated):
    return " ".join([x for x in label.split("_") if x not in repeated]).title()


def get_table(measure_table, group_by):
    """
    Group the table by some number of user-provided parameters and generate
    a regression model for each one
    """
    rows = {}
    for subgroup, subgroup_data in measure_table.groupby(
        measure_table.name.str.contains("new")
    ):
        repeated = autoselect_labels(subgroup_data.name.unique())
        if subgroup:
            key = f"New {repeated[0]}".title()
            lags = 0
        else:
            key = f"All {repeated[0]}".title()
            lags = 2
        for subsubgroup, subsubgroup_data in subgroup_data.groupby(group_by):
            label = translate_group(subsubgroup, repeated)
            if len(subsubgroup_data.group_0.unique()) > 1:
                # subsubgroup_data = add_control_variables(subsubgroup_data, "group")
                formula = get_formula(
                    subsubgroup_data,
                    interaction_term="group",
                    reference="Depression register",
                )
                model = get_regression(subsubgroup_data, lags, formula)
                forest_plot(model)
            else:
                fourier = get_fourier(subsubgroup_data)
                formula = get_formula(
                    subsubgroup_data, fourier_terms=fourier.columns
                )
                model = get_regression(
                    pandas.concat([subsubgroup_data, fourier], axis=1),
                    lags,
                    formula,
                )
            row = get_coef(model, subgroup)
            rows[(key, label)] = row
    index = pandas.MultiIndex.from_tuples(rows.keys())
    table = pandas.concat(rows.values(), keys=index)
    table.index = index
    return table


def forest_plot(model):
    import matplotlib.pyplot as plt

    coef = model.params
    cis = model.conf_int().rename(columns={0: "lci", 1: "uci"})
    df = pandas.concat([coef, cis], axis=1)
    df = df[df.index.str.contains(":")]

    keys = list(df.index)
    indices = []
    for key in keys:
        x, y = key.split(":")
        indices.append((x, (y.split(".")[-1]).rstrip("]")))

    renamed = df.set_index(pandas.MultiIndex.from_tuples(indices))

    import code

    code.interact(local=locals())
    diff = df.uci - df.lci
    plt.figure(figsize=(6, 4), dpi=150)
    plt.errorbar(
        x=renamed[0].values,
        y=df.index.values,
        xerr=diff,
        color="black",
        capsize=3,
        linestyle="None",
        linewidth=1,
        marker="o",
        markersize=5,
        mfc="black",
        mec="black",
    )
    plt.axvline(x=1, linewidth=0.8, linestyle="--", color="black")
    plt.tick_params(axis="both", which="major", labelsize=8)
    plt.xlabel("Odds Ratio and 95% Confidence Interval", fontsize=8)
    plt.tight_layout()
    # plt.savefig('raw_forest_plot.png')
    plt.show()


def translate_to_ci(coef, cis, name, round_to=2):
    """
    Select and format coefs and cis from the mapping
    """
    df = pandas.concat([coef, cis], axis=1)
    mapping = {
        "time": ("", "Pre-COVID-19 monthly slope (95% CI)"),
        "mar20": ("", "March 2020 (95% CI)"),
        "april20": ("", "April 2020 (95% CI)"),
        "slope": ("Lockdown period", "Change in slope (95% CI)"),
        "step": ("Lockdown period", "Level shift (95% CI)"),
        "slope2": ("Recovery period", "Change in slope (95% CI)"),
        "step2": ("Recovery period", "Level shift (95% CI)"),
    }
    df = df.loc[mapping.keys()]
    df = df.set_index(pandas.MultiIndex.from_tuples(mapping.values()))
    pcnt = (
        round(df[0], 2).astype(str)
        + "% ("
        + round(df.lci, 2).astype(str)
        + "% to "
        + round(df.uci, 2).astype(str)
        + "%)"
    )
    pcnt.name = name
    return pandas.DataFrame(pcnt).transpose()


def check_residuals(resid):
    """
    Visualize autocorrelation with the acf and pacf
    Check normality of residuals, and qq plot
    """
    sm.graphics.tsa.plot_acf(resid, lags=40)
    plt.show()

    sm.graphics.tsa.plot_pacf(resid, lags=20, method="ywm")
    plt.show()

    resid.plot(kind="kde")
    plt.show()

    sm.qqplot(resid, scipy.stats.t, fit=True, line="45")
    plt.show()


def get_fourier(df):
    fourier_gen = Fourier(12, order=2)
    fourier_vars = fourier_gen.in_sample(df.index)
    fourier_vars.columns = ["s1", "c1", "s2", "c2"]
    return fourier_vars


def get_regression(df, lags, formula):
    model = smf.glm(
        formula,
        data=df,
        family=sm.families.NegativeBinomial(),
        exposure=df.denominator,
    ).fit(maxiter=200)
    if not model.converged:
        raise ConvergenceWarning("Failed to converge")
    model_errors = smf.glm(
        formula,
        data=df,
        family=sm.families.NegativeBinomial(),
        exposure=df.denominator,
    ).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, maxiter=200)
    print(model.summary())
    print(model_errors.summary())
    #check_residuals(model.resid_pearson)
    #check_residuals(model_errors.resid_pearson)
    return model_errors


def get_formula(df, fourier_terms=None, interaction_term=None, reference=None):
    formula = (
        "numerator ~ time + step + slope + mar20 + april20 + step2 + slope2"
    )
    if interaction_term:
        df.group = df.group.astype("category")
        levels = list(df.group.unique())
        levels.remove(reference)
        levels = [reference] + levels
        df.group = df.group.cat.reorder_categories(levels)
        formula += f"+ step*{interaction_term} + slope*{interaction_term} + step2*{interaction_term} + slope2*{interaction_term}"
    if fourier_terms is not None:
        formula += "+ " + "+".join(fourier_terms)
    return formula


def get_ols(df):
    dummies = pandas.get_dummies(
        pandas.to_datetime(df.date).dt.month_name(), prefix=None
    )
    df = pandas.concat([df, dummies], axis=1)
    model = smf.ols(
        formula="rate ~ time + step + slope"
        + "+"
        + "+ ".join(list(dummies.columns)),
        data=df,
    )
    res = model.fit()
    print(res.summary())
    # check_residuals(res.resid)
    return res


def get_coef(res, name):
    coef = res.params
    cis = res.conf_int().rename(columns={0: "lci", 1: "uci"})
    pcnt_change = translate_to_ci(coef, cis, name)
    return pcnt_change


def plot_model(res, df, start, step2):
    predictions = res.get_prediction(df).summary_frame(alpha=0.05)

    # counterfactual assumes no interventions
    cf_df = df.copy()
    cf_df["slope"] = 0.0
    cf_df["step"] = 0.0
    cf_df["mar20"] = 0.0
    cf_df["slope2"] = 0.0
    cf_df["step2"] = 0.0

    # counter-factual predictions
    cf = res.get_prediction(cf_df).summary_frame(alpha=0.05)
    # TODO: add group into df so we can group the plot

    cf2_df = df.copy()
    cf2_df["slope2"] = 0.0
    cf2_df["step2"] = 0.0

    # counter-factual predictions
    cf2 = res.get_prediction(cf2_df).summary_frame(alpha=0.05)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot depression register data
    ax.scatter(
        df["time"],
        df["value"],
        facecolors="none",
        edgecolors="steelblue",
        label="depression register data",
        linewidths=2,
    )

    # Plot model mean rate prediction
    ax.plot(
        df["time"][:start],
        predictions["mean"][:start],
        "b-",
        label="model prediction",
    )
    ax.plot(df["time"][start:], predictions["mean"][start:], "b-")

    # Plot counterfactual mean rate with 95% cis
    ax.plot(
        df["time"][start:], cf["mean"][start:], "k.", label="counterfactual"
    )
    ax.fill_between(
        df["time"][start:],
        cf["mean_ci_lower"][start:],
        cf["mean_ci_upper"][start:],
        color="k",
        alpha=0.1,
        label="counterfactual 95% CI",
    )
    ax.plot(
        df["time"][step2:],
        cf2["mean"][step2:],
        "k.",
        alpha=0.5,
        label="counterfactual recovery",
    )
    ax.fill_between(
        df["time"][start:],
        cf2["mean_ci_lower"][start:],
        cf2["mean_ci_upper"][start:],
        color="k",
        alpha=0.1,
        label="counterfactual recovery 95% CI",
    )

    # Plot line marking intervention
    ax.axvline(x=start + 0.5, color="r", label="March 2020")
    ax.axvline(x=step2 + 0.5, color="c", label="recovery")
    ax.legend(loc="best")
    plt.xlabel("Months")
    plt.ylabel("Depression register rate per 1,000")
    plt.show()


def get_its_variables(dataframe, cutdate1, cutdate2):
    df = dataframe.copy()
    min_year = min(pandas.to_datetime(df.date).dt.year)
    df["rate"] = 1000 * df["value"].astype(float)
    df["time"] = 12 * (pandas.to_datetime(df.date).dt.year - min_year) + (
        pandas.to_datetime(df.date).dt.month
    )
    cutmonth1 = df[df["date"] == cutdate1].iloc[0].time - 1
    cutmonth2 = df[df["date"] == cutdate2].iloc[0].time - 1
    #NOTE: dropping Nov 2022 (not full month of data)
    end = df.iloc[-1].time
    df["step"] = df.apply(lambda x: 1 if x.time > cutmonth1 else 0, axis=1)
    df["slope"] = df.apply(lambda x: max(x.time - cutmonth1, 0), axis=1)
    df["step2"] = df.apply(lambda x: 1 if x.time > cutmonth2 else 0, axis=1)
    df["slope2"] = df.apply(lambda x: max(x.time - cutmonth2, 0), axis=1)
    df["mar20"] = df.apply(
        lambda x: 1 if x.time == cutmonth1 + 1 else 0, axis=1
    )
    df["april20"] = df.apply(
        lambda x: 1 if x.time == cutmonth1 + 2 else 0, axis=1
    )
    df["index"] = df["time"]
    df = df.set_index("index")
    return (df, cutmonth1, cutmonth2, end)


def add_control_variables(df, groupby):
    df[groupby] = df[groupby].astype(int)
    df["time_z"] = df["time"] * df[groupby]
    df["step_z"] = df["step"] * df[groupby]
    df["slope_z"] = df["slope"] * df[groupby]
    df["step2_z"] = df["step2"] * df[groupby]
    df["slope2_z"] = df["slope2"] * df[groupby]
    return df


def get_measure_tables(input_file):
    # The `date` column is assigned by the measures framework.
    measure_table = pandas.read_csv(input_file, parse_dates=["date"])

    return measure_table


def coerce_numeric(table):
    """
    The denominator and value columns should contain only numeric values
    Other values, such as the REDACTED string, or values introduced by error,
    should not be plotted
    Use a copy to avoid SettingWithCopyWarning
    Leave NaN values in df so missing data are not inferred
    """
    coerced = table.copy()
    coerced["numerator"] = pandas.to_numeric(
        coerced["numerator"], errors="coerce"
    )
    coerced["denominator"] = pandas.to_numeric(
        coerced["denominator"], errors="coerce"
    )
    coerced["value"] = pandas.to_numeric(coerced["value"], errors="coerce")
    return coerced


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


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


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
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    measures_pattern = args.measures_pattern
    measures_list = args.measures_list
    output_dir = args.output_dir

    measure_table = get_measure_tables(input_file)

    # Parse the names field to determine which subset to use
    subset = subset_table(measure_table, measures_pattern, measures_list)
    numeric = coerce_numeric(subset)
    # NOTE: remove the incomplete November month
    numeric = numeric[numeric["date"] != "2022-11-01"]
    df, step_time, step2_time, end = get_its_variables(
        numeric, "2020-03-01", "2021-04-01"
    )
    dummies = pandas.get_dummies(
        pandas.to_datetime(df.date).dt.month_name(), prefix=None
    )
    df = pandas.concat([df, dummies], axis=1)
    # df = df[df["group"] == "1"]
    # df = add_control_variables(df, "group")
    # df = df.reset_index()
    #formula = get_formula(df)
    #model = get_regression(df, 2, formula)
    # plot_model(model, df, step_time, step2_time)
    table = get_table(df, "category_0")
    dfi.export(table, "its_table.png")


if __name__ == "__main__":
    main()
