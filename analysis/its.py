import argparse
import pathlib
import fnmatch
import pandas

import matplotlib.pyplot as plt
import dataframe_image as dfi

import statsmodels.api as sm
import statsmodels.formula.api as smf

# from statsmodels.tsa.arima.model import ARIMA
import scipy as sp

plt.style.use("seaborn-whitegrid")


def get_table(measure_table, groupby=["name"]):
    """
    Group the table by some number of user-provided parameters and generate
    a regression model for each one
    """
    rows = {}
    for subgroup, subgroup_data in measure_table.groupby(groupby):
        df, step_time, step2_time, end = get_its_variables(
            subgroup_data, "2020-03-01", "2021-04-01"
        )
        poisson = get_poisson(df)
        row = get_coef(poisson, subgroup)
        if type(subgroup) == tuple:
            rows[subgroup] = row
        else:
            rows[(subgroup,)] = row
    index = pandas.MultiIndex.from_tuples(rows.keys())
    table = pandas.concat(rows.values(), keys=index)
    table.index = index
    return table


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
        round(100 * df[0], 2).astype(str)
        + "% ("
        + round(100 * df.lci, 2).astype(str)
        + "% to "
        + round(100 * df.uci, 2).astype(str)
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

    sm.qqplot(resid, sp.stats.t, fit=True, line="45")
    plt.show()


def get_poisson(df):
    model = smf.glm(
        "numerator ~ time + step + slope + step2 + slope2 + mar20 + april20",
        data=df,
        family=sm.families.Poisson(),
        exposure=df.denominator,
    )
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 4}, maxiter=200)
    print(res.summary())
    # check_residuals(res.resid_pearson)
    import code

    code.interact(local=locals())
    return res


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


"""
def get_its_arima(df, start, step2, end):
    dummies = pandas.get_dummies(
        pandas.to_datetime(df.date).dt.month_name(), prefix=None
    )
    df = pandas.concat([df, dummies], axis=1)
    df = df.reset_index()
    arima_results = ARIMA(
        df["rate"],
        df[["time", "step", "slope", "step2", "slope2", "mar20"]],
        order=(1, 0, 0),
    ).fit()
    print(arima_results.summary())
    # check_residuals(arima_results.resid)
    predictions = arima_results.get_prediction(0, end - 1)

    arima_cf = ARIMA(
        df["rate"][:start], df["time"][:start], order=(1, 0, 0)
    ).fit()
    # Model predictions means
    y_pred = predictions.predicted_mean
    # Counterfactual mean and 95% confidence interval
    y_cf = arima_cf.get_forecast(
        end - 1, exog=df["time"][start:]
    ).summary_frame(alpha=0.05)
"""


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
    df.numerator = df.numerator.astype(int)
    df.denominator = df.denominator.astype(int)
    return (df, cutmonth1, cutmonth2, end)


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
    df, step_time, step2_time, end = get_its_variables(
        numeric, "2020-03-01", "2021-04-01"
    )
    table = get_table(df, groupby=["name", df.name.str.contains("new")])
    dfi.export(table, output_dir + "its_table.png")


if __name__ == "__main__":
    main()
