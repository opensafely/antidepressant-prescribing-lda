import argparse
import pathlib
import fnmatch
import pandas
import numpy
import scipy

import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tools.sm_exceptions import ConvergenceWarning


STEP_TIME_1 = pandas.to_datetime("2020-03-01")
STEP_TIME_2 = pandas.to_datetime("2021-04-01")

##################
# Model building
##################


def get_fourier(df):
    """
    Create four harmonics to adjust for seasonality

    """
    fourier_gen = Fourier(12, order=2)
    fourier_vars = fourier_gen.in_sample(df.index)
    fourier_vars.columns = ["s1", "c1", "s2", "c2"]
    return fourier_vars[["s1", "c1"]]


def get_formula(
    df, fourier_terms=None, interaction_group=None, reference=None
):
    """
    Build formula depending upon whether it has an interaction for panel
    data

    """
    formula = (
        "numerator ~ time + step + slope + mar20 + april20 + step2 + slope2"
    )
    if interaction_group is not None:
        # TODO: raise an error if more than 1 category value
        df[interaction_group] = df[interaction_group].astype("category")
        levels = list(df[interaction_group].unique())
        levels.remove(reference)
        levels = [reference] + levels
        df[interaction_group] = df[interaction_group].cat.reorder_categories(
            levels
        )
        formula += f"+ step*{interaction_group} + slope*{interaction_group} + step2*{interaction_group} + slope2*{interaction_group}"
    if fourier_terms is not None:
        formula += "+ " + "+".join(fourier_terms)
    return formula


def get_regression(df, lags, formula, interaction_group):
    """
    Fit the model
    Adjust the standard errors with NeweyWest (single sequence) or for panel
    data

    """
    if interaction_group is None:
        model_errors = smf.poisson(
            formula,
            data=df,
            exposure=df.denominator,
        ).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, maxiter=200)
    else:
        model_errors = smf.poisson(
            formula,
            data=df,
            exposure=df.denominator,
        ).fit(
            cov_type="hac-groupsum",
            cov_kwds={"time": df.index, "maxlags": lags},
            maxiter=200,
        )
    print(model_errors.summary())
    if not model_errors.mle_retvals["converged"]:
        raise ConvergenceWarning("Failed to converge")
    # check_residuals(model_errors.resid_pearson)
    return model_errors


def is_bool_as_int(series):
    """Does series have bool values but an int dtype?"""
    if not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_numeric_dtype(series):
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    elif not pandas.api.types.is_bool_dtype(
        series
    ) and pandas.api.types.is_object_dtype(series):
        try:
            series = series.astype(int)
        except ValueError:
            return False
        series = series.dropna()
        return ((series == 0) | (series == 1)).all()
    else:
        return False


def series_to_bool(series):
    if is_bool_as_int(series):
        return series.astype(int).astype(bool)
    else:
        return series


def flatten(df):
    """
    Filter for rows where that value is true

    """
    df = df.dropna(axis=1, how="all")
    df.group_0 = series_to_bool(df.group_0)
    if len(df["category_0"].unique()) == 1 and df["group_0"].dtype == "bool":
        df = df[df["group_0"]]
    return df


def get_its_variables(dataframe, cutdate1, cutdate2):
    """
    Format the measures file into an interrupted time series dataframe

    """
    df = dataframe.copy()
    min_year = min(pandas.to_datetime(df.date).dt.year)
    df["rate"] = 1000 * df["value"].astype(float)
    df["time"] = 12 * (pandas.to_datetime(df.date).dt.year - min_year) + (
        pandas.to_datetime(df.date).dt.month
    )
    cutmonth1 = df[df["date"] == cutdate1].iloc[0].time - 1
    cutmonth2 = df[df["date"] == cutdate2].iloc[0].time - 1
    # NOTE: dropping Nov 2022 (not full month of data)
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
    return df


def get_model(
    measure_table,
    pattern,
    interaction_group=None,
    reference=None,
    convert_flat=False,
):
    """
    Return both the its dataframe and the fitted model

    """
    subset = subset_table(measure_table, pattern)
    if interaction_group is not None:
        subset = subset[~subset[interaction_group].isnull()]
    numeric = coerce_numeric(subset)
    # NOTE: remove the incomplete November month
    numeric = numeric[numeric["date"] != "2022-11-01"]
    if convert_flat:
        numeric = flatten(numeric)
    if interaction_group is not None:
        bool_as_int = is_bool_as_int(numeric[interaction_group])
        interaction_category = (
            f"{interaction_group.replace('group', 'category')}"
        )
        if bool_as_int:
            numeric[interaction_group] = numeric.apply(
                lambda x: f"Recored {x[interaction_category]}"
                if x[interaction_group] == "1"
                else f"No recorded {x[interaction_category]}",
                axis=1,
            )
    df = get_its_variables(numeric, STEP_TIME_1, STEP_TIME_2)
    fourier = get_fourier(df)
    formula = get_formula(
        df,
        fourier_terms=fourier,
        interaction_group=interaction_group,
        reference=reference,
    )
    df = pandas.concat([df, fourier], axis=1)
    model = get_regression(df, 2, formula, interaction_group)
    return (model, df)


def lrtest(smaller, bigger):
    ll_smaller = smaller.llf
    ll_bigger = bigger.llf
    stat = -2 * (ll_smaller - ll_bigger)
    return scipy.stats.chi2.sf(stat, 2)


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


#####################
# Plotting functions
#####################


def plot(
    fig,
    pos,
    model,
    df,
    interaction_group,
    other_ax=None,
    title=None,
    ylabel=None,
):
    """
    Plot of observed and fitted values with vertical lines for interruptions

    """
    row, col, index = pos
    ax = fig.add_subplot(row, col, index, sharex=other_ax, sharey=other_ax)
    for group, data in df.groupby(interaction_group):
        predictions = model.get_prediction(data).summary_frame(alpha=0.05)
        ax.plot(
            data["date"], 1000 * predictions["predicted"], label=f"{group}"
        )
        ax.scatter(data["date"], 1000 * data["value"])

    # Plot line marking intervention
    ax.axvline(
        x=pandas.to_datetime(STEP_TIME_1), linestyle="--", color="black"
    )
    ax.axvline(
        x=pandas.to_datetime(STEP_TIME_2), linestyle="--", color="yellow"
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="x-small")
    return ax


def get_ci_df(model):
    """
    Takes a model and returns a dataframe with the coefficient, confidence
    intervals, and error bar

    """
    coef = model.params
    coef.name = "coef"
    cis = model.conf_int().rename(columns={0: "lci", 1: "uci"})
    df = pandas.concat([coef, cis], axis=1)
    df["error"] = df.coef - df.lci
    return df


def get_ci_label(df, round_to=2, pcnt=True):
    if pcnt:
        df = df.apply(
            lambda x: 100 * (numpy.exp(x) - 1) if is_numeric_dtype(x) else x,
            axis=0,
        )
        label = df.apply(
            lambda x: f"{x.coef:.2f}% ({x.lci:.2f}% to {x.uci:.2f}%)"
            if x.coef != 0
            else f"{'-': <15}",
            axis=1,
        )
    else:
        df = df.apply(
            lambda x: numpy.exp(x) if is_numeric_dtype(x) else x, axis=0
        )
        label = df.apply(
            lambda x: f"{x.coef:.2f} ({x.lci:.2f} to {x.uci:.2f})"
            if x.coef != 1
            else f"{'Ref': <27}",
            axis=1,
        )
    df["label"] = label
    return df


def pcnt_change(
    measure_table,
    pattern,
    interaction_group=None,
    reference=None,
    convert_flat=False,
):
    """
    Format a model for a forest plot

    """
    model, its_data = get_model(
        measure_table, pattern, interaction_group, reference, convert_flat
    )
    category_name = "category_0"
    if interaction_group is not None:
        category_name = f"category_{interaction_group.split('_')[-1]}"
    name = its_data[category_name].unique()[0]
    df = get_ci_df(model)
    df = df[df.index.str.contains("T.")]
    keys = list(df.index)
    indices = []
    for key in keys:
        x, y = key.split(".")
        if ":" in x:
            x = x.split(":")[0]
        else:
            x = "baseline"
        group = y.rstrip("]")
        indices.append((x, group))
    df["category"] = name
    df.index = pandas.MultiIndex.from_tuples(
        indices, names=["change", "group"]
    )

    # Manually make the reference group
    # As statsmodels does not include it
    first = df.index.get_level_values(1)[0]
    ref = df[df.index.isin([first], level=1)].copy()
    ref[["coef", "lci", "uci", "error"]] = 0
    ref = ref.reset_index()
    ref["group"] = reference
    ref = ref.set_index(["change", "group"])

    df = pandas.concat([df, ref])
    return df


def group_forest(df, columns, as_rr=[]):
    """
    Create a forest plot with a column for each study period, and a row for
    each subgroup.

    """
    mapping = {
        "baseline": "Baseline Relative Risk",
        "slope": "Lockdown vs. Pre-COVID\nSlope change",
        "slope2": "Recovery vs. Lockdown\nSlope change",
        "step": "Lockdown\nLevel shift",
        "step2": "Recovery\nLevel shift",
    }

    rr = get_ci_label(df[df.index.isin(as_rr, level=0)], pcnt=False)
    pcnt = get_ci_label(df[df.index.isin(columns, level=0)])

    df = pandas.concat([rr, pcnt])
    rows = (
        df.loc[df.index.get_level_values(0)[0]]
        .groupby(["category"])
        .size()
        .values
    )
    ncols = len(df.index.get_level_values(0).unique())
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=ncols,
        figsize=(4 * ncols, 1.5 * len(rows)),
        gridspec_kw={"height_ratios": rows},
        sharex="col",
    )
    grouped = df.groupby(["category", "change"])
    for i, (key, ax) in enumerate(zip(grouped.groups.keys(), axes.flatten())):
        grp = grouped.get_group(key)
        if key[1] in as_rr:
            x_label = "Relative Risk (95% CI)"
            ax_line = 1
        else:
            x_label = "Percent Change (95% CI)"
            ax_line = 0
        if "slope" in key[1]:
            color = "tab:blue"
        elif "step" in key[1]:
            color = "tab:orange"
        else:
            color = "k"
        if i % ncols == 0:
            ax.set_ylabel(key[0])
            y_ticks = [
                "   ".join(x)
                for x in list(zip(grp.index.get_level_values(1), grp.label))
            ]
        else:
            y_ticks = grp.label.to_list()
        ax.errorbar(
            x=grp.coef.values,
            y=list(range(len(y_ticks))),
            xerr=(grp.coef - grp.lci, grp.uci - grp.coef),
            color=color,
            capsize=3,
            linestyle="None",
            linewidth=1,
            marker="o",
            markersize=5,
            mfc=color,
            mec=color,
        )
        ax.set_yticks(list(range(len(y_ticks))), y_ticks)
        if i < ncols:
            ax.set_title(
                mapping[key[1]], loc="center", fontsize=12, fontweight="bold"
            )
        ax.axvline(x=ax_line, linewidth=0.8, linestyle="--", color="black")
        ax.set_xlabel(x_label, fontsize=8)
    plt.tight_layout()
    return fig


def compute_rr(fig, pos, model, df, other_ax=None, title=None):
    row, col, index = pos
    ax = fig.add_subplot(row, col, index, sharex=other_ax, sharey=other_ax)

    # We will only compute RR for restriction and recovery
    df = df.set_index("date")
    intervention_period = df[df.index >= STEP_TIME_1]

    # Compute the fitted values
    fitted = model.get_prediction(intervention_period).summary_frame(
        alpha=0.05
    )
    fitted.index = intervention_period.index

    # Compute the counterfactual (no COVID-19)
    cf_df = intervention_period.copy()
    cols = ["step", "slope", "step2", "slope2", "mar20", "april20"]
    cf_df[cols] = 0
    cf = model.get_prediction(cf_df).summary_frame(alpha=0.05)
    cf.index = intervention_period.index

    # RR is ratio of fitted to predicted
    estimate = numpy.log(fitted["predicted"] / cf["predicted"])
    estimate.name = "RR"

    # Compute the variance of the RR
    # i.e. sum the variance of the terms when simplifying (fitted/cf)
    # Var(A + B) = Var(A) + Var(B) + 2Cov(A, B)
    # or
    # Var(A(x) + B(y)) = x^2Var(A) + y^2Var(B) + 2Cov(A, B)

    # 1. Subset dataset and vcov matrix to relevant terms
    df = intervention_period[cols]
    vcov_full = model.cov_params()
    vcov = vcov_full[cols].loc[cols]

    # 2. Use matrix multiplication to sum the variances and covariances
    var = pandas.Series(numpy.diag((df.dot(vcov)).dot(df.T)), df.index)
    se = numpy.sqrt(var)

    RR = pandas.DataFrame(estimate)
    RR["lci"] = RR["RR"] - 1.96 * se
    RR["uci"] = RR["RR"] + 1.96 * se
    RR = numpy.exp(RR)

    plt.vlines(RR.index, RR.lci, RR.uci, color="k")
    ax.plot(RR.index, RR.RR, color="k")
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(
        "Relative Risk of Antidepressant Prescribing (95% CI)\nCompared to no COVID-19 counterfactual"
    )
    ax.set_xlabel("COVID-19 Period")
    return ax


def translate_to_ci(coefs, name):
    """
    Create a table from the confidence interval dataframe
    """
    df = get_ci_label(coefs)
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
    row = df.label
    row.name = name
    return pandas.DataFrame(row).transpose()


def plot_cf(fig, pos, model, df, other_ax=None, title=None):
    row, col, index = pos
    ax = fig.add_subplot(row, col, index, sharex=other_ax, sharey=other_ax)

    df = df.set_index("date")
    predictions = model.get_prediction(df).summary_frame(alpha=0.05)
    predictions.index = df.index

    # counterfactual assumes no interventions
    cf_df = df.copy()
    cf_df["slope"] = 0.0
    cf_df["step"] = 0.0
    cf_df["mar20"] = 0.0
    cf_df["april20"] = 0.0
    cf_df["slope2"] = 0.0
    cf_df["step2"] = 0.0

    # counter-factual predictions
    cf = model.get_prediction(cf_df).summary_frame(alpha=0.05)
    cf.index = df.index

    # Plot observed data
    ax.scatter(
        df.index,
        1000 * df["value"],
        facecolors="none",
        edgecolors="steelblue",
        label="observed",
        linewidths=2,
    )
    # Plot fitted line
    ax.plot(
        df.index,
        1000 * predictions["predicted"],
        label="model prediction",
        color="k",
    )

    # Plot counterfactual mean rate
    ax.plot(
        df[STEP_TIME_1:].index,
        1000 * cf[STEP_TIME_1:]["predicted"],
        "r--",
        label="No COVID-19 counterfactual",
    )

    # Plot line marking intervention
    ax.axvline(x=STEP_TIME_1, label="March 2020")
    ax.axvline(x=STEP_TIME_2, label="recovery", color="green")
    ax.legend(loc="best")
    plt.xlabel("Months")
    plt.ylabel("Rate per 1,000")
    return ax


#######################
# Measure file loading
#######################


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


def subset_table(measure_table, measures_pattern):
    """
    Given either a pattern of list of names, extract the subset of a joined
    measures file based on the 'name' column
    """
    measures_list = match_paths(measure_table["name"], measures_pattern)
    if len(measures_list) == 0:
        raise ValueError(
            f"Pattern did not match any files: {measures_pattern}"
        )

    return measure_table[measure_table["name"].isin(measures_list)]


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


#######################################
# Tables and figures
#######################################


def figure_1(measure_table):
    # Figure 1
    fig = plt.figure(figsize=(16, 8), dpi=150)

    # All
    model_all, all_data = get_model(
        measure_table, "antidepressant_any_all_total_rate"
    )
    ax = plot(
        fig,
        (3, 2, 1),
        model_all,
        all_data,
        interaction_group="group_0",
        title="Antidepressant Prescribing",
    )
    model_aut, aut_data = get_model(
        measure_table,
        "antidepressant_any_autism_total_rate",
        interaction_group="group_0",
        reference="No recorded autism",
    )
    plot(
        fig,
        (3, 2, 3),
        model_aut,
        aut_data,
        interaction_group="group_0",
        other_ax=ax,
        ylabel="Rate per 1,000 registered patients",
    )
    model_ld, ld_data = get_model(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        interaction_group="group_0",
        reference="No recorded learning_disability",
    )
    plot(
        fig,
        (3, 2, 5),
        model_ld,
        ld_data,
        interaction_group="group_0",
        other_ax=ax,
    )

    # New
    model_new, new_data = get_model(
        measure_table, "antidepressant_any_new_all_total_rate"
    )
    ax_new = plot(
        fig,
        (3, 2, 2),
        model_new,
        new_data,
        interaction_group="group_0",
        title="New Antidepressant Prescribing",
    )
    model_aut_new, aut_new_data = get_model(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        interaction_group="group_0",
        reference="No recorded autism",
    )
    plot(
        fig,
        (3, 2, 4),
        model_aut_new,
        aut_new_data,
        interaction_group="group_0",
        other_ax=ax_new,
        ylabel="Rate per 1,000 antidepressant naive patients",
    )
    model_ld_new, ld_new_data = get_model(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        interaction_group="group_0",
        reference="No recorded learning_disability",
    )
    plot(
        fig,
        (3, 2, 6),
        model_ld_new,
        ld_new_data,
        interaction_group="group_0",
        other_ax=ax_new,
    )

    plt.savefig("figure_1.png")


def forest(measure_table):
    # Forest plot
    model_aut = pcnt_change(
        measure_table,
        "antidepressant_any_autism_total_rate",
        interaction_group="group_0",
        reference="No recorded autism",
    )
    model_ld = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        interaction_group="group_0",
        reference="No recorded learning_disability",
    )
    model_new_aut = pcnt_change(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        interaction_group="group_0",
        reference="No recorded autism",
    )
    model_new_ld = pcnt_change(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        interaction_group="group_0",
        reference="No recorded learning_disability",
    )
    new = pandas.concat([model_new_aut, model_new_ld]).reset_index()
    new.group = new.group + " new"
    new = new.set_index(["change", "group"])
    df = pandas.concat([model_aut, model_ld, new])

    group_forest(df, ["slope", "slope2", "step", "step2"], as_rr=["baseline"])
    plt.savefig("figure_2.png")


def forest_any(measure_table):
    # Forest plot
    # TODO: see if ethnicity/imd errors with panel corrections occur in R
    model_age = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_age_band_rate",
        interaction_group="group_0",
        reference="30-39",
    )
    model_carehome = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_carehome_rate",
        interaction_group="group_0",
        reference="No recorded carehome",
    )
    model_ethnicity = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_ethnicity_rate",
        interaction_group="group_0",
        reference="White",
    )
    model_imd = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_imd_rate",
        interaction_group="group_0",
        reference="3",
    )
    model_region = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_region_rate",
        interaction_group="group_0",
        reference="South East",
    )
    model_sex = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_sex_rate",
        interaction_group="group_0",
        reference="M",
    )
    model_diagnosis = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_diagnosis_18+_rate",
        interaction_group="group_0",
        reference="Depression register",
    )
    model_prescription = pcnt_change(
        measure_table,
        "antidepressant_any_all_breakdown_prescription_count",
        interaction_group="group_0",
        reference="ssri",
    )
    df = pandas.concat(
        [
            model_age,
            model_carehome,
            model_ethnicity,
            model_imd,
            model_region,
            model_sex,
            model_diagnosis,
            model_prescription,
        ]
    )

    group_forest(df, ["slope", "slope2", "step", "step2"], as_rr=["baseline"])
    plt.savefig("figure_3.png")


def forest_autism(measure_table):
    # Forest plot
    model_carehome = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_carehome_rate",
        interaction_group="group_1",
        reference="No recorded carehome",
        convert_flat=True,
    )
    model_age = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_age_band_rate",
        interaction_group="group_1",
        reference="30-39",
        convert_flat=True,
    )
    model_ethnicity = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_ethnicity_rate",
        interaction_group="group_1",
        reference="White",
        convert_flat=True,
    )
    model_imd = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_imd_rate",
        interaction_group="group_1",
        reference="3",
        convert_flat=True,
    )
    # model_region = pcnt_change(
    #    measure_table,
    #    "antidepressant_any_autism_breakdown_region_rate",
    #    interaction_group="group_1",
    #    reference="South East",
    #    convert_flat=True,
    # )
    model_sex = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_sex_rate",
        interaction_group="group_1",
        reference="M",
        convert_flat=True,
    )
    model_diagnosis = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_diagnosis_18+_rate",
        interaction_group="group_1",
        reference="Depression register",
        convert_flat=True,
    )
    model_prescription = pcnt_change(
        measure_table,
        "antidepressant_any_autism_breakdown_prescription_count",
        interaction_group="group_1",
        reference="ssri",
    )
    df = pandas.concat(
        [
            model_age,
            model_carehome,
            model_ethnicity,
            model_imd,
            # model_region,
            model_sex,
            model_diagnosis,
            model_prescription,
        ]
    )

    group_forest(df, ["slope", "slope2", "step", "step2"], as_rr=["baseline"])
    plt.savefig("aut_breakdown.png")


def forest_ld(measure_table):
    # Forest plot
    model_carehome = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_carehome_rate",
        interaction_group="group_1",
        reference="No recorded carehome",
        convert_flat=True,
    )
    model_age = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_age_band_rate",
        interaction_group="group_1",
        reference="30-39",
        convert_flat=True,
    )
    model_ethnicity = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_ethnicity_rate",
        interaction_group="group_1",
        reference="White",
        convert_flat=True,
    )
    model_imd = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_imd_rate",
        interaction_group="group_1",
        reference="3",
        convert_flat=True,
    )
    model_region = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_region_rate",
        interaction_group="group_1",
        reference="South East",
        convert_flat=True,
    )
    model_sex = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_sex_rate",
        interaction_group="group_1",
        reference="M",
        convert_flat=True,
    )
    model_diagnosis = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_diagnosis_18+_rate",
        interaction_group="group_1",
        reference="Depression register",
        convert_flat=True,
    )
    model_prescription = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_breakdown_prescription_count",
        interaction_group="group_1",
        reference="ssri",
    )
    df = pandas.concat(
        [
            model_age,
            model_carehome,
            model_ethnicity,
            model_imd,
            model_region,
            model_sex,
            model_diagnosis,
            model_prescription,
        ]
    )

    group_forest(df, ["slope", "slope2", "step", "step2"], as_rr=["baseline"])
    plt.savefig("ld_breakdown.png")


def table_any_new(measure_table):
    model_all, _ = get_model(
        measure_table, "antidepressant_any_all_total_rate"
    )
    model_new, _ = get_model(
        measure_table, "antidepressant_any_new_all_total_rate"
    )
    all_coef = translate_to_ci(get_ci_df(model_all), "All prescribing")
    new_coef = translate_to_ci(get_ci_df(model_new), "New prescribing")
    table2 = pandas.concat([all_coef, new_coef])
    table2.to_csv("table2.csv")


def plot_all_cf(measure_table):
    fig = plt.figure(figsize=(14, 14), dpi=150)

    model_all, all_data = get_model(
        measure_table, "antidepressant_any_all_total_rate"
    )
    plot_cf(
        fig,
        (2, 1, 1),
        model_all,
        all_data,
        title="Any Antidepressant",
    )

    model_all_new, all_data_new = get_model(
        measure_table, "antidepressant_any_new_all_total_rate"
    )
    plot_cf(
        fig,
        (2, 1, 2),
        model_all_new,
        all_data_new,
        title="New Antidepressant",
    )
    plt.savefig("cf.png")


def plot_all_rr(measure_table):
    fig = plt.figure(figsize=(14, 14), dpi=150)

    model_all, all_data = get_model(
        measure_table, "antidepressant_any_all_total_rate"
    )
    ax = compute_rr(
        fig,
        (2, 1, 1),
        model_all,
        all_data,
        title="Any Antidepressant",
    )

    model_all_new, all_data_new = get_model(
        measure_table, "antidepressant_any_new_all_total_rate"
    )
    compute_rr(
        fig,
        (2, 1, 2),
        model_all_new,
        all_data_new,
        other_ax=ax,
        title="New Antidepressant",
    )
    plt.savefig("rr.png")


def plot_any_breakdowns(measure_table):
    fig = plt.figure(figsize=(14, 12), dpi=150, constrained_layout=True)

    # All
    model_age_band, age_band_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_age_band_rate",
        reference="30-39",
        interaction_group="group_0",
    )
    plot(
        fig,
        (4, 2, 1),
        model_age_band,
        age_band_data,
        interaction_group="group_0",
        title="Age band",
    )
    model_carehome, carehome_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_carehome_rate",
        interaction_group="group_0",
        reference="No recorded carehome",
    )
    plot(
        fig,
        (4, 2, 2),
        model_carehome,
        carehome_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Carehome",
    )
    model_diagnosis, diagnosis_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_diagnosis_18+_rate",
        interaction_group="group_0",
        reference="Depression register",
    )
    plot(
        fig,
        (4, 2, 3),
        model_diagnosis,
        diagnosis_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Diagnosis",
        ylabel="Rate per 1,000 registered patients",
    )
    model_ethnicity, ethnicity_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_ethnicity_rate",
        interaction_group="group_0",
        reference="White",
    )
    plot(
        fig,
        (4, 2, 4),
        model_ethnicity,
        ethnicity_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Ethnicity",
    )
    model_imd, imd_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_imd_rate",
        interaction_group="group_0",
        reference="3",
    )
    plot(
        fig,
        (4, 2, 5),
        model_imd,
        imd_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="IMD",
    )
    model_region, region_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_region_rate",
        interaction_group="group_0",
        reference="South East",
    )
    plot(
        fig,
        (4, 2, 6),
        model_region,
        region_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Region",
    )
    model_sex, sex_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_sex_rate",
        interaction_group="group_0",
        reference="M",
    )
    plot(
        fig,
        (4, 2, 7),
        model_sex,
        sex_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Sex",
    )
    model_prescription, prescription_data = get_model(
        measure_table,
        "antidepressant_any_all_breakdown_prescription_count",
        interaction_group="group_0",
        reference="ssri",
    )
    plot(
        fig,
        (4, 2, 8),
        model_prescription,
        prescription_data,
        interaction_group="group_0",
        # other_ax=ax,
        title="Prescription",
    )

    plt.savefig("any_breakdown.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to single joined measures file",
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
    output_dir = args.output_dir

    measure_table = get_measure_tables(input_file)
    measure_table = measure_table[
        ~(
            measure_table["name"].str.contains("prescription")
            & (
                (measure_table["group_0"] == "Unknown")
                | (measure_table["group_1"] == "Unknown")
            )
        )
    ].reset_index(drop=True)

    # figure_1(measure_table)
    # forest(measure_table)

    # forest_any(measure_table)
    # forest_autism(measure_table)
    # forest_ld(measure_table)

    # plot_all_cf(measure_table)
    # plot_all_rr(measure_table)
    # table_any_new(measure_table)


if __name__ == "__main__":
    main()
