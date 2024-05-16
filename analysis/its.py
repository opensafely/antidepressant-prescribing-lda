import argparse
import pathlib
import fnmatch
import pandas
import numpy
import scipy

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pandas.api.types import is_numeric_dtype

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.deterministic import Fourier


STEP_TIME_1 = pandas.to_datetime("2020-03-01")
STEP_TIME_2 = pandas.to_datetime("2021-03-01")

# prescription needs to be count
DEMOGRAPHICS = [
    "total",
    "age_band",
    "carehome",
    "diagnosis_18+",
    "ethnicity",
    "imd",
    "region",
    "sex",
    "prescription",
]

MAPPING = {
    "baseline": "Baseline Relative Risk",
    "slope": "Lockdown vs. Pre-COVID\nSlope change",
    "slope2": "Recovery vs. Lockdown\nSlope change",
    "step": "Lockdown\nLevel shift",
    "step2": "Recovery\nLevel shift",
}

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


def get_formula(df, fourier_terms=None, group=None, reference=None):
    """
    Build formula depending upon whether it has an interaction for panel
    data

    """
    formula = (
        "numerator ~ time + step + slope + mar20 + april20 + step2 + slope2"
    )
    if group is not None:
        # TODO: raise an error if more than 1 category value
        df[group] = df[group].astype("category")
        levels = list(df[group].unique())
        levels.remove(reference)
        levels = [reference] + levels
        df[group] = df[group].cat.reorder_categories(levels)
        formula += (
            f"+ step*{group} + slope*{group} + step2*{group} + slope2*{group}"
        )
    if fourier_terms is not None:
        formula += "+ " + "+".join(fourier_terms)
    return formula


def get_regression(df, lags, formula, interaction=False):
    """
    Fit the model
    Adjust the standard errors with NeweyWest (single sequence) or for panel
    data

    """
    if not interaction:
        try:
            model_errors = smf.poisson(
                formula,
                data=df,
                exposure=df.denominator,
            ).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, maxiter=200)
        except Exception:
            return None
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
    if not model_errors.mle_retvals["converged"]:
        return None
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


def bool_to_category(subset, group):
    """
    We will want our plots to have the text 'Recorded x' rather
    than True/False
    """
    subset = subset.copy()
    category = f"{group.replace('group', 'category')}"
    bool_as_int = is_bool_as_int(subset[group])
    if bool_as_int:
        subset[group] = subset.apply(
            lambda x: f"Recorded {x[category]}"
            if x[group] == "1"
            else f"No recorded {x[category]}",
            axis=1,
        )
    return subset


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


def get_model_short(subset, group, reference, interaction):
    df = get_its_variables(subset, STEP_TIME_1, STEP_TIME_2)
    fourier = get_fourier(df)
    formula = get_formula(
        df,
        fourier_terms=fourier,
        group=group,
        reference=reference,
    )
    df = pandas.concat([df, fourier], axis=1)
    model = get_regression(df, 2, formula, interaction)
    return (model, df)


def get_models(
    measure_table,
    pattern,
    group,
    reference=None,
    interaction=False,
):
    """
    Return a list of (dataframe, fitted model) tuples
    Based on whether it is an interaction model, a single subset, or separate
    models for each subset
    """
    subset = subset_table(measure_table, pattern)
    subset = bool_to_category(subset, group)
    subset = subset[~subset[group].isnull()]

    # If group_1 is used, then we only want to take the true values for 0
    # We do not support two-level interactions
    if group == "group_1":
        subset = subset_group(subset, "group_0", "1")

    models = []
    if interaction:
        if not reference:
            raise Exception("Need to have a reference group")
        # We can return one model with the interaction
        models.append(get_model_short(subset, group, reference, interaction))
    elif reference:
        # We can return one model subsetting the reference
        subset = subset_group(subset, group, reference)
        models.append(get_model_short(subset, group, reference, interaction))
    else:
        # Return a list of models, one for each category
        for subgroup, subgroup_data in subset.groupby(group):
            models.append(get_model_short(subgroup_data, None, None, False))
    return models


def lrtest(smaller, bigger):
    ll_smaller = smaller.llf
    ll_bigger = bigger.llf
    stat = -2 * (ll_smaller - ll_bigger)
    return scipy.stats.chi2.sf(stat, 2)


def check_residuals(model, path, name):
    """
    Visualize autocorrelation with the acf and pacf
    Check normality of residuals, and qq plot
    """
    resid = model.resid_pearson

    sm.graphics.tsa.plot_acf(resid, lags=40)
    plt.savefig(path / f"{name}_acf.png")

    sm.graphics.tsa.plot_pacf(resid, lags=20, method="ywm")
    plt.savefig(path / f"{name}_pacf.png")


def output_acf_pacf(measure_table, output_dir):
    residuals_dir = output_dir / "residuals"
    residuals_dir.mkdir(exist_ok=True)
    model_all = get_models(
        measure_table, "antidepressant_any_all_total_rate", "group_0"
    )[0][0]
    check_residuals(model_all, residuals_dir, "model_all_noerr")


###########################
# Plotting helper functions
###########################


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


def get_ci_label(df, precision=1, how="pcnt"):
    """
    Translate a dataframe with coef, lci, uci into a CI string
    Either as percent change or RR
    If a coef is 0 or 1 (for pcnt, RR), then it is the ref group
    If a coef is nan, display as "-"
    """
    if how == "nop":
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'-': <15}"
            if x.coef == 0
            else f"{x.coef:.{precision}f} ({x.lci:.{precision}f} to {x.uci:.{precision}f})",  # noqa: E501
            axis=1,
        )
    elif how == "pcnt":
        df = df.apply(
            lambda x: 100 * (numpy.exp(x) - 1) if is_numeric_dtype(x) else x,
            axis=0,
        )
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'-': <15}"
            if x.coef == 0
            else f"{x.coef:.{precision}f}% ({x.lci:.{precision}f}% to {x.uci:.{precision}f}%)",  # noqa: E501
            axis=1,
        )
    elif how == "rr":
        df = df.apply(
            lambda x: numpy.exp(x) if is_numeric_dtype(x) else x, axis=0
        )
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'Ref': <27}"
            if x.coef == 1
            else f"{x.coef:.{precision}f} ({x.lci:.{precision}f} to {x.uci:.{precision}f})",  # noqa: E501
            axis=1,
        )
    # Rate
    else:
        df = df.apply(
            lambda x: 1000 * x if is_numeric_dtype(x) else x,
            axis=0,
        )
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'-': <15}"
            if x.coef == 0
            else f"{x.coef:.{precision}f} ({x.lci:.{precision}f} to {x.uci:.{precision}f})",  # noqa: E501
            axis=1,
        )
    df["label"] = label
    return df


def translate_to_ci(coefs, name):
    """
    Create a table from the confidence interval dataframe
    """
    df = get_ci_label(coefs, how="pcnt")
    mapping = {
        "time": ("", "Pre-COVID-19 monthly slope (95% CI)"),
        "step": ("Lockdown period", "Level shift (95% CI)"),
        "slope": ("Lockdown period", "Change in slope (95% CI)"),
        "step2": ("Recovery period", "Level shift (95% CI)"),
        "slope2": ("Recovery period", "Change in slope (95% CI)"),
    }
    df = df.loc[mapping.keys()]
    df = df.set_index(pandas.MultiIndex.from_tuples(mapping.values()))
    row = df.label
    row.name = name
    return pandas.DataFrame(row).transpose()


def plot_cf(model, df, ax):
    # Single model with counterfactual
    predictions = model.get_prediction(df).summary_frame(alpha=0.05)
    predictions.index = df.index
    df = df.set_index("date")

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
        s=10,
        facecolors="k",
        linewidths=2,
    )
    # Plot fitted line
    ax.plot(
        df.index,
        1000 * predictions["predicted"],
        label="Fitted values",
        color="k",
    )

    # Plot counterfactual mean rate
    ax.plot(
        df[STEP_TIME_1:].index,
        1000 * cf[STEP_TIME_1:]["predicted"],
        "r--",
        label="No COVID-19 counterfactual",
    )

    # Plot counterfactual CI
    ax.fill_between(
        df[STEP_TIME_1:].index,
        1000 * cf[STEP_TIME_1:]["ci_lower"],
        1000 * cf[STEP_TIME_1:]["ci_upper"],
        color="gray",
        alpha=0.1,
    )


def plot_group(
    measure_table, output_dir, pattern, group, rr=False, legend_per_ax=True
):
    """
    Given a measures pattern, create a panel plot for each unique group
    Option to display either the counter factual plot, or the relative risk
    plot
    """
    models = get_models(measure_table, pattern, group)
    total_rows = (len(models)) // 2

    fig = plt.figure(figsize=(16, 5 * total_rows))

    for index, model in enumerate(models):
        category = model[1][group].iloc[0]
        ax = add_subplot(
            fig,
            (total_rows, 2, index + 1),
            [model],
            group=group,
            how="rr" if rr else "cf",
            title=f"{category.title()}",
            legend_per_ax=legend_per_ax,
        )
        if index < (len(models) - 2):
            ax.set_xticklabels([])

    fig.legend(*ax.get_legend_handles_labels())
    fig.supylabel("Rate per 1,000 registered patients")
    fig.supxlabel("Date")
    plt.savefig(output_dir / f"{pattern}_fig.png")


def group_forest(df, as_pcnt=[], as_rr=[], as_rate=[], mapping=None):
    """
    Create a forest plot with a column for each study period, and a row for
    each subgroup.

    """
    frames = []
    if as_rr:
        rr = get_ci_label(
            df[df.index.isin(as_rr, level=0)], precision=2, how="rr"
        )
        frames.append(rr)
    if as_pcnt:
        pcnt = get_ci_label(
            df[df.index.isin(as_pcnt, level=0)], precision=1, how="pcnt"
        )
        frames.append(pcnt)
    if as_rate:
        rate = get_ci_label(
            df[df.index.isin(as_rate, level=0)], precision=1, how="rate"
        )
        frames.append(rate)

    df = pandas.concat(frames)
    # Ensure they all have the same data
    reset = df.reset_index()
    all_cols = reset[["change"]].drop_duplicates()
    required = reset[["group", "category"]].drop_duplicates()
    all_rows = required.merge(all_cols, how="cross")
    df = reset.merge(
        all_rows,
        how="outer",
        left_on=["change", "group", "category"],
        right_on=["change", "group", "category"],
    ).set_index(["change", "group"])
    df.loc[:, "label"] = df.label.fillna("-")

    rows = (
        df.loc[df.index.get_level_values(0)[0]]
        .groupby(["category"], sort=False)
        .size()
        .values
    )
    ncols = len(df.index.get_level_values(0).unique())
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=ncols,
        figsize=(5.5 * ncols, 1.5 * len(rows)),
        gridspec_kw={"height_ratios": rows},
        # sharex="all",
        sharex="col",
    )
    grouped = df.groupby(["category", "change"], sort=False)
    for i, (key, ax) in enumerate(zip(grouped.first().index, axes.flatten())):
        grp = grouped.get_group(key)
        if key[1] in as_rr:
            x_label = "Relative Risk (95% CI)"
            ax_line = 1
        elif key[1] in as_rate:
            x_label = "Rate Difference per 1,000 (95% CI)"
            ax_line = 0
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
            # NOTE: we cannot rjust the entire thing because font
            # has different widths for different characters
            y_ticks = [
                "    ".join(x)
                if x[1] != "-"
                else x[0] + "    " + x[1].rjust(38)
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
                (mapping[key[1]] if mapping else key[1]),
                loc="center",
                fontsize=12,
                fontweight="bold",
            )
        if i >= (ncols * len(rows) - ncols):
            ax.set_xlabel(x_label)
        ax.axvline(x=ax_line, linewidth=0.8, linestyle="--", color="black")
    plt.tight_layout()
    return fig


def expanding_gmean_log(s):
    """
    Compute the geometric mean, but take the log and use cumsum for efficiency
    And to avoid overflow
    """
    return numpy.log(s).cumsum() / (numpy.arange(len(s)) + 1)


def compute_coef(row, vcov):
    """
    Compute the coefficients needed for the geometric mean
    Since we are on the log scale, we can sum the coefficients for each time
    period, and use sum(1...n) = n(n+1)/2 to simplify
    When raising to the power of 1/n, we divide each coefficient by n
    """
    step_total = row["slope"]
    step2_total = row["slope2"]
    geom_coef = pandas.Series(
        {
            "step": 1,
            "slope": (step_total + 1) / 2,
            "step2": (step2_total / step_total),
            "slope2": (step2_total * (step2_total + 1)) / (step_total * 2),
            "mar20": 1 / step_total,
            "april20": (1 / step_total) if step_total > 1 else 0,
        }
    )
    geom_se = numpy.sqrt(geom_coef.dot(vcov).dot(geom_coef))
    return geom_se


def compute_difference(model, df, rr=True):
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

    if rr:
        # RR is ratio of fitted to predicted
        estimate = numpy.log(fitted["predicted"] / cf["predicted"])
        estimate.name = "coef"

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

        diff = pandas.DataFrame(estimate)
        diff["lci"] = diff["coef"] - 1.96 * se
        diff["uci"] = diff["coef"] + 1.96 * se
        diff = numpy.exp(diff)
        return (diff, df, vcov)
    else:
        estimate = fitted["predicted"] - cf["predicted"]
        estimate.name = "coef"
        se = numpy.sqrt(
            (fitted["se"] * fitted["se"]).add(
                (cf["se"] * cf["se"]), fill_value=0
            )
        )
        diff = pandas.DataFrame(estimate)
        diff["lci"] = diff["coef"] - 1.96 * se
        diff["uci"] = diff["coef"] + 1.96 * se
        diff["error"] = se
        return (diff, df, None)


def get_summary(model, df, rr=True, average=True, time=None):
    diff, df, vcov = compute_difference(model, df, rr)
    if not average:
        # If no time is specified, take the last month
        if time:
            row = diff.loc[time:time]
        else:
            row = diff.iloc[-1:]
        if rr:
            return numpy.log(row)
        else:
            return row
    if rr:
        gm = expanding_gmean_log(diff["coef"])
        geom_se = df.apply(compute_coef, vcov=vcov, axis=1)
        MEAN = pandas.DataFrame(gm)
        MEAN["lci"] = MEAN["coef"] - 1.96 * geom_se
        MEAN["uci"] = MEAN["coef"] + 1.96 * geom_se
        MEAN["error"] = geom_se
        return MEAN
    else:
        raise Exception("Average mean difference is not supported")


def display_difference(model, df, ax, rr=True, average=True):
    diff, _, _ = compute_difference(model, df, rr)
    result = get_summary(model, df, rr, average)
    if rr:
        ax.axhline(y=1.0, color="r", linestyle="--")
        ax.set_ylabel(
            "Relative Risk of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        m = get_ci_label(result, precision=2, how="rr").iloc[-1].label
    else:
        ax.axhline(y=0.0, color="r", linestyle="--")
        ax.set_ylabel(
            "Rate Difference of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        m = get_ci_label(result, precision=2, how="rate").iloc[-1].label
        diff = 1000 * diff
    plt.vlines(diff.index, diff.lci, diff.uci, color="k")
    ax.plot(diff.index, diff["coef"], color="k")
    ax.text(
        1.05,
        0.7,
        "Mean over\ntime period:\n" + m,
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "red", "alpha": 0.5},
    )


def add_subplot(
    fig,
    pos,
    models,
    group=None,
    how="rr",
    other_ax=None,
    title=None,
    ylabel=None,
    legend_per_ax=True,
):
    """
    Create subplot of a figure
    Plot of observed and fitted values with vertical lines for interruptions
    If there is only one group (no interaction) plot the counterfactual

    """
    row, col, index = pos
    ax = fig.add_subplot(row, col, index, sharex=other_ax, sharey=other_ax)
    # Plot observed and fitted subgroups on the same plot
    if len(models) > 1:
        for model, data in models:
            predictions = model.get_prediction(data).summary_frame(alpha=0.05)
            ax.plot(
                data["date"],
                1000 * predictions["predicted"],
                label=f"{data[group].iloc[0]}",
            )
            ax.scatter(data["date"], 1000 * data["value"])
    else:
        model, df = models[0]
        # Interaction
        if group and df[group].nunique() > 1:
            for group, data in df.groupby(group):
                predictions = model.get_prediction(data).summary_frame(
                    alpha=0.05
                )
                ax.plot(
                    data["date"],
                    1000 * predictions["predicted"],
                    label=f"{group}",
                )
                ax.scatter(data["date"], 1000 * data["value"])

        else:
            if how == "rr" or how == "rate":
                display_difference(model, df, ax, how == "rr")
            else:
                plot_cf(model, df, ax)
        # Plot line marking intervention
        ax.axvline(
            x=STEP_TIME_1, linestyle="--", color="blue", label="Lockdown"
        )
        ax.axvline(
            x=STEP_TIME_2, linestyle="--", color="green", label="Recovery"
        )
    ax.set_title(title, fontsize="large")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%b-%Y"))
    if not legend_per_ax:
        ax.legend().set_visible(False)
    return ax


#######################
# Measure file loading
#######################


def get_measure_tables(input_file):
    # The `date` column is assigned by the measures framework.
    measure_table = pandas.read_csv(
        input_file,
        parse_dates=["date"],
        dtype={"numerator": float, "denominator": float, "value": float},
        na_values="[REDACTED]",
    )

    return measure_table


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


def subset_group(measure_table, group, reference):
    return measure_table[measure_table[group] == reference]


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(files, pattern):
    return fnmatch.filter(files, pattern)


#######################################
# Tables and figures
#######################################

# Figure 1 and supplemental
def plot_all_cf(measure_table, output_dir, legend_per_ax=True, how="rr"):
    fig = plt.figure(figsize=(14, 14), dpi=150)
    models = get_models(
        measure_table,
        "antidepressant_any_all_total_rate",
        "group_0",
    )
    ax = add_subplot(
        fig,
        (2, 1, 1),
        models,
        how=how,
        title="Any Antidepressant",
        ylabel="Rate per 1,000 registered patients" if how == "cf" else "",
        legend_per_ax=legend_per_ax,
    )
    models_new = get_models(
        measure_table, "antidepressant_any_new_all_total_rate", "group_0"
    )
    add_subplot(
        fig,
        (2, 1, 2),
        models_new,
        how=how,
        title="New Antidepressant",
        ylabel="Rate per 1,000 AD naive registered patients"
        if how == "cf"
        else "",
        legend_per_ax=legend_per_ax,
    )
    fig.legend(*ax.get_legend_handles_labels())
    fig.supxlabel("Date")
    if how == "rr":
        fig.supylabel(
            "Relative Risk of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        plt.savefig(output_dir / "f1_sup.png", bbox_inches="tight")
    elif how == "rate":
        fig.supylabel(
            "Rate Difference of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        plt.savefig(output_dir / "f1_sup.png", bbox_inches="tight")
    else:
        plt.savefig(output_dir / "figure_1.png", bbox_inches="tight")


# Figure 2 and supplemental
def figure_2(measure_table, output_dir, how="rr", legend_per_ax=True):
    # Figure 1
    fig = plt.figure(figsize=(10, 16), dpi=150)
    # fig = plt.figure(figsize=(16, 8), dpi=150)

    models_aut = get_models(
        measure_table,
        "antidepressant_any_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    ax = add_subplot(
        fig,
        (4, 1, 1),
        models_aut,
        group="group_0",
        how=how,
        ylabel="Rate per 1,000 autism patients" if how != "rr" else "",
        title="Antidepressant Prescribing Autism",
        legend_per_ax=legend_per_ax,
    )

    models_ld = get_models(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    add_subplot(
        fig,
        (4, 1, 3),
        models_ld,
        how=how,
        group="group_0",
        ylabel="Rate per 1,000 LD patients" if how != "rr" else "",
        title="Antidepressant Prescribing Learning Disability",
        legend_per_ax=legend_per_ax,
    )

    models_aut_new = get_models(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    add_subplot(
        fig,
        (4, 1, 2),
        models_aut_new,
        group="group_0",
        how=how,
        ylabel="Rate per 1,000 AD naive autism patients"
        if how != "rr"
        else "",
        title="New Antidepressant Prescribing Autism",
        legend_per_ax=legend_per_ax,
    )

    models_ld_new = get_models(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    add_subplot(
        fig,
        (4, 1, 4),
        models_ld_new,
        group="group_0",
        how=how,
        ylabel="Rate per 1,000 AD naive LD patients" if how != "rr" else "",
        title="New Antidepressant Prescribing Learning Disability",
        legend_per_ax=legend_per_ax,
    )
    fig.legend(*ax.get_legend_handles_labels())
    fig.supxlabel("Date")
    # Set all axes y labels to empty string
    plt.subplots_adjust(hspace=0.4)
    # bbox_inches allows us to keep the legend
    if how == "rr":
        fig.supylabel(
            "Relative Risk of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        plt.savefig(output_dir / "f2_sup.png", bbox_inches="tight")
    elif how == "rate":
        fig.supylabel(
            "Rate Difference of Antidepressant Prescribing (95% CI)\n"
            + "Compared to no COVID-19 counterfactual"
        )
        plt.savefig(output_dir / "f2_sup.png", bbox_inches="tight")
    else:
        plt.savefig(output_dir / "figure_2.png", bbox_inches="tight")


# Table 3
def table_any_new(measure_table, output_dir):
    model_all = get_models(
        measure_table, "antidepressant_any_all_total_rate", "group_0"
    )[0][0]
    model_new = get_models(
        measure_table, "antidepressant_any_new_all_total_rate", "group_0"
    )[0][0]
    model_aut = get_models(
        measure_table,
        "antidepressant_any_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )[0][0]
    model_ld = get_models(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )[0][0]
    model_aut_new = get_models(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )[0][0]
    model_ld_new = get_models(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )[0][0]

    all_coef = translate_to_ci(get_ci_df(model_all), "All prescribing")
    aut_coef = translate_to_ci(get_ci_df(model_aut), "Autism prescribing")
    ld_coef = translate_to_ci(get_ci_df(model_ld), "LD prescribing")
    new_coef = translate_to_ci(get_ci_df(model_new), "New prescribing")
    aut_new_coef = translate_to_ci(
        get_ci_df(model_aut_new), "Autism new prescribing"
    )
    ld_new_coef = translate_to_ci(
        get_ci_df(model_ld_new), "LD new prescribing"
    )
    table = pandas.concat(
        [all_coef, aut_coef, ld_coef, new_coef, aut_new_coef, ld_new_coef]
    )
    table.to_html(output_dir / "table3.html")


# Figure 3
def forest_mean_rr(
    measure_table,
    output_dir,
    population="all",
    column_titles={},
    rr=True,
    average=False,
):
    results = []
    titles = []
    for demo in DEMOGRAPHICS:
        extension = f"breakdown_{demo}_rate"
        group = "group_1"
        if demo == "total":
            extension = "total_rate"
            group = "group_0"
        elif demo == "prescription":
            extension = f"breakdown_{demo}_count"
        for column, title in column_titles.items():
            titles.append(title)
            try:
                results.append(
                    table_mean(
                        measure_table,
                        f"antidepressant_any_{column}_{extension}",
                        title,
                        "group_0" if "all" in column else group,
                        reference=f"Recorded {population}"
                        if group == "group_0" and population != "all"
                        else None,
                        rr=rr,
                        average=average,
                    )
                )
            except:
                pass
    df = pandas.concat(results)
    group_forest(
        df, as_rr=(titles if rr else []), as_rate=(titles if not rr else [])
    )
    output_name = (
        f"forest_{'_'.join(column_titles.keys())}_"
        + f"{'gm' if average else 'end'}.png"
    )
    plt.savefig(output_dir / output_name, bbox_inches="tight")


def first_lockdown(tup, label):
    model, data = tup
    predictions = model.get_prediction(data).summary_frame(alpha=0.05)
    return 1000 * pandas.DataFrame(
        {
            "Study Start": predictions.predicted.iloc[0],
            "Feb 2020": predictions.predicted.iloc[26],
        },
        index=[label],
    )


def model_fitted_rate(measure_table, output_dir):
    model_all = get_models(
        measure_table, "antidepressant_any_all_total_rate", "group_0"
    )[0]
    model_new = get_models(
        measure_table, "antidepressant_any_new_all_total_rate", "group_0"
    )[0]
    model_aut = get_models(
        measure_table,
        "antidepressant_any_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )[0]
    model_ld = get_models(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )[0]
    model_aut_new = get_models(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )[0]
    model_ld_new = get_models(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )[0]
    all_coef = first_lockdown(model_all, "All prescribing")
    aut_coef = first_lockdown(model_aut, "Autism prescribing")
    ld_coef = first_lockdown(model_ld, "LD prescribing")
    new_coef = first_lockdown(model_new, "New prescribing")
    aut_new_coef = first_lockdown(model_aut_new, "Autism new prescribing")
    ld_new_coef = first_lockdown(model_ld_new, "LD new prescribing")
    table = pandas.concat(
        [all_coef, aut_coef, ld_coef, new_coef, aut_new_coef, ld_new_coef]
    )
    table.to_csv(output_dir / "model_fitted_rates.csv", index=True)


def table_mean(
    measure_table,
    pattern,
    label,
    group=None,
    reference=None,
    rr=False,
    average=True,
    time=None,
):
    category = f"{group.replace('group', 'category')}"
    models = get_models(measure_table, pattern, group, reference=reference)
    rows = {}
    for model, data in models:
        group_label = "Overall" if len(models) == 1 else data[group].iloc[-1]
        category_label = (
            "population" if len(models) == 1 else data[category].iloc[-1]
        )
        # label = "Overall" if len(models) == 1 else label
        if model:
            coefs = get_summary(model, data, rr, average, time)
            row = coefs.iloc[-1].copy()
        else:
            row = pandas.Series()
        row["category"] = category_label
        rows[(label, group_label)] = row
    df = pandas.concat(rows, axis=1).T
    # Cast all the data as float, but skip errors per-column
    # When we have nans, we have an empty series initialised with a string
    # This causes the other columns to be converted to objects
    df = df.apply(pandas.to_numeric, errors="ignore")
    df.index.names = ["change", "group"]
    return df


def mean_difference(measure_table, output_dir):
    results_recovery = []

    row_all = table_mean(
        measure_table,
        "antidepressant_any_all_total_rate",
        "All Prescribing",
        "group_0",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_all.category = "all"
    results_recovery.append(row_all)
    row_new = table_mean(
        measure_table,
        "antidepressant_any_new_all_total_rate",
        "New Prescribing",
        "group_0",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_new.category = "new"
    results_recovery.append(row_new)
    row_autism = table_mean(
        measure_table,
        "antidepressant_any_autism_total_rate",
        "Autism Prescribing",
        "group_0",
        reference="Recorded autism",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_autism.category = "all"
    results_recovery.append(row_autism)
    row_new_autism = table_mean(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        "New Autism Prescribing",
        "group_0",
        reference="Recorded autism",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_new_autism.category = "new"
    results_recovery.append(row_new_autism)
    row_ld = table_mean(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        "LD Prescribing",
        "group_0",
        reference="Recorded learning_disability",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_ld.category = "all"
    results_recovery.append(row_ld)
    row_new_ld = table_mean(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        "New LD Prescribing",
        "group_0",
        reference="Recorded learning_disability",
        rr=False,
        average=False,
        time=STEP_TIME_2,
    )
    row_new_ld.category = "new"
    results_recovery.append(row_new_ld)
    df_recovery = pandas.concat(results_recovery)
    df_recovery = df_recovery.reset_index()
    df_recovery.group = "Recovery Start (March 2021)"
    df_recovery = df_recovery.set_index(["change", "group"])

    results_last = []
    row_all = table_mean(
        measure_table,
        "antidepressant_any_all_total_rate",
        "All Prescribing",
        "group_0",
        rr=False,
        average=False,
    )
    row_all.category = "all"
    results_last.append(row_all)
    row_new = table_mean(
        measure_table,
        "antidepressant_any_new_all_total_rate",
        "New Prescribing",
        "group_0",
        rr=False,
        average=False,
    )
    row_new.category = "new"
    results_last.append(row_new)
    row_autism = table_mean(
        measure_table,
        "antidepressant_any_autism_total_rate",
        "Autism Prescribing",
        "group_0",
        reference="Recorded autism",
        rr=False,
        average=False,
    )
    row_autism.category = "all"
    results_last.append(row_autism)
    row_new_autism = table_mean(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        "New Autism Prescribing",
        "group_0",
        reference="Recorded autism",
        rr=False,
        average=False,
    )
    row_new_autism.category = "new"
    results_last.append(row_new_autism)
    row_ld = table_mean(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        "LD Prescribing",
        "group_0",
        reference="Recorded learning_disability",
        rr=False,
        average=False,
    )
    row_ld.category = "all"
    results_last.append(row_ld)
    row_new_ld = table_mean(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        "New LD Prescribing",
        "group_0",
        reference="Recorded learning_disability",
        rr=False,
        average=False,
    )
    row_new_ld.category = "new"
    results_last.append(row_new_ld)
    df_last = pandas.concat(results_last)
    df_last = df_last.reset_index()
    df_last.group = "Study End (December 2022)"
    df_last = df_last.set_index(["change", "group"])

    df = pandas.concat([df_recovery, df_last])
    df = df.sort_index()

    df.index = df.index.swaplevel()
    df.index.names = ["change", "group"]
    group_forest(
        df,
        as_rate=["Recovery Start (March 2021)", "Study End (December 2022)"],
    )
    plt.savefig(output_dir / "forest_mean_diff.png", bbox_inches="tight")


def plot_openprescribing(measure_table, output_dir, legend_per_ax=False):
    fig = plt.figure(figsize=(10, 10), dpi=150)

    models = get_models(
        measure_table,
        "antidepressant_any_all_openprescribing_total_rate",
        "group_0",
    )
    ax = add_subplot(
        fig,
        (2, 1, 1),
        models,
        how="cf",
        title="Any Antidepressant",
        ylabel="Rate per 1,000 registered patients",
        legend_per_ax=legend_per_ax,
    )
    ax = add_subplot(
        fig,
        (2, 1, 2),
        models,
        how="rr",
        legend_per_ax=legend_per_ax,
    )
    fig.legend(*ax.get_legend_handles_labels())
    fig.supxlabel("Date")
    plt.savefig(output_dir / "openprescribing.png", bbox_inches="tight")


def plot_any_breakdowns(measure_table, output_dir, new=False):
    fig = plt.figure(figsize=(14, 20), dpi=150)
    if new:
        new_string = "new_"
    else:
        new_string = ""
    # All
    models_age_band = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_age_band_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 1),
        models_age_band,
        group="group_0",
        title="Age band",
    )
    models_carehome = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_carehome_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 2),
        models_carehome,
        group="group_0",
        title="Carehome",
    )
    models_diagnosis = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_diagnosis_18+_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 3),
        models_diagnosis,
        group="group_0",
        title="Diagnosis",
    )
    models_ethnicity = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_ethnicity_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 4),
        models_ethnicity,
        group="group_0",
        title="Ethnicity",
    )
    models_imd = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_imd_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 5),
        models_imd,
        group="group_0",
        title="IMD",
    )
    models_region = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_region_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 6),
        models_region,
        group="group_0",
        title="Region",
    )
    models_sex = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_sex_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (8, 1, 7),
        models_sex,
        group="group_0",
        title="Sex",
    )
    if not new:
        models_prescription = get_models(
            measure_table,
            f"antidepressant_any_{new_string}all_breakdown_prescription_count",
            group="group_0",
        )
        add_subplot(
            fig,
            (8, 1, 8),
            models_prescription,
            group="group_0",
            title="Prescription",
        )
    fig.supylabel("Rate per 1,000 registered patients")
    fig.supxlabel("Date")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(output_dir / f"any_{new_string}breakdown.png")


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

    import seaborn as sns

    sns.set_theme(style="whitegrid")

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

    # Replace prescription the denominator with the correct one
    # (b/c of measures framework, they all have the same denom)
    for name, table in measure_table[
        measure_table["name"].str.contains("prescription")
    ].groupby("name"):
        on = ["date"]
        total_name = name.replace("breakdown_prescription_count", "total_rate")
        totals = measure_table[measure_table["name"] == total_name]
        if totals.group_0.iloc[0] != "population":
            on.append("group_0")
        totals = totals[["date", "denominator", "group_0"]]
        denominators = pandas.merge(
            table, totals, on=on, how="left", suffixes=("_x", "")
        )
        denominators.index = table.index
        measure_table.loc[denominators.index, "denominator"] = denominators

    measure_table.loc[:, "value"] = (
        measure_table.numerator / measure_table.denominator
    )

    # Figure 1
    # CF plot for all, new
    plot_all_cf(measure_table, output_dir, how="cf", legend_per_ax=False)

    # Figure 2
    # CF plot for autism/LD
    figure_2(measure_table, output_dir, how="cf", legend_per_ax=False)

    # Table 3
    # Table of coefficients for each of the models
    table_any_new(measure_table, output_dir)

    # Figure 3
    # Forest plot of GM for all overall and by breakdown
    forest_mean_rr(
        measure_table,
        output_dir,
        population="all",
        column_titles={"all": "All Prescribing", "new_all": "New Prescribing"},
        rr=True,
        average=True,
    )

    # For context in manuscript text
    # i.e. from the study start to feb 2020 absolute prescribing increased
    # from a to b per 1,000
    model_fitted_rate(measure_table, output_dir)

    # Figure S1
    plot_all_cf(measure_table, output_dir, how="rr", legend_per_ax=False)

    # Figure S2
    figure_2(measure_table, output_dir, how="rr", legend_per_ax=False)

    # Figure S3
    # Fitted plot by demographic group
    # Can specify new/ not new
    # Can modify the function to use interaction
    plot_any_breakdowns(measure_table, output_dir, new=False)

    # AD by breakdown
    # Figure S4
    plot_group(
        measure_table,
        output_dir,
        "antidepressant_any_all_breakdown_diagnosis_18+_rate",
        "group_0",
        legend_per_ax=False,
    )

    # Figure S5
    plot_group(
        measure_table,
        output_dir,
        "antidepressant_any_all_breakdown_age_band_rate",
        "group_0",
        legend_per_ax=False,
    )

    # Figure S6
    plot_group(
        measure_table,
        output_dir,
        "antidepressant_any_all_breakdown_prescription_count",
        "group_0",
        legend_per_ax=False,
    )

    # Figure S7
    # OpenPrescribing
    # NOTE: this might not work on the server
    # op_file = pathlib.Path("analysis/openprescribing/openprescribing.csv")
    # op_table = get_measure_tables(op_file)
    # plot_openprescribing(op_table, output_dir)

    # Figure S8
    mean_difference(measure_table, output_dir)

    # Figure S9
    forest_mean_rr(
        measure_table,
        output_dir,
        population="all",
        column_titles={"all": "All Prescribing", "new_all": "New Prescribing"},
        rr=False,
        average=False,
    )


if __name__ == "__main__":
    main()
