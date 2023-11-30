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
]
# DEMOGRAPHICS = ["age_band", "carehome", "diagnosis_18+", "ethnicity", "imd", "region", "sex"]

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
        except Exception as e:
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
        raise ConvergenceWarning("Failed to converge")
    return model_errors


# Is used in get_models
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


# NOTE: used in get_models
def bool_to_category(subset, group):
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


#####################
# Plotting functions
#####################


def plot_group(measure_table, pattern, group, rr=False):
    """
    Given a measures pattern, create a panel plot for each unique group
    Option to display either the counter factual plot, or the relative risk
    plot
    """
    models = get_models(measure_table, pattern, group)
    total_rows = (len(models)) // 2

    fig = plt.figure(figsize=(22, 4 * total_rows), dpi=150)

    for index, model in enumerate(models):
        category = model[1][group].iloc[0]
        ax = add_subplot(
            fig,
            (total_rows, 2, index + 1),
            [model],
            group=group,
            rr=rr,
            title=f"{category.title()}",
        )
        if index < (len(models) - 2):
            ax.set_xticklabels([])
    fig.legend(*ax.get_legend_handles_labels(), fontsize="x-small")
    fig.supylabel("Rate per 1,000 registered patients")
    plt.savefig(f"{pattern}_fig.png")


def add_subplot(
    fig,
    pos,
    models,
    group=None,
    rr=False,
    other_ax=None,
    title=None,
    ylabel=None,
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
            if rr:
                display_rr(model, df, ax)
            else:
                plot_cf(model, df, ax)
        # Plot line marking intervention
        ax.axvline(
            x=STEP_TIME_1, linestyle="--", color="blue", label="Lockdown"
        )
        ax.axvline(
            x=STEP_TIME_2, linestyle="--", color="green", label="Recovery"
        )
        ax.set_title(title, fontsize="x-small")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize="x-small")
        ax.legend(fontsize="x-small")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="x-small")
    return ax


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
    """
    Translate a dataframe with coef, lci, uci into a CI string
    Either as percent change or RR
    If a coef is 0 or 1 (for pcnt, RR), then it is the ref group
    If a coef is nan, display as "-"
    """
    if pcnt:
        df = df.apply(
            lambda x: 100 * (numpy.exp(x) - 1) if is_numeric_dtype(x) else x,
            axis=0,
        )
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'-': <15}"
            if x.coef == 0
            else f"{x.coef:.2f}% ({x.lci:.2f}% to {x.uci:.2f}%)",
            axis=1,
        )
    else:
        df = df.apply(
            lambda x: numpy.exp(x) if is_numeric_dtype(x) else x, axis=0
        )
        label = df.apply(
            lambda x: "-"
            if x.coef != x.coef
            else f"{'Ref': <27}"
            if x.coef == 1
            else f"{x.coef:.2f} ({x.lci:.2f} to {x.uci:.2f})",
            axis=1,
        )
    df["label"] = label
    return df


def pcnt_change(
    measure_table,
    pattern,
    group,
    reference=None,
    interaction=False,
):
    """
    Format a model for a forest plot
    """
    category_name = f"category_{group.split('_')[-1]}"
    models = get_models(measure_table, pattern, group, reference, interaction)
    # Interaction
    # TODO: here too handle no model
    if interaction and len(models) == 1:
        model, data = models[0]
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

    else:
        all_data = {}
        for model, its_data in models:
            subgroup = its_data[group].iloc[0]
            if model:
                subgroup_cis = get_ci_df(model)
            else:
                subgroup_cis = pandas.DataFrame(dtype="float64")
            all_data[subgroup] = subgroup_cis
    # Ensure that every model has the same set of indices
    # Even if we do not know the list before
    # TODO: This will re-order the levels- check that they are correct and consistent
    full_index = set(pandas.concat(all_data).index.get_level_values(1))
    for key, value in all_data.items():
        all_data[key] = value.reindex(full_index)
    df = pandas.concat(all_data)
    df["category"] = models[0][1][category_name].iloc[0]
    return df


def group_forest(df, as_pcnt=[], as_rr=[], mapping=None):
    """
    Create a forest plot with a column for each study period, and a row for
    each subgroup.

    """
    frames = []
    if as_rr:
        rr = get_ci_label(df[df.index.isin(as_rr, level=0)], pcnt=False)
        frames.append(rr)

    if as_pcnt:
        pcnt = get_ci_label(df[df.index.isin(as_pcnt, level=0)], pcnt=True)
        frames.append(pcnt)

    df = pandas.concat(frames)

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
        ax.axvline(x=ax_line, linewidth=0.8, linestyle="--", color="black")
        ax.set_xlabel(x_label, fontsize=8)
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


def compute_gm(model, df):
    RR, df, vcov = compute_rr(model, df)
    gm = expanding_gmean_log(RR.RR)
    gm.name = "coef"
    geom_se = df.apply(compute_coef, vcov=vcov, axis=1)
    GM = pandas.DataFrame(gm)
    GM["lci"] = GM["coef"] - 1.96 * geom_se
    GM["uci"] = GM["coef"] + 1.96 * geom_se
    GM["error"] = geom_se
    return GM


def compute_rr(model, df):
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
    return (RR, df, vcov)


def display_rr(model, df, ax):
    RR, _, _ = compute_rr(model, df)

    plt.vlines(RR.index, RR.lci, RR.uci, color="k")
    ax.plot(RR.index, RR.RR, color="k")
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.set_ylabel(
        "Relative Risk of Antidepressant Prescribing (95% CI)\nCompared to no COVID-19 counterfactual"
    )
    gm = get_ci_label(compute_gm(model, df), pcnt=False).iloc[-1].label
    ax.text(
        0.8,
        0.8,
        "Geometric mean\nof RR over\ntime period:\n" + gm,
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "red", "alpha": 0.5},
    )


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


#######################
# Supplemental
#######################
def output_acf_pacf(measure_table, output_dir):
    residuals_dir = output_dir / "residuals"
    residuals_dir.mkdir(exist_ok=True)
    model_all = get_models(
        measure_table, "antidepressant_any_all_total_rate", "group_0"
    )[0][0]
    check_residuals(model_all, residuals_dir, "model_all_noerr")


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


def figure_2(measure_table, output_dir, rr=False):
    # Figure 1
    fig = plt.figure(figsize=(16, 8), dpi=150)

    models_aut = get_models(
        measure_table,
        "antidepressant_any_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    add_subplot(
        fig,
        (2, 2, 1),
        models_aut,
        group="group_0",
        rr=rr,
        ylabel="Rate per 1,000 autism patients",
        title="Antidepressant Prescribing Autism",
    )
    models_ld = get_models(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    add_subplot(
        fig,
        (2, 2, 2),
        models_ld,
        rr=rr,
        group="group_0",
        ylabel="Rate per 1,000 LD patients",
        title="Antidepressant Prescribing Learning Disability",
    )

    models_aut_new = get_models(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    add_subplot(
        fig,
        (2, 2, 3),
        models_aut_new,
        group="group_0",
        rr=rr,
        ylabel="Rate per 1,000 AD naive autism patients",
        title="New Antidepressant Prescribing Autism",
    )
    models_ld_new = get_models(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    add_subplot(
        fig,
        (2, 2, 4),
        models_ld_new,
        group="group_0",
        rr=rr,
        ylabel="Rate per 1,000 AD naive LD patients",
        title="New Antidepressant Prescribing Learning Disability",
    )

    plt.savefig(output_dir / "figure_2.png")


def forest(measure_table, output_dir):
    # Forest plot
    model_aut = pcnt_change(
        measure_table,
        "antidepressant_any_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    model_ld = pcnt_change(
        measure_table,
        "antidepressant_any_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    model_new_aut = pcnt_change(
        measure_table,
        "antidepressant_any_new_autism_total_rate",
        group="group_0",
        reference="Recorded autism",
    )
    model_new_ld = pcnt_change(
        measure_table,
        "antidepressant_any_new_learning_disability_total_rate",
        group="group_0",
        reference="Recorded learning_disability",
    )
    new = pandas.concat([model_new_aut, model_new_ld]).reset_index()
    new.group = new.group + " new"
    new = new.set_index(["change", "group"])
    df = pandas.concat([model_aut, model_ld, new])

    group_forest(
        df,
        as_pcnt=["slope", "slope2", "step", "step2"],
        as_rr=["baseline"],
        mapping=MAPPING,
    )
    plt.savefig(output_dir / "figure_3.png")


# TODO: we could make a loop instead of doing each individually
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
    new_coef = translate_to_ci(get_ci_df(model_new), "New prescribing")
    aut_coef = translate_to_ci(get_ci_df(model_aut), "Autism prescribing")
    aut_new_coef = translate_to_ci(
        get_ci_df(model_aut_new), "Autism new prescribing"
    )
    ld_coef = translate_to_ci(get_ci_df(model_ld), "LD prescribing")
    ld_new_coef = translate_to_ci(
        get_ci_df(model_ld_new), "LD new prescribing"
    )
    table = pandas.concat(
        [all_coef, new_coef, aut_coef, aut_new_coef, ld_coef, ld_new_coef]
    )
    table.to_html(output_dir / "table3.html")


def table_geometric_mean(
    measure_table, pattern, label, group=None, reference=None
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
            coefs = compute_gm(model, data)
            row = coefs.iloc[-1]
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


def forest_geometric_mean(measure_table, output_dir, new=False):
    results = []
    if new:
        new_string = "new_"
    else:
        new_string = ""
    for demo in DEMOGRAPHICS:
        extension = f"breakdown_{demo}_rate"
        group = "group_1"
        if demo == "total":
            extension = "total_rate"
            group = "group_0"
        elif demo == "prescription":
            extension = f"breakdown_{demo}_count"
        results.append(
            table_geometric_mean(
                measure_table,
                f"antidepressant_any_{new_string}all_{extension}",
                "All prescribing",
                "group_0",
            )
        )
        results.append(
            table_geometric_mean(
                measure_table,
                f"antidepressant_any_{new_string}autism_{extension}",
                "Autism",
                group,
                reference="Recorded autism" if group == "group_0" else None,
            )
        )
        results.append(
            table_geometric_mean(
                measure_table,
                f"antidepressant_any_{new_string}learning_disability_{extension}",
                "Learning Disability",
                group,
                reference="Recorded learning_disability"
                if group == "group_0"
                else None,
            )
        )
    df = pandas.concat(results)
    group_forest(
        df, as_rr=["All prescribing", "Autism", "Learning Disability"]
    )


# TODO: HANDLE MODEL FAILURES (LIKE GM)
def forest_pcnt_change(measure_table, output_dir, population="all", new=False):
    results = []
    if new:
        new_string = "new_"
    else:
        new_string = ""
    for demo in DEMOGRAPHICS:
        print(demo)
        extension = f"breakdown_{demo}_rate"
        group = "group_1"
        if demo == "total":
            extension = "total_rate"
            group = "group_0"
        elif demo == "prescription":
            extension = f"breakdown_{demo}_count"
        results.append(
            pcnt_change(
                measure_table,
                f"antidepressant_any_{new_string}{population}_{extension}",
                "group_0" if population == "all" else group,
                reference=f"Recorded {population}"
                if group == "group_0" and population != "all"
                else None,
            )
        )
    df = pandas.concat(results)
    df.index = pandas.MultiIndex.from_tuples(
        list(zip(df.index.get_level_values(1), df.index.get_level_values(0)))
    )
    # TODO: should we have category as index and column?
    df.index.names = ["change", "group"]
    # df = df.reset_index().set_index(["change", "category"])
    group_forest(
        df, as_pcnt=["slope", "slope2", "step", "step2"], mapping=MAPPING
    )
    plt.savefig(output_dir / f"forest_{population}{new_string}.png")


def plot_all_cf(measure_table, output_dir, rr=False):
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
        rr=rr,
        title="Any Antidepressant",
    )

    models_new = get_models(
        measure_table, "antidepressant_any_new_all_total_rate", "group_0"
    )
    add_subplot(
        fig,
        (2, 1, 2),
        models_new,
        rr=rr,
        title="New Antidepressant",
    )
    fig.legend(*ax.get_legend_handles_labels(), fontsize="x-small")
    plt.savefig(output_dir / "cf.png")


# NOTE: this function is used
def plot_any_breakdowns(measure_table, output_dir, new=False):
    fig = plt.figure(figsize=(14, 12), dpi=150, constrained_layout=True)
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
        (4, 2, 1),
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
        (4, 2, 2),
        models_carehome,
        group="group_0",
        title="Carehome",
        ylabel="Rate per 1,000 registered patients",
    )
    models_diagnosis = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_diagnosis_18+_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (4, 2, 3),
        models_diagnosis,
        group="group_0",
        title="Diagnosis",
        ylabel="Rate per 1,000 registered patients",
    )
    models_ethnicity = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_ethnicity_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (4, 2, 4),
        models_ethnicity,
        group="group_0",
        title="Ethnicity",
        ylabel="Rate per 1,000 registered patients",
    )
    models_imd = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_imd_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (4, 2, 5),
        models_imd,
        group="group_0",
        title="IMD",
        ylabel="Rate per 1,000 registered patients",
    )
    models_region = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_region_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (4, 2, 6),
        models_region,
        group="group_0",
        title="Region",
        ylabel="Rate per 1,000 registered patients",
    )
    models_sex = get_models(
        measure_table,
        f"antidepressant_any_{new_string}all_breakdown_sex_rate",
        group="group_0",
    )
    add_subplot(
        fig,
        (4, 2, 7),
        models_sex,
        group="group_0",
        title="Sex",
        ylabel="Rate per 1,000 registered patients",
    )
    if not new:
        models_prescription = get_models(
            measure_table,
            f"antidepressant_any_{new_string}all_breakdown_prescription_count",
            group="group_0",
        )
        add_subplot(
            fig,
            (4, 2, 8),
            models_prescription,
            group="group_0",
            title="Prescription",
            ylabel="Rate per 1,000 registered patients",
        )

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

    forest_pcnt_change(
        measure_table, output_dir, population="autism", new=False
    )
    plt.savefig("forest_pcnt.png")
    import code

    code.interact(local=locals())
    # out = pcnt_change(
    #    measure_table,
    #    "antidepressant_any_all_breakdown_age_band_rate",
    #    "group_0",
    # )

    # Figure 1
    # CF plot for all, new
    # plot_all_cf(measure_table, output_dir, rr=True)

    # Figure 2
    # CF plot for autism/LD
    # figure_2(measure_table, output_dir, rr=True)

    # Table 3
    # Table of coefficients for each of the models
    # table_any_new(measure_table, output_dir)

    # Fitted plot by demographic group
    # Can specify new/ not new
    # Can modify the function to use interaction
    # plot_any_breakdowns(measure_table, output_dir, new=False)

    # Last figure
    # Forest plot of GM for all, aut, ld; overall and by breakdown
    # forest_geometric_mean(measure_table, output_dir, new=True)
    # plt.savefig("forest_gm.png")

    # output_acf_pacf(measure_table, output_dir)


if __name__ == "__main__":
    main()
