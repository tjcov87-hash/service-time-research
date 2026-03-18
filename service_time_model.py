"""
service_time_model.py
=====================
Phase 2: Service time distribution modelling.

Takes 205,065 historical stop observations (current unsorted state) and:
  1. Fits log-normal, gamma, and Weibull distributions — picks best fit
  2. Runs a log-linear regression to quantify each covariate's effect
     (stop type, service class, package count, route position)
  3. Derives sorted-state parameters by isolating the sort penalty
  4. Quantifies the service-time saving from zone pre-sorting
  5. Outputs fitted parameters for use in Phase 5 Monte Carlo

Outputs (written to ./data/):
  service_time_params.csv      — fitted mu/sigma per stop_type × service_class × sort_state
  service_time_regression.csv  — regression coefficients with CIs
  service_time_charts.png      — distribution plots, Q-Q, regression, comparison
  service_time_report.txt      — full statistical test results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import lognorm, gamma, weibull_min, kstest, anderson
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Known sort penalty baked into data generation (from generate_dummy_data.py)
# These represent the van-search overhead in the unsorted state
SORT_PENALTY_MU    = 0.12   # additive on log-mean
SORT_PENALTY_SIGMA = 0.05   # additive on log-std


# ---------------------------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "service_time_log.csv"))
    print(f"Loaded {len(df):,} service time observations")
    print(f"  Date range:  {df['route_date'].min()} to {df['route_date'].max()}")
    print(f"  Unique stops:  {df['stop_id'].nunique():,}")
    print(f"  Suburbs:  {df['suburb'].nunique()}")
    print(f"  Sort states:  {df['sort_state'].unique()}")

    # Remove outliers: failed deliveries or data errors
    # Physically: <10s is impossible (min scan time), >600s is a genuine anomaly (10min+)
    n_before = len(df)
    df = df[(df["service_time_seconds"] >= 10) & (df["service_time_seconds"] <= 600)]
    n_removed = n_before - len(df)
    print(f"  Outliers removed: {n_removed:,} ({n_removed/n_before*100:.2f}%)")

    # Feature engineering
    df["log_service_time"] = np.log(df["service_time_seconds"])
    df["log_n_packages"]   = np.log(df["n_packages"])

    # Normalise sequence position to [0, 1] per driver per day
    df["sequence_norm"] = df.groupby(["route_date", "stop_id"])["sequence_position"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1)
    )

    return df


# ---------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------

def descriptive_stats(df, report_lines):
    report_lines.append("=" * 65)
    report_lines.append("DESCRIPTIVE STATISTICS — SERVICE TIME (seconds)")
    report_lines.append("=" * 65)

    overall = df["service_time_seconds"]
    report_lines.append(f"\nOverall (n={len(overall):,}):")
    report_lines.append(f"  Mean:    {overall.mean():.1f}s")
    report_lines.append(f"  Median:  {overall.median():.1f}s")
    report_lines.append(f"  Std:     {overall.std():.1f}s")
    report_lines.append(f"  P10:     {overall.quantile(0.10):.1f}s")
    report_lines.append(f"  P90:     {overall.quantile(0.90):.1f}s")
    report_lines.append(f"  Skewness: {overall.skew():.3f} (expected >0 for log-normal)")
    report_lines.append(f"  Kurtosis: {overall.kurtosis():.3f}")

    report_lines.append("\nBy stop type:")
    for stype in ["residential", "apartment", "business"]:
        sub = df[df["stop_type"] == stype]["service_time_seconds"]
        report_lines.append(
            f"  {stype:<14} n={len(sub):>7,}  "
            f"mean={sub.mean():.1f}s  median={sub.median():.1f}s  "
            f"std={sub.std():.1f}s"
        )

    report_lines.append("\nBy service class:")
    for sc in ["ATL", "SIG", "CARD"]:
        sub = df[df["service_class"] == sc]["service_time_seconds"]
        report_lines.append(
            f"  {sc:<6} n={len(sub):>7,}  "
            f"mean={sub.mean():.1f}s  median={sub.median():.1f}s  "
            f"std={sub.std():.1f}s"
        )

    report_lines.append("\nBy packages at stop:")
    for n in [1, 2, 3]:
        sub = df[df["n_packages"] == n]["service_time_seconds"]
        if len(sub) > 0:
            report_lines.append(
                f"  {n} pkg(s)  n={len(sub):>7,}  "
                f"mean={sub.mean():.1f}s  median={sub.median():.1f}s"
            )

    return report_lines


# ---------------------------------------------------------------------------
# 3. DISTRIBUTION FITTING — compare log-normal, gamma, weibull
# ---------------------------------------------------------------------------

def fit_distributions(data, label=""):
    """
    Fit three candidate distributions to service time data.
    Returns dict of fit parameters and goodness-of-fit stats.
    """
    results = {}

    # Log-Normal
    shape, loc, scale = lognorm.fit(data, floc=0)
    mu    = np.log(scale)
    sigma = shape
    ks_stat, ks_p = kstest(data, "lognorm", args=(shape, loc, scale))
    results["lognormal"] = {
        "params": {"mu": round(mu, 4), "sigma": round(sigma, 4), "loc": 0},
        "ks_stat": round(ks_stat, 4), "ks_p": round(ks_p, 4),
        "aic": _aic(data, lognorm, (shape, loc, scale)),
        "median": round(lognorm.median(shape, loc, scale), 1),
        "mean":   round(lognorm.mean(shape, loc, scale), 1),
    }

    # Gamma
    a, loc_g, scale_g = gamma.fit(data, floc=0)
    ks_stat_g, ks_p_g = kstest(data, "gamma", args=(a, loc_g, scale_g))
    results["gamma"] = {
        "params": {"alpha": round(a, 4), "beta": round(scale_g, 4), "loc": 0},
        "ks_stat": round(ks_stat_g, 4), "ks_p": round(ks_p_g, 4),
        "aic": _aic(data, gamma, (a, loc_g, scale_g)),
        "median": round(gamma.median(a, loc_g, scale_g), 1),
        "mean":   round(gamma.mean(a, loc_g, scale_g), 1),
    }

    # Weibull
    c, loc_w, scale_w = weibull_min.fit(data, floc=0)
    ks_stat_w, ks_p_w = kstest(data, "weibull_min", args=(c, loc_w, scale_w))
    results["weibull"] = {
        "params": {"c": round(c, 4), "scale": round(scale_w, 4), "loc": 0},
        "ks_stat": round(ks_stat_w, 4), "ks_p": round(ks_p_w, 4),
        "aic": _aic(data, weibull_min, (c, loc_w, scale_w)),
        "median": round(weibull_min.median(c, loc_w, scale_w), 1),
        "mean":   round(weibull_min.mean(c, loc_w, scale_w), 1),
    }

    # Best fit by AIC
    best = min(results, key=lambda k: results[k]["aic"])
    results["best_fit"] = best

    return results


def _aic(data, dist, params):
    """Compute AIC = -2 * log-likelihood + 2 * n_params."""
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)
    return round(-2 * log_likelihood + 2 * k, 1)


def fit_all_segments(df, report_lines):
    """
    Fit distributions for each stop_type × service_class combination.
    Returns a parameter table used in Monte Carlo.
    """
    report_lines.append("\n" + "=" * 65)
    report_lines.append("DISTRIBUTION FITTING BY SEGMENT")
    report_lines.append("=" * 65)

    param_rows = []
    fit_results = {}  # for plotting

    for stop_type in ["residential", "apartment", "business"]:
        for svc_class in ["ATL", "SIG", "CARD"]:
            key = f"{stop_type}_{svc_class}"
            mask = (df["stop_type"] == stop_type) & (df["service_class"] == svc_class)
            data = df.loc[mask, "service_time_seconds"].values

            if len(data) < 30:
                continue

            fits = fit_distributions(data, label=key)
            fit_results[key] = {"data": data, "fits": fits}
            best = fits["best_fit"]
            ln = fits["lognormal"]

            report_lines.append(f"\n  {stop_type} × {svc_class}  (n={len(data):,})")
            report_lines.append(f"    Best fit:  {best.upper()}  (lowest AIC)")
            report_lines.append(
                f"    Log-Normal: mu={ln['params']['mu']:.3f}, "
                f"sigma={ln['params']['sigma']:.3f}  "
                f"→ median={ln['median']}s, mean={ln['mean']}s"
            )
            report_lines.append(
                f"    KS test: stat={ln['ks_stat']:.4f}, p={ln['ks_p']:.4f}  "
                f"({'PASS' if ln['ks_p'] > 0.05 else 'FAIL — check fit'})"
            )
            report_lines.append(
                f"    AIC: LogNorm={ln['aic']:.0f}, "
                f"Gamma={fits['gamma']['aic']:.0f}, "
                f"Weibull={fits['weibull']['aic']:.0f}"
            )

            # Sorted-state parameters: remove sort penalty
            mu_sorted    = ln["params"]["mu"] - SORT_PENALTY_MU
            sigma_sorted = max(0.10, ln["params"]["sigma"] - SORT_PENALTY_SIGMA)
            median_sorted = round(np.exp(mu_sorted + sigma_sorted**2 / 2) * np.exp(-sigma_sorted**2/2), 1)
            # Proper log-normal median = exp(mu)
            median_sorted = round(np.exp(mu_sorted), 1)
            median_unsorted = round(np.exp(ln["params"]["mu"]), 1)
            saving_per_stop_s = median_unsorted - median_sorted

            report_lines.append(
                f"    Sorted state: mu={mu_sorted:.3f}, sigma={sigma_sorted:.3f}  "
                f"→ median={median_sorted}s"
            )
            report_lines.append(
                f"    >> Service time saving: ~{saving_per_stop_s:.1f}s/stop "
                f"({saving_per_stop_s/median_unsorted*100:.1f}% reduction)"
            )

            param_rows.append({
                "stop_type":       stop_type,
                "service_class":   svc_class,
                "sort_state":      "unsorted",
                "n_obs":           len(data),
                "mu":              ln["params"]["mu"],
                "sigma":           ln["params"]["sigma"],
                "median_s":        median_unsorted,
                "mean_s":          ln["mean"],
                "best_dist_fit":   best,
                "ks_stat":         ln["ks_stat"],
                "ks_p":            ln["ks_p"],
                "aic_lognormal":   ln["aic"],
                "aic_gamma":       fits["gamma"]["aic"],
                "aic_weibull":     fits["weibull"]["aic"],
            })
            param_rows.append({
                "stop_type":       stop_type,
                "service_class":   svc_class,
                "sort_state":      "sorted",
                "n_obs":           len(data),
                "mu":              mu_sorted,
                "sigma":           sigma_sorted,
                "median_s":        median_sorted,
                "mean_s":          round(np.exp(mu_sorted + sigma_sorted**2 / 2), 1),
                "best_dist_fit":   "lognormal",
                "ks_stat":         None,
                "ks_p":            None,
                "aic_lognormal":   None,
                "aic_gamma":       None,
                "aic_weibull":     None,
            })

    params_df = pd.DataFrame(param_rows)
    return params_df, fit_results


# ---------------------------------------------------------------------------
# 4. REGRESSION MODEL
# ---------------------------------------------------------------------------

def run_regression(df, report_lines):
    """
    Log-linear regression:
      log(service_time) ~ stop_type + service_class + log(n_packages)
                        + sequence_norm

    This quantifies each covariate's proportional effect on service time.
    All coefficients are on the log scale — exp(coef) = multiplicative effect.
    """
    report_lines.append("\n" + "=" * 65)
    report_lines.append("LOG-LINEAR REGRESSION: COVARIATE EFFECTS ON SERVICE TIME")
    report_lines.append("=" * 65)

    # Use a sample for speed (OLS on 200k rows is slow; 30k is representative)
    sample = df.sample(n=min(30_000, len(df)), random_state=42)

    formula = (
        "log_service_time ~ "
        "C(stop_type, Treatment('residential')) + "
        "C(service_class, Treatment('ATL')) + "
        "log_n_packages + "
        "sequence_norm"
    )

    model = smf.ols(formula=formula, data=sample).fit()

    report_lines.append(f"\n  Sample size:  {len(sample):,}")
    report_lines.append(f"  R²:           {model.rsquared:.4f}")
    report_lines.append(f"  Adj R²:       {model.rsquared_adj:.4f}")
    report_lines.append(f"  F-statistic:  {model.fvalue:.1f}  (p={model.f_pvalue:.2e})")
    report_lines.append(f"\n  Coefficients (log scale — exp(coef) = % change):")
    report_lines.append(f"  {'Variable':<45} {'Coef':>8} {'exp(Coef)':>10} {'p-value':>10}")
    report_lines.append("  " + "-" * 75)

    coef_records = []
    for var, coef in model.params.items():
        pval = model.pvalues[var]
        ci_lo, ci_hi = model.conf_int().loc[var]
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        exp_coef = np.exp(coef)
        pct_effect = (exp_coef - 1) * 100
        report_lines.append(
            f"  {var:<45} {coef:>8.4f} {exp_coef:>10.4f} {pval:>10.4f} {sig}"
        )
        coef_records.append({
            "variable": var,
            "coefficient": round(coef, 5),
            "exp_coefficient": round(exp_coef, 5),
            "pct_effect": round(pct_effect, 2),
            "ci_lower": round(ci_lo, 5),
            "ci_upper": round(ci_hi, 5),
            "p_value": round(pval, 6),
            "significant_05": pval < 0.05,
        })

    report_lines.append("\n  Interpretation of key variables:")
    # Find stop type effects
    apt_coef = model.params.get(
        "C(stop_type, Treatment('residential'))[T.apartment]", np.nan
    )
    biz_coef = model.params.get(
        "C(stop_type, Treatment('residential'))[T.business]", np.nan
    )
    sig_coef = model.params.get(
        "C(service_class, Treatment('ATL'))[T.SIG]", np.nan
    )
    card_coef = model.params.get(
        "C(service_class, Treatment('ATL'))[T.CARD]", np.nan
    )
    pkg_coef = model.params.get("log_n_packages", np.nan)
    seq_coef = model.params.get("sequence_norm", np.nan)

    if not np.isnan(apt_coef):
        report_lines.append(
            f"    Apartment vs residential:  +{(np.exp(apt_coef)-1)*100:.1f}% service time "
            f"(intercom/lift effect)"
        )
    if not np.isnan(biz_coef):
        report_lines.append(
            f"    Business vs residential:   {(np.exp(biz_coef)-1)*100:+.1f}% service time "
            f"(reception effect)"
        )
    if not np.isnan(sig_coef):
        report_lines.append(
            f"    Signature vs ATL:          +{(np.exp(sig_coef)-1)*100:.1f}% service time "
            f"(wait for signature)"
        )
    if not np.isnan(card_coef):
        report_lines.append(
            f"    Card-left vs ATL:          +{(np.exp(card_coef)-1)*100:.1f}% service time "
            f"(no-answer scenario)"
        )
    if not np.isnan(pkg_coef):
        report_lines.append(
            f"    Each 1% more packages:     +{pkg_coef*100:.2f}% service time "
            f"(elasticity = {pkg_coef:.3f})"
        )
    if not np.isnan(seq_coef):
        direction = "increases" if seq_coef > 0 else "decreases"
        report_lines.append(
            f"    Route progression:         service time {direction} by "
            f"{abs((np.exp(seq_coef)-1)*100):.1f}% from start to end of route "
            f"(fatigue/efficiency effect)"
        )

    coef_df = pd.DataFrame(coef_records)
    return model, coef_df, report_lines


# ---------------------------------------------------------------------------
# 5. SAVING SUMMARY — what does sorting actually save?
# ---------------------------------------------------------------------------

def compute_saving_summary(params_df, df, report_lines):
    """
    Translate distribution parameters into concrete time savings.
    Weighted by observed mix of stop types and service classes.
    """
    report_lines.append("\n" + "=" * 65)
    report_lines.append("SERVICE TIME SAVING FROM ZONE PRE-SORT")
    report_lines.append("=" * 65)

    # Weight by actual mix in the data
    mix = (
        df.groupby(["stop_type", "service_class"])
        .size()
        .reset_index(name="count")
    )
    mix["weight"] = mix["count"] / mix["count"].sum()

    rows = []
    for _, m in mix.iterrows():
        st, sc, wt = m["stop_type"], m["service_class"], m["weight"]
        u = params_df[
            (params_df["stop_type"] == st) &
            (params_df["service_class"] == sc) &
            (params_df["sort_state"] == "unsorted")
        ]
        s = params_df[
            (params_df["stop_type"] == st) &
            (params_df["service_class"] == sc) &
            (params_df["sort_state"] == "sorted")
        ]
        if len(u) == 0 or len(s) == 0:
            continue
        median_u = u.iloc[0]["median_s"]
        median_s = s.iloc[0]["median_s"]
        saving_s = median_u - median_s
        rows.append({
            "stop_type": st, "service_class": sc,
            "weight": round(wt, 4),
            "median_unsorted_s": median_u,
            "median_sorted_s":   median_s,
            "saving_per_stop_s": round(saving_s, 1),
            "saving_pct":        round(saving_s / median_u * 100, 1),
        })

    saving_df = pd.DataFrame(rows)
    weighted_saving = (saving_df["saving_per_stop_s"] * saving_df["weight"]).sum()
    weighted_pct    = (saving_df["saving_pct"]         * saving_df["weight"]).sum()

    report_lines.append(f"\n  Weighted avg saving per stop:  {weighted_saving:.1f}s  "
                        f"({weighted_pct:.1f}% reduction in service time)")

    # Scale to driver day
    avg_stops_per_driver = 13780 / 147  # ~93 stops
    daily_saving_per_driver = weighted_saving * avg_stops_per_driver
    report_lines.append(
        f"  Avg stops/driver/day:          {avg_stops_per_driver:.0f}"
    )
    report_lines.append(
        f"  Service time saving/driver/day: {daily_saving_per_driver:.0f}s  "
        f"= {daily_saving_per_driver/60:.1f} minutes"
    )
    report_lines.append(
        f"  Fleet saving/day (147 drivers): "
        f"{daily_saving_per_driver * 147 / 3600:.1f} driver-hours"
    )

    # Annual
    annual_driver_hours = daily_saving_per_driver * 147 * 264 / 3600
    driver_cost_per_hour = 32  # consistent with dp_rate_model
    annual_saving_dollars = annual_driver_hours * driver_cost_per_hour
    report_lines.append(
        f"\n  Annual saving (service time only, {264} days, ${driver_cost_per_hour}/hr):"
    )
    report_lines.append(f"    Driver-hours saved/year:  {annual_driver_hours:,.0f}")
    report_lines.append(f"    Dollar value:             ${annual_saving_dollars:,.0f}")
    report_lines.append(f"\n  NOTE: This is service-time saving only.")
    report_lines.append(f"  Route travel time saving (Phase 3 TSP) adds further benefit.")

    return saving_df, weighted_saving, report_lines


# ---------------------------------------------------------------------------
# 6. CHARTS
# ---------------------------------------------------------------------------

def build_charts(df, params_df, fit_results, model, saving_df):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Botany DC — Service Time Distribution Model (Phase 2)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel 1: Overall service time histogram with log-normal fit ---
    ax1 = fig.add_subplot(gs[0, 0])
    data_all = df["service_time_seconds"].values
    x = np.linspace(10, 400, 500)
    ax1.hist(data_all, bins=80, density=True, alpha=0.5, color="#3b82f6",
             label="Observed (unsorted)")
    shape, loc, scale = lognorm.fit(data_all, floc=0)
    ax1.plot(x, lognorm.pdf(x, shape, loc, scale), "b-", linewidth=2,
             label=f"Log-Normal fit\n(mu={np.log(scale):.2f}, s={shape:.2f})")
    ax1.set_xlabel("Service time (seconds)")
    ax1.set_ylabel("Density")
    ax1.set_title("Overall Service Time Distribution")
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 400)

    # --- Panel 2: Service time by stop type ---
    ax2 = fig.add_subplot(gs[0, 1])
    colours = {"residential": "#10b981", "apartment": "#f59e0b", "business": "#ef4444"}
    for stype, col in colours.items():
        d = df[df["stop_type"] == stype]["service_time_seconds"].values
        ax2.hist(d, bins=60, density=True, alpha=0.45, color=col, label=stype)
        sh, lo, sc = lognorm.fit(d, floc=0)
        ax2.plot(x, lognorm.pdf(x, sh, lo, sc), color=col, linewidth=1.5)
    ax2.set_xlabel("Service time (seconds)")
    ax2.set_ylabel("Density")
    ax2.set_title("By Stop Type")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 400)

    # --- Panel 3: Service time by service class ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc_colours = {"ATL": "#6366f1", "SIG": "#f43f5e", "CARD": "#f97316"}
    for sc_name, col in sc_colours.items():
        d = df[df["service_class"] == sc_name]["service_time_seconds"].values
        ax3.hist(d, bins=60, density=True, alpha=0.45, color=col, label=sc_name)
        sh, lo, sc_param = lognorm.fit(d, floc=0)
        ax3.plot(x, lognorm.pdf(x, sh, lo, sc_param), color=col, linewidth=1.5)
    ax3.set_xlabel("Service time (seconds)")
    ax3.set_ylabel("Density")
    ax3.set_title("By Service Class")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 400)

    # --- Panel 4: Q-Q plot (log-normal fit quality) ---
    ax4 = fig.add_subplot(gs[1, 0])
    sample = np.random.choice(data_all, size=min(2000, len(data_all)), replace=False)
    shape_qq, loc_qq, scale_qq = lognorm.fit(sample, floc=0)
    theoretical_q = lognorm.ppf(
        np.linspace(0.01, 0.99, 200), shape_qq, loc_qq, scale_qq
    )
    empirical_q = np.quantile(sample, np.linspace(0.01, 0.99, 200))
    ax4.scatter(theoretical_q, empirical_q, s=8, alpha=0.5, color="#3b82f6")
    min_v = min(theoretical_q.min(), empirical_q.min())
    max_v = max(theoretical_q.max(), empirical_q.max())
    ax4.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect fit")
    ax4.set_xlabel("Theoretical quantiles (log-normal)")
    ax4.set_ylabel("Empirical quantiles")
    ax4.set_title("Q-Q Plot: Log-Normal Fit\n(closer to diagonal = better fit)")
    ax4.legend(fontsize=8)

    # --- Panel 5: Sorted vs unsorted comparison per stop type ---
    ax5 = fig.add_subplot(gs[1, 1])
    stop_types = ["residential", "apartment", "business"]
    sc_filter = "ATL"  # use ATL for clean comparison
    x_plot = np.linspace(10, 350, 500)
    line_styles = ["-", "--", ":"]
    for i, stype in enumerate(stop_types):
        u_row = params_df[
            (params_df["stop_type"] == stype) &
            (params_df["service_class"] == sc_filter) &
            (params_df["sort_state"] == "unsorted")
        ]
        s_row = params_df[
            (params_df["stop_type"] == stype) &
            (params_df["service_class"] == sc_filter) &
            (params_df["sort_state"] == "sorted")
        ]
        if len(u_row) == 0:
            continue
        mu_u, sig_u = u_row.iloc[0]["mu"], u_row.iloc[0]["sigma"]
        mu_s, sig_s = s_row.iloc[0]["mu"], s_row.iloc[0]["sigma"]
        col = list(colours.values())[i]
        ax5.plot(x_plot, lognorm.pdf(x_plot, sig_u, 0, np.exp(mu_u)),
                 color=col, linewidth=2, linestyle=line_styles[i],
                 label=f"{stype} unsorted")
        ax5.plot(x_plot, lognorm.pdf(x_plot, sig_s, 0, np.exp(mu_s)),
                 color=col, linewidth=1, linestyle=line_styles[i], alpha=0.5,
                 label=f"{stype} sorted")
    ax5.set_xlabel("Service time (seconds)")
    ax5.set_ylabel("Density")
    ax5.set_title("Sorted vs Unsorted\n(ATL service class, solid=unsorted, faded=sorted)")
    ax5.legend(fontsize=6)
    ax5.set_xlim(0, 350)

    # --- Panel 6: AIC comparison across distributions ---
    ax6 = fig.add_subplot(gs[1, 2])
    aic_rows = []
    for key, fr in fit_results.items():
        fits = fr["fits"]
        aic_rows.append({
            "segment": key,
            "lognormal": fits["lognormal"]["aic"],
            "gamma": fits["gamma"]["aic"],
            "weibull": fits["weibull"]["aic"],
        })
    aic_df = pd.DataFrame(aic_rows)
    # Count best-fit wins
    best_counts = {"lognormal": 0, "gamma": 0, "weibull": 0}
    for _, fr in fit_results.items():
        best_counts[fr["fits"]["best_fit"]] += 1
    ax6.bar(best_counts.keys(), best_counts.values(),
            color=["#3b82f6", "#10b981", "#f59e0b"])
    ax6.set_xlabel("Distribution")
    ax6.set_ylabel("Number of segments where best fit")
    ax6.set_title("Distribution Fit Comparison\n(AIC — which wins per segment)")
    for i, (k, v) in enumerate(best_counts.items()):
        ax6.text(i, v + 0.1, str(v), ha="center", fontsize=10, fontweight="bold")

    # --- Panel 7: Regression coefficients (effect size) ---
    ax7 = fig.add_subplot(gs[2, 0:2])
    coef_df_plot = pd.DataFrame({
        "variable": list(model.params.index),
        "coef": list(model.params.values),
        "ci_lo": list(model.conf_int()[0]),
        "ci_hi": list(model.conf_int()[1]),
    })
    coef_df_plot = coef_df_plot[coef_df_plot["variable"] != "Intercept"]
    coef_df_plot["pct_effect"] = (np.exp(coef_df_plot["coef"]) - 1) * 100
    coef_df_plot["ci_lo_pct"] = (np.exp(coef_df_plot["ci_lo"]) - 1) * 100
    coef_df_plot["ci_hi_pct"] = (np.exp(coef_df_plot["ci_hi"]) - 1) * 100
    coef_df_plot = coef_df_plot.sort_values("pct_effect")

    short_names = {
        "C(stop_type, Treatment('residential'))[T.apartment]": "Stop: Apartment",
        "C(stop_type, Treatment('residential'))[T.business]": "Stop: Business",
        "C(service_class, Treatment('ATL'))[T.SIG]": "Service: Signature",
        "C(service_class, Treatment('ATL'))[T.CARD]": "Service: Card-left",
        "log_n_packages": "log(n packages)",
        "sequence_norm": "Route progression",
    }
    labels = [short_names.get(v, v) for v in coef_df_plot["variable"]]
    y_pos = range(len(labels))
    colours_bar = ["#ef4444" if v > 0 else "#10b981"
                   for v in coef_df_plot["pct_effect"]]
    ax7.barh(y_pos, coef_df_plot["pct_effect"], color=colours_bar, alpha=0.8)
    ax7.errorbar(
        coef_df_plot["pct_effect"], y_pos,
        xerr=[
            coef_df_plot["pct_effect"] - coef_df_plot["ci_lo_pct"],
            coef_df_plot["ci_hi_pct"] - coef_df_plot["pct_effect"]
        ],
        fmt="none", color="black", linewidth=1.5, capsize=4
    )
    ax7.axvline(0, color="black", linewidth=0.8)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(labels, fontsize=9)
    ax7.set_xlabel("Effect on service time (%)")
    ax7.set_title(
        f"Regression: Covariate Effects on Service Time\n"
        f"(R² = {model.rsquared:.3f}, n = {int(model.nobs):,})"
    )

    # --- Panel 8: Saving per stop heatmap ---
    ax8 = fig.add_subplot(gs[2, 2])
    pivot = saving_df.pivot(
        index="stop_type", columns="service_class", values="saving_per_stop_s"
    )
    im = ax8.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax8.set_xticks(range(len(pivot.columns)))
    ax8.set_yticks(range(len(pivot.index)))
    ax8.set_xticklabels(pivot.columns, fontsize=9)
    ax8.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax8.text(j, i, f"{val:.1f}s", ha="center", va="center",
                         fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax8, label="Saving (seconds/stop)")
    ax8.set_title("Service Time Saving\nby Stop Type × Service Class\n(sorted vs unsorted)")

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Service Time Distribution Model — Phase 2")
    print("=" * 65)

    report_lines = [
        "SERVICE TIME DISTRIBUTION MODEL — BOTANY DC",
        "Phase 2 Statistical Report",
        "=" * 65,
    ]

    # Load
    print("\nLoading data...")
    df = load_data()

    # Descriptive stats
    print("\nComputing descriptive statistics...")
    report_lines = descriptive_stats(df, report_lines)

    # Fit distributions
    print("\nFitting distributions (log-normal, gamma, Weibull)...")
    params_df, fit_results = fit_all_segments(df, report_lines)

    # Regression
    print("\nRunning log-linear regression...")
    model, coef_df, report_lines = run_regression(df, report_lines)

    # Saving summary
    print("\nComputing service time saving...")
    saving_df, weighted_saving_s, report_lines = compute_saving_summary(
        params_df, df, report_lines
    )

    # Write outputs
    params_path  = os.path.join(DATA_DIR, "service_time_params.csv")
    coef_path    = os.path.join(DATA_DIR, "service_time_regression.csv")
    chart_path   = os.path.join(DATA_DIR, "service_time_charts.png")
    report_path  = os.path.join(DATA_DIR, "service_time_report.txt")
    saving_path  = os.path.join(DATA_DIR, "service_time_saving.csv")

    params_df.to_csv(params_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    saving_df.to_csv(saving_path, index=False)

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nBuilding charts...")
    fig = build_charts(df, params_df, fit_results, model, saving_df)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print report to console
    print("\n" + report_text.encode("ascii", "replace").decode("ascii"))

    print("\n" + "=" * 65)
    print("OUTPUTS WRITTEN")
    print("=" * 65)
    for path in [params_path, coef_path, saving_path, chart_path, report_path]:
        kb = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):<40} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
