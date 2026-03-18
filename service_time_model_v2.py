"""
service_time_model_v2.py
------------------------
Statistically models per-stop service time using a three-component model:

  T_stop(n, density, cell, sig) =
        T_walk(density)              — vehicle-to-door walk, density-dependent
      + T_handoff(cell, sort_state)  — first-package door interaction
      + (n-1) × T_increment(cell)   — each additional package at same stop
      + sig × T_signature            — signature-required premium

Key improvements over v1:
  - Separates walk time (density-driven) from handoff time (stop-type-driven)
  - Multi-package sliding scale with per-stop-type increments
  - Parameters sourced from literature meta-analysis (data/service_time_params_v2.csv)
  - Sort benefit applied only to handoff component (walk time unaffected)

Outputs:
  data/service_time_report_v2.txt
  data/service_time_charts_v2.png
  data/service_time_saving_v2.csv

Run: python service_time_model_v2.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "data")
PARAMS_CSV = os.path.join(DATA_DIR, "service_time_params_v2.csv")
STOPS_CSV  = os.path.join(DATA_DIR, "stop_master.csv")
PKG_CSV    = os.path.join(DATA_DIR, "package_manifest.csv")
HIST_CSV   = os.path.join(DATA_DIR, "service_time_log.csv")

OUT_REPORT = os.path.join(DATA_DIR, "service_time_report_v2.txt")
OUT_CHARTS = os.path.join(DATA_DIR, "service_time_charts_v2.png")
OUT_SAVING = os.path.join(DATA_DIR, "service_time_saving_v2.csv")

# ---------------------------------------------------------------------------
# Sort benefit parameters
# Sort benefit applies ONLY to handoff component (not walk, not increment)
# Based on original model β₄ = -0.12 log units (8.7s saving at mean)
# ---------------------------------------------------------------------------
SORT_BENEFIT = {
    "handoff_reduction_pct":  0.12,   # 12% reduction in handoff time when sorted
    "increment_reduction_pct": 0.06,  # 6% reduction in per-package increment when sorted
    "signature_unchanged":     True,  # signature time unaffected by sort state
}

# Simulation constants
N_SIM       = 50_000    # Monte Carlo draws for distribution fitting
WORKING_DAYS = 253
N_DRIVERS   = 147

# ---------------------------------------------------------------------------
# Step 1 — Load model parameters
# ---------------------------------------------------------------------------
def load_params(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index("segment")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Walk time function (density-dependent)
# ---------------------------------------------------------------------------
WALK = {
    "walk_factor":    1.35,
    "walk_speed_mps": 1.2,
    "walk_fixed_sec": 12.0,   # raised: vehicle egress + gate/door approach
}

def walk_time_mean(density_stops_km2: float) -> float:
    d = max(density_stops_km2, 1.0)
    d_m = WALK["walk_factor"] * 500.0 / np.sqrt(d)
    return 2 * d_m / WALK["walk_speed_mps"] + WALK["walk_fixed_sec"]


def sample_walk_time(density: float, n: int = 1) -> np.ndarray:
    """Log-normal walk time samples with ~30% CV."""
    mu    = walk_time_mean(density)
    sigma = mu * 0.30
    sigma_ln = np.sqrt(np.log(1 + (sigma / mu) ** 2))
    mu_ln    = np.log(mu) - 0.5 * sigma_ln ** 2
    return np.random.lognormal(mu_ln, sigma_ln, n)


# ---------------------------------------------------------------------------
# Step 3 — Per-stop service time sampler
# ---------------------------------------------------------------------------
def sample_stop_service_time(
    cell_id: str,
    n_packages: int,
    density: float,
    params: pd.DataFrame,
    signature_required: bool = False,
    sorted_state: bool = True,
    n_sim: int = 1,
) -> np.ndarray:
    """
    Draw n_sim service time samples for a stop with given characteristics.

    Returns array of shape (n_sim,) in seconds.
    """
    if cell_id not in params.index:
        cell_id = "C4"   # fallback to suburban residential

    p = params.loc[cell_id]

    # -- Walk time (density-driven, sort-state-independent) --
    t_walk = sample_walk_time(density, n_sim)

    # -- Base handoff (first package, sort-state-dependent) --
    mu_h    = p["handoff_lognorm_mu"]
    sig_h   = p["handoff_lognorm_sigma"]
    t_handoff = np.random.lognormal(mu_h, sig_h, n_sim)

    if sorted_state:
        # Sort benefit: reduce handoff by SORT_BENEFIT factor
        # Modelled as multiplicative — sorted packages need less search/handling at door
        sort_factor = 1.0 - SORT_BENEFIT["handoff_reduction_pct"]
        t_handoff   = t_handoff * sort_factor

    # -- Increment for additional packages (if n > 1) --
    extra_pkgs = max(0, n_packages - 1)
    if extra_pkgs > 0:
        incr_mean = p["increment_per_pkg_mean_sec"]
        incr_sd   = p["increment_per_pkg_sd_sec"]
        if sorted_state:
            incr_mean = incr_mean * (1.0 - SORT_BENEFIT["increment_reduction_pct"])
        # Increment per package is normally distributed (shorter, less skewed)
        t_increment = np.random.normal(incr_mean, incr_sd, (n_sim, extra_pkgs))
        t_increment = np.clip(t_increment, 5, incr_mean * 4)
        t_increment = t_increment.sum(axis=1)
    else:
        t_increment = np.zeros(n_sim)

    # -- Signature premium --
    if signature_required:
        sig_mean = p["signature_premium_mean_sec"]
        sig_sd   = p["signature_premium_sd_sec"]
        t_sig    = np.random.normal(sig_mean, sig_sd, n_sim)
        t_sig    = np.clip(t_sig, 10, sig_mean * 3)
    else:
        t_sig = np.zeros(n_sim)

    total = t_walk + t_handoff + t_increment + t_sig
    return np.maximum(total, 15.0)   # floor: 15s minimum realistic stop time


# ---------------------------------------------------------------------------
# Step 4 — Analyse distribution of total stop time per cell
# ---------------------------------------------------------------------------
def fit_distribution(samples: np.ndarray) -> dict:
    """Fit lognormal and gamma, return best-fit parameters and AIC."""
    samples = samples[samples > 0]
    n       = len(samples)

    # Lognormal
    sig_ln, loc_ln, scale_ln = stats.lognorm.fit(samples, floc=0)
    mu_ln = np.log(scale_ln)
    ll_ln = stats.lognorm.logpdf(samples, sig_ln, loc=0, scale=scale_ln).sum()
    aic_ln = 2 * 2 - 2 * ll_ln

    # Gamma
    a_gm, loc_gm, scale_gm = stats.gamma.fit(samples, floc=0)
    ll_gm = stats.gamma.logpdf(samples, a_gm, loc=0, scale=scale_gm).sum()
    aic_gm = 2 * 2 - 2 * ll_gm

    best = "lognormal" if aic_ln < aic_gm else "gamma"
    return {
        "best_fit":       best,
        "lognorm_mu":     round(mu_ln, 4),
        "lognorm_sigma":  round(sig_ln, 4),
        "gamma_alpha":    round(a_gm, 4),
        "gamma_beta":     round(scale_gm, 4),
        "aic_lognorm":    round(aic_ln, 1),
        "aic_gamma":      round(aic_gm, 1),
        "empirical_mean": round(samples.mean(), 1),
        "empirical_sd":   round(samples.std(), 1),
        "empirical_p10":  round(np.percentile(samples, 10), 1),
        "empirical_p50":  round(np.percentile(samples, 50), 1),
        "empirical_p90":  round(np.percentile(samples, 90), 1),
    }


# ---------------------------------------------------------------------------
# Step 5 — Compute fleet-wide saving from sort benefit
# ---------------------------------------------------------------------------
def compute_saving(params: pd.DataFrame,
                   stops_df: pd.DataFrame,
                   pkg_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cell, compute mean service time per stop in sorted vs unsorted state
    weighted by actual stop/package distribution in the data.

    Returns a saving summary DataFrame.
    """
    # Packages per stop from manifest
    if pkg_df is not None and not pkg_df.empty:
        pkg_id_col   = "parcel_id" if "parcel_id" in pkg_df.columns else pkg_df.columns[0]
        pkg_per_stop = (pkg_df.groupby("stop_id")[pkg_id_col]
                        .count().reset_index(name="n_packages"))
    else:
        pkg_per_stop = pd.DataFrame({"stop_id": [], "n_packages": []})

    # Map stop_master to cells
    CBD_SUBURBS  = {"Sydney CBD"}
    INNER_SUBURBS = {"Surry Hills","Darlinghurst","Paddington","Waterloo","Alexandria",
                     "Glebe","Double Bay","Rose Bay","Bondi","Kensington"}

    def assign_cell(row):
        st  = str(row.get("stop_type", "residential")).lower()
        sub = str(row.get("suburb", "")).strip()
        if sub in CBD_SUBURBS:
            if "apartment" in st: return "C2"
            if "business"  in st: return "C3"
            return "C1"
        else:
            if "apartment" in st: return "C5"
            if "business"  in st: return "C6"
            return "C4"

    if stops_df is not None and not stops_df.empty:
        stops = stops_df.copy()
        stops["cell"]    = stops.apply(assign_cell, axis=1)
        stops["density"] = stops["cell"].map({
            "C1": 3000, "C2": 2500, "C3": 4000,
            "C4": 800,  "C5": 1200, "C6": 1500,
        })
        stops["signature"] = False   # not in stop master; default to no
    else:
        # Synthetic stop distribution
        n_stops = 13_780
        cells   = ["C1","C2","C3","C4","C5","C6"]
        weights = [0.05, 0.25, 0.10, 0.45, 0.10, 0.05]
        rng     = np.random.default_rng(42)
        stops   = pd.DataFrame({
            "stop_id":   range(n_stops),
            "cell":      rng.choice(cells, n_stops, p=weights),
            "density":   rng.uniform(500, 3000, n_stops),
            "signature": rng.random(n_stops) < 0.08,
        })

    # Merge packages per stop
    if not pkg_per_stop.empty:
        stops = stops.merge(pkg_per_stop, on="stop_id", how="left")
        stops["n_packages"] = stops["n_packages"].fillna(1).astype(int)
    else:
        stops["n_packages"] = 1

    rows = []
    N_DRAW = 5_000

    for cell_id in params.index:
        cell_stops = stops[stops.get("cell", pd.Series()) == cell_id] if "cell" in stops.columns else stops.iloc[:100]
        if len(cell_stops) == 0:
            continue

        # Sample representative density and package count
        density    = cell_stops["density"].mean() if "density" in cell_stops.columns else CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)
        n_pkgs_arr = cell_stops["n_packages"].values
        sig_arr    = cell_stops.get("signature", pd.Series([False]*len(cell_stops))).values

        # Draw service times for unsorted and sorted
        t_unsorted_total = 0.0
        t_sorted_total   = 0.0
        n_valid          = 0

        for i in range(min(200, len(cell_stops))):
            n_pkgs = int(n_pkgs_arr[i % len(n_pkgs_arr)])
            sig    = bool(sig_arr[i % len(sig_arr)])

            u = sample_stop_service_time(cell_id, n_pkgs, density, params,
                                         sig, sorted_state=False, n_sim=1)[0]
            s = sample_stop_service_time(cell_id, n_pkgs, density, params,
                                         sig, sorted_state=True,  n_sim=1)[0]
            t_unsorted_total += u
            t_sorted_total   += s
            n_valid += 1

        mean_unsorted = t_unsorted_total / n_valid
        mean_sorted   = t_sorted_total   / n_valid
        saving_per_stop = mean_unsorted - mean_sorted
        n_stops_this_cell = len(cell_stops)

        rows.append({
            "cell_id":             cell_id,
            "cell_name":           params.loc[cell_id, "cell_name"],
            "stop_type":           params.loc[cell_id, "stop_type"],
            "n_stops_sample":      n_stops_this_cell,
            "mean_density":        round(density, 0),
            "mean_n_packages":     round(cell_stops["n_packages"].mean() if "n_packages" in cell_stops.columns else 1.85, 2),
            "mean_unsorted_sec":   round(mean_unsorted, 1),
            "mean_sorted_sec":     round(mean_sorted, 1),
            "saving_per_stop_sec": round(saving_per_stop, 1),
            "saving_per_stop_pct": round(saving_per_stop / mean_unsorted * 100, 1),
            "annual_saving_hrs":   round(saving_per_stop * n_stops_this_cell * WORKING_DAYS / 3600, 1),
            "annual_saving_usd":   round(saving_per_stop * n_stops_this_cell * WORKING_DAYS / 3600 * 32 * 1.35, 0),
        })

    return pd.DataFrame(rows)


CELL_DENSITY_REF_DEFAULT = {
    "C1": 3000, "C2": 2500, "C3": 4000,
    "C4": 800,  "C5": 1200, "C6": 1500,
}


# ---------------------------------------------------------------------------
# Step 6 — Sliding scale chart
# ---------------------------------------------------------------------------
def make_charts(params: pd.DataFrame):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#f8fafc")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    n_range = np.arange(1, 11)
    palette = {"sorted": "#10b981", "unsorted": "#ef4444"}

    for idx, cell_id in enumerate(params.index):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        p  = params.loc[cell_id]
        density = CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)
        t_walk  = walk_time_mean(density)

        for state, sorted_flag, label in [
            ("unsorted", False, "Unsorted (current)"),
            ("sorted",   True,  "Zone sorted"),
        ]:
            means = []
            p10s  = []
            p90s  = []
            for n in n_range:
                samp = sample_stop_service_time(
                    cell_id, n, density, params,
                    signature_required=False,
                    sorted_state=sorted_flag,
                    n_sim=5000,
                )
                means.append(samp.mean())
                p10s.append(np.percentile(samp, 10))
                p90s.append(np.percentile(samp, 90))

            color = palette[state]
            ax.plot(n_range, means, color=color, linewidth=2.2, label=label, marker="o", markersize=4)
            ax.fill_between(n_range, p10s, p90s, color=color, alpha=0.12)

        # Annotate walk time
        ax.axhline(t_walk, color="#6366f1", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(1.2, t_walk + 3, f"walk={t_walk:.0f}s", fontsize=6, color="#6366f1")

        src = p["source_handoff"] if "source_handoff" in p.index else "?"
        ax.set_title(
            f"{p['cell_name']}\n"
            f"density={density} stops/km²  [{src[:10]}]",
            fontsize=8.5, fontweight="bold", pad=5,
        )
        ax.set_xlabel("Packages at stop", fontsize=7.5)
        ax.set_ylabel("Service time (seconds)", fontsize=7.5)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_xlim(1, 10)
        ax.set_xticks(n_range)
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values(): sp.set_color("#e2e8f0")
        ax.tick_params(colors="#64748b", labelsize=7)

    fig.suptitle(
        "Service Time Model v2 — Sliding Scale by Packages at Stop\n"
        "Shaded band = P10–P90 uncertainty  |  Dotted line = walk component  |  "
        "Green=sorted, Red=unsorted",
        fontsize=10, fontweight="bold", y=1.01, color="#1e293b",
    )
    plt.savefig(OUT_CHARTS, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Charts saved: {OUT_CHARTS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SERVICE TIME MODEL v2")
    print("Walk + Handoff + Multi-Package Increment")
    print("=" * 65)

    print("\n[1] Loading parameters from meta-analysis...")
    params = load_params(PARAMS_CSV)
    print(f"    {len(params)} cells loaded")

    # Load operational data if available
    try:
        stops_df = pd.read_csv(STOPS_CSV)
        print(f"    Stop master: {len(stops_df)} stops")
    except FileNotFoundError:
        stops_df = None
        print("    Stop master not found — using synthetic distribution")

    try:
        pkg_df = pd.read_csv(PKG_CSV)
        print(f"    Package manifest: {len(pkg_df)} packages")
    except FileNotFoundError:
        pkg_df = None
        print("    Package manifest not found — using synthetic pkg distribution")

    # Print full sliding scale
    print("\n[2] Service time sliding scale (mean seconds per stop):")
    print(f"    {'Cell':<22} {'Walk':>5}  {'n=1':>5}  {'n=2':>5}  {'n=3':>5}  {'n=5':>5} {'n=10':>5}   Sort saving @n=1")
    print("    " + "-" * 74)
    for cell_id, p in params.iterrows():
        density = CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)
        t_walk  = walk_time_mean(density)
        h       = p["handoff_mean_sec"]
        incr    = p["increment_per_pkg_mean_sec"]
        h_srt   = h * (1 - SORT_BENEFIT["handoff_reduction_pct"])
        vals_u  = [t_walk + h           + max(0, n-1)*incr for n in [1,2,3,5,10]]
        vals_s  = [t_walk + h_srt       + max(0, n-1)*incr*(1-SORT_BENEFIT["increment_reduction_pct"]) for n in [1,2,3,5,10]]
        saving  = vals_u[0] - vals_s[0]
        print(f"    {p['cell_name']:<22} {t_walk:>4.0f}s "
              f" {vals_u[0]:>4.0f}s"
              f" {vals_u[1]:>4.0f}s"
              f" {vals_u[2]:>4.0f}s"
              f" {vals_u[3]:>4.0f}s"
              f" {vals_u[4]:>4.0f}s"
              f"   -{saving:.0f}s ({saving/vals_u[0]*100:.1f}%)")

    print("\n[3] Fitting distributions per cell (sorted state)...")
    fit_results = {}
    for cell_id in params.index:
        density = CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)
        samp    = sample_stop_service_time(cell_id, 1, density, params,
                                           sorted_state=True, n_sim=N_SIM)
        fit     = fit_distribution(samp)
        fit_results[cell_id] = fit
        print(f"    {cell_id} {params.loc[cell_id,'cell_name']:<22} "
              f"mean={fit['empirical_mean']:.0f}s  p50={fit['empirical_p50']:.0f}s  "
              f"p90={fit['empirical_p90']:.0f}s  best={fit['best_fit']}")

    print("\n[4] Computing fleet-wide on-road saving...")
    saving_df = compute_saving(params, stops_df, pkg_df)
    total_annual = saving_df["annual_saving_usd"].sum()
    total_hrs    = saving_df["annual_saving_hrs"].sum()
    saving_df.to_csv(OUT_SAVING, index=False)
    print(f"    Saved: {OUT_SAVING}")
    print(f"\n    On-road service time saving:")
    for _, r in saving_df.iterrows():
        print(f"      {r['cell_name']:<22}  "
              f"{r['saving_per_stop_sec']:.1f}s/stop  "
              f"({r['saving_per_stop_pct']:.1f}%)  "
              f"${r['annual_saving_usd']:,.0f}/yr")
    print(f"\n    TOTAL ON-ROAD SAVING (service time component): ${total_annual:,.0f}/yr  "
          f"({total_hrs:,.0f} hrs)")

    print("\n[5] Generating sliding scale charts...")
    try:
        make_charts(params)
    except Exception as e:
        print(f"    WARNING: {e}")

    # Write report
    lines = [
        "SERVICE TIME MODEL v2 — RESULTS",
        "=" * 60,
        "",
        "MODEL STRUCTURE",
        "  T_stop(n, density, cell, sig) =",
        "    T_walk(density)                  [Poisson NN walk model]",
        "    + T_handoff(cell, sorted)        [first package, log-normal]",
        "    + (n-1) x T_increment(cell)      [additional packages]",
        "    + sig x T_signature              [signature premium +45s]",
        "",
        "WALK TIME MODEL (Poisson nearest-neighbour)",
        "  d_walk = 1.35 x 500 / sqrt(density)  [metres, one-way]",
        "  T_walk = 2 x d_walk / 1.2 + 8        [seconds, round trip]",
        "",
        "SORT BENEFIT",
        f"  Handoff reduction: {SORT_BENEFIT['handoff_reduction_pct']*100:.0f}%",
        f"  Increment reduction: {SORT_BENEFIT['increment_reduction_pct']*100:.0f}%",
        "  Walk time: unaffected (same route)",
        "",
        "CELL PARAMETERS (literature-grounded + Australia-adjusted)",
        "-" * 60,
    ]
    for cell_id, p in params.iterrows():
        lines += [
            f"  {cell_id} {p['cell_name']}",
            f"    Density ref:    {CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)} stops/km2",
            f"    Walk time:      {walk_time_mean(CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)):.0f}s",
            f"    Handoff mean:   {p['handoff_mean_sec']:.0f}s  (sorted: {p['handoff_mean_sec']*(1-SORT_BENEFIT['handoff_reduction_pct']):.0f}s)",
            f"    Increment/pkg:  {p['increment_per_pkg_mean_sec']:.0f}s",
            f"    Source:         {p['source_handoff']}",
            "",
        ]
    lines += [
        "",
        "SLIDING SCALE SUMMARY",
        "-" * 60,
        f"  {'Cell':<22} n=1(U)  n=1(S)  n=2(U)  n=5(U)  n=10(U)",
    ]
    for cell_id, p in params.iterrows():
        density = CELL_DENSITY_REF_DEFAULT.get(cell_id, 800)
        t_walk  = walk_time_mean(density)
        h       = p["handoff_mean_sec"]; hs = h*(1-SORT_BENEFIT["handoff_reduction_pct"])
        i       = p["increment_per_pkg_mean_sec"]
        lines.append(
            f"  {p['cell_name']:<22} "
            f"{t_walk+h:>5.0f}s  {t_walk+hs:>5.0f}s  "
            f"{t_walk+h+i:>5.0f}s  {t_walk+h+4*i:>6.0f}s  {t_walk+h+9*i:>6.0f}s"
        )
    lines += ["", "=" * 60]

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n    Report saved: {OUT_REPORT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
