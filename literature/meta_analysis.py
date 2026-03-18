"""
meta_analysis.py
----------------
Reads literature_search_log.csv, service_time_extraction.csv and produces:
  1. Pooled service time parameters per geography × stop-type cell (meta_analysis_results.csv)
  2. Synthesised covariate coefficients (covariate_synthesis.csv)
  3. Updated service_time_params_v2.csv  — drop-in replacement for the model
  4. meta_analysis_charts.png            — forest plots + funnel plots

Run: python literature/meta_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
PARENT      = os.path.dirname(HERE)
EXTRACT_CSV = os.path.join(HERE, "service_time_extraction.csv")
LOG_CSV     = os.path.join(HERE, "literature_search_log.csv")
COV_CSV     = os.path.join(HERE, "covariate_synthesis.csv")
META_OUT    = os.path.join(HERE, "meta_analysis_results.csv")
PARAMS_OUT  = os.path.join(PARENT, "data", "service_time_params_v2.csv")
CHARTS_OUT  = os.path.join(HERE, "meta_analysis_charts.png")

# ---------------------------------------------------------------------------
# Cell definitions — geography × stop-type matrix
# ---------------------------------------------------------------------------
CELLS = {
    "C1": {"name": "CBD Residential",   "stop_type": "Residential SFD", "geo": "CBD / Urban Dense"},
    "C2": {"name": "Urban MDU",          "stop_type": "MDU / Apartment",  "geo": "CBD / Urban Dense"},
    "C3": {"name": "CBD Commercial",     "stop_type": "Commercial",       "geo": "CBD / Urban Dense"},
    "C4": {"name": "Suburban Residential","stop_type": "Residential SFD", "geo": "Suburban / Fringe"},
    "C5": {"name": "Suburban MDU",        "stop_type": "MDU / Apartment",  "geo": "Suburban / Fringe"},
    "C6": {"name": "Suburban Commercial", "stop_type": "Commercial",       "geo": "Suburban / Fringe"},
}

# Synthetic fallback values from current model (seconds)
# These are used if a cell has fewer than 2 qualifying studies
# Updated to reflect literature findings:
#   - C2 MDU raised to 220s: Winkenbach (Boston) MDU ~240s; locker baseline 1620s confirms
#     MDU is materially longer than residential; 220s reflects B2C parcel with lobby
#   - C4 Suburban Residential: Winkenbach median ~49s (handoff only), Gevaers ~180s,
#     Santiago median 138s. Weighted toward 120s as full stop (walk+handoff+return)
#   - C3 CBD Commercial: Seattle freight study 984s (freight, not parcel); NYC ~540s (mixed).
#     B2C parcel adjustment ~40% of freight -> ~150-200s. Use 175s.
SYNTHETIC_FALLBACK = {
    "C1": {"mean": 110, "sd": 38},   # CBD residential: suburban + 30% CBD access penalty
    "C2": {"mean": 220, "sd": 85},   # MDU: Winkenbach Boston 240s; locker study confirms
    "C3": {"mean": 175, "sd": 70},   # CBD commercial: freight studies adjusted down ~40% for parcel
    "C4": {"mean": 120, "sd": 45},   # Suburban residential: Santiago 138s median; Winkenbach ~75s mean
    "C5": {"mean": 200, "sd": 78},   # Suburban MDU: lower than urban MDU (easier access)
    "C6": {"mean": 95,  "sd": 38},   # Suburban commercial: faster than CBD commercial
}

# B2C calibration adjustments for studies measuring freight/mixed traffic
# Applied where extraction notes flag non-B2C context
B2C_ADJUSTMENT = {
    "freight_to_parcel": 0.40,   # Freight dwell (Seattle) -> B2C parcel: ~40% of freight time
    "mixed_to_parcel":   0.55,   # Mixed B2B/B2C (NYC) -> pure B2C parcel
}

# Australia adjustment factors (multiplicative on mean)
# Based on known differences: compact inner-ring blocks, high MDU penetration
AUSTRALIA_ADJUSTMENT = {
    "C1": 1.05,   # CBD residential slightly longer (security, lifts)
    "C2": 1.10,   # MDU heavier in Sydney inner suburbs
    "C3": 1.00,   # Commercial broadly comparable
    "C4": 0.95,   # Suburban residential slightly faster (shorter walks, single-story)
    "C5": 1.05,   # Suburban MDU comparable to international
    "C6": 0.95,   # Suburban commercial slightly faster
}

# ---------------------------------------------------------------------------
# Step 1 — Load extraction data
# ---------------------------------------------------------------------------
def load_extraction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
    # Only use rows that have been filled in
    df = df[df["data_status"].isin(["extracted", "partial"])].copy()
    df["service_time_mean_sec"]   = pd.to_numeric(df["service_time_mean_sec"],   errors="coerce")
    df["service_time_median_sec"] = pd.to_numeric(df["service_time_median_sec"], errors="coerce")
    df["service_time_sd_sec"]     = pd.to_numeric(df["service_time_sd_sec"],     errors="coerce")
    df["sample_size_stops"]       = pd.to_numeric(df["sample_size_stops"],       errors="coerce")
    df["quality_score"]           = pd.to_numeric(df["quality_score"],           errors="coerce")

    # Where mean is missing but median is present, estimate mean from median
    # For lognormal: mean = median * exp(sigma^2/2); use CV=0.45 -> sigma~0.41 -> factor ~1.09
    missing_mean = df["service_time_mean_sec"].isna() & df["service_time_median_sec"].notna()
    df.loc[missing_mean, "service_time_mean_sec"] = df.loc[missing_mean, "service_time_median_sec"] * 1.09

    # Impute sample size = 500 if missing (conservative weight)
    df["sample_size_stops"] = df["sample_size_stops"].fillna(500)

    # Apply B2C calibration flag from notes column
    if "notes" in df.columns:
        notes_str    = df["notes"].fillna("").astype(str)
        freight_mask = notes_str.str.contains("NOT B2C|freight-attracting", na=False, case=False)
        mixed_mask   = notes_str.str.contains("Mixed B2B", na=False, case=False)
        df.loc[freight_mask, "service_time_mean_sec"] *= B2C_ADJUSTMENT["freight_to_parcel"]
        df.loc[mixed_mask,   "service_time_mean_sec"] *= B2C_ADJUSTMENT["mixed_to_parcel"]
        df["b2c_adjusted"] = freight_mask | mixed_mask

    # Only use quality >= 3
    df = df[df["quality_score"] >= 3]
    return df


# ---------------------------------------------------------------------------
# Step 2 — Variance-weighted pooling per cell
# ---------------------------------------------------------------------------
def pool_cell(rows: pd.DataFrame) -> dict:
    """
    Inverse-variance weighted pooling of means.
    Where SD is missing, impute from coefficient of variation across rows.
    Returns dict with pooled stats.
    """
    rows = rows.copy().reset_index(drop=True)
    # Drop any rows where mean is still NaN after imputation
    rows = rows.dropna(subset=["service_time_mean_sec"])
    if len(rows) == 0:
        return None

    # Impute missing SD using median CV from rows that have SD
    mask_sd = rows["service_time_sd_sec"].notna()
    if mask_sd.sum() > 0:
        cv_estimates = (rows.loc[mask_sd, "service_time_sd_sec"] /
                        rows.loc[mask_sd, "service_time_mean_sec"])
        median_cv = cv_estimates.median()
    else:
        median_cv = 0.45   # literature default: ~45% CV for service time
    rows.loc[~mask_sd, "service_time_sd_sec"] = (
        rows.loc[~mask_sd, "service_time_mean_sec"] * median_cv
    )

    n   = rows["sample_size_stops"].values
    mu  = rows["service_time_mean_sec"].values
    sd  = rows["service_time_sd_sec"].values

    # Standard error of each study mean
    se  = sd / np.sqrt(n)
    w   = 1.0 / (se ** 2)          # inverse-variance weights

    pooled_mean = np.sum(w * mu) / np.sum(w)
    pooled_se   = np.sqrt(1.0 / np.sum(w))
    pooled_sd   = np.average(sd, weights=n)  # sample-size weighted SD

    # I² heterogeneity statistic
    k   = len(rows)
    Q   = np.sum(w * (mu - pooled_mean) ** 2)
    df_ = k - 1
    i2  = max(0.0, (Q - df_) / Q * 100) if Q > 0 else 0.0

    if i2 < 25:
        het = "low"
    elif i2 < 75:
        het = "moderate"
    else:
        het = "high"

    # Log-normal parameters (for model consumption)
    sigma_ln = np.sqrt(np.log(1 + (pooled_sd / pooled_mean) ** 2))
    mu_ln    = np.log(pooled_mean) - 0.5 * sigma_ln ** 2

    return {
        "n_papers":          k,
        "total_sample_stops": int(n.sum()),
        "pooled_mean_sec":   round(pooled_mean, 1),
        "pooled_sd_sec":     round(pooled_sd, 1),
        "pooled_p10_sec":    round(stats.lognorm.ppf(0.10, s=sigma_ln, scale=np.exp(mu_ln)), 1),
        "pooled_p90_sec":    round(stats.lognorm.ppf(0.90, s=sigma_ln, scale=np.exp(mu_ln)), 1),
        "ci_low_mean_sec":   round(pooled_mean - 1.96 * pooled_se, 1),
        "ci_high_mean_sec":  round(pooled_mean + 1.96 * pooled_se, 1),
        "i_squared_pct":     round(i2, 1),
        "heterogeneity":     het,
        "dist_family":       "lognormal",
        "lognorm_mu":        round(mu_ln, 4),
        "lognorm_sigma":     round(sigma_ln, 4),
        "contributing_papers": "; ".join(rows["paper_id"].tolist()),
        "data_source":       "literature",
    }


# ---------------------------------------------------------------------------
# Step 3 — Run pooling for all cells
# ---------------------------------------------------------------------------
def run_meta_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for cell_id, meta in CELLS.items():
        # Match rows to cell
        if df.empty or "geography_cell" not in df.columns:
            cell_rows = pd.DataFrame()
        else:
            mask = (
                df["geography_cell"].str.contains(cell_id, na=False) |
                (
                    df["stop_type"].str.lower().str.contains(meta["stop_type"].lower().split("/")[0].strip(), na=False) &
                    df["urban_classification"].str.lower().str.contains(meta["geo"].lower().split("/")[0].strip(), na=False)
                )
            )
            cell_rows = df[mask]

        row = {"cell_id": cell_id, "cell_name": meta["name"],
               "stop_type": meta["stop_type"], "geography": meta["geo"]}

        if len(cell_rows) >= 2:
            pooled = pool_cell(cell_rows)
            if pooled is None:
                pooled = {}
            row.update(pooled)
            if pooled and not np.isnan(row.get("pooled_mean_sec", float("nan"))):
                # Apply Australia adjustment
                adj = AUSTRALIA_ADJUSTMENT[cell_id]
                row["au_mean_sec"]    = round(row["pooled_mean_sec"] * adj, 1)
                row["au_adjustment"]  = adj
                row["final_mean_sec"] = row["au_mean_sec"]
                row["final_sd_sec"]   = round(row["pooled_sd_sec"] * adj, 1)
                row["data_source"]    = "literature"
            else:
                # Pool returned NaN — fall back to synthetic
                fallback = SYNTHETIC_FALLBACK[cell_id]
                row.update({
                    "n_papers": len(cell_rows),
                    "total_sample_stops": 0,
                    "pooled_mean_sec": fallback["mean"],
                    "pooled_sd_sec": fallback["sd"],
                    "data_source": "synthetic_fallback (pool NaN)",
                    "au_adjustment": AUSTRALIA_ADJUSTMENT[cell_id],
                    "final_mean_sec": fallback["mean"],
                    "final_sd_sec": fallback["sd"],
                })
        else:
            # Fall back to synthetic
            fallback = SYNTHETIC_FALLBACK[cell_id]
            row.update({
                "n_papers": len(cell_rows),
                "total_sample_stops": int(cell_rows["sample_size_stops"].sum()) if len(cell_rows) else 0,
                "pooled_mean_sec":  fallback["mean"],
                "pooled_sd_sec":    fallback["sd"],
                "ci_low_mean_sec":  None,
                "ci_high_mean_sec": None,
                "i_squared_pct":    None,
                "heterogeneity":    "n/a - insufficient studies",
                "dist_family":      "lognormal",
                "lognorm_mu":       None,
                "lognorm_sigma":    None,
                "contributing_papers": "; ".join(cell_rows["paper_id"].tolist()) if len(cell_rows) else "",
                "data_source":      "synthetic_fallback",
                "au_adjustment":    AUSTRALIA_ADJUSTMENT[cell_id],
                "final_mean_sec":   fallback["mean"],
                "final_sd_sec":     fallback["sd"],
            })

        results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 4 — Build service_time_params_v2.csv (model input format)
# ---------------------------------------------------------------------------
def build_params_v2(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Output format matches service_time_model.py consumption:
      segment, stop_type, service_class, mean_sec, sd_sec,
      dist_family, lognorm_mu, lognorm_sigma, source
    """
    rows = []
    for _, r in meta_df.iterrows():
        mean = r["final_mean_sec"]
        sd   = r["final_sd_sec"]
        sigma_ln = np.sqrt(np.log(1 + (sd / mean) ** 2))
        mu_ln    = np.log(mean) - 0.5 * sigma_ln ** 2
        rows.append({
            "segment":       r["cell_id"],
            "cell_name":     r["cell_name"],
            "stop_type":     r["stop_type"],
            "geography":     r["geography"],
            "mean_sec":      mean,
            "sd_sec":        sd,
            "p10_sec":       round(stats.lognorm.ppf(0.10, s=sigma_ln, scale=np.exp(mu_ln)), 1),
            "p90_sec":       round(stats.lognorm.ppf(0.90, s=sigma_ln, scale=np.exp(mu_ln)), 1),
            "dist_family":   "lognormal",
            "lognorm_mu":    round(mu_ln, 4),
            "lognorm_sigma": round(sigma_ln, 4),
            "n_papers":      r.get("n_papers", 0),
            "total_n_stops": r.get("total_sample_stops", 0),
            "i_squared":     r.get("i_squared_pct", None),
            "source":        r.get("data_source", "synthetic_fallback"),
            "au_adjustment": r.get("au_adjustment", 1.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5 — Forest plot charts
# ---------------------------------------------------------------------------
def make_charts(meta_df: pd.DataFrame, extraction_df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#f8fafc")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    cells_with_data = meta_df[meta_df["n_papers"] >= 2]

    for idx, (_, cell_row) in enumerate(meta_df.iterrows()):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        cell_id = cell_row["cell_id"]

        # Get individual study means for this cell
        if extraction_df.empty or "geography_cell" not in extraction_df.columns:
            studies = pd.DataFrame()
        else:
            mask = extraction_df["geography_cell"].str.contains(cell_id, na=False)
            studies = extraction_df[mask].dropna(subset=["service_time_mean_sec"])

        if len(studies) == 0:
            ax.text(0.5, 0.5, "No data yet\n(pending extraction)",
                    ha="center", va="center", fontsize=9, color="#94a3b8",
                    transform=ax.transAxes)
            ax.set_title(cell_row["cell_name"], fontsize=9, fontweight="bold", pad=6)
            ax.set_xlim(0, 300); ax.set_yticks([])
            _style_ax(ax)
            continue

        # Forest plot: individual studies as dots, pooled as diamond
        y_positions = list(range(len(studies)))
        means  = studies["service_time_mean_sec"].values
        sd_col = studies["service_time_sd_sec"].copy()
        sd_col = sd_col.where(sd_col.notna(), other=pd.Series(means * 0.45, index=studies.index))
        sds    = sd_col.values
        ns     = studies["sample_size_stops"].values
        labels = (studies["authors"].str.split(",").str[0] + " (" +
                  studies["year"].astype(str) + ")").tolist()

        for i, (m, sd, n, lbl) in enumerate(zip(means, sds, ns, labels)):
            se = sd / np.sqrt(n)
            ax.errorbar(m, i, xerr=1.96 * se, fmt="o", color="#3b82f6",
                        capsize=3, markersize=5, linewidth=1.2)
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 300,
                    i, f"  {lbl}", va="center", fontsize=6.5, color="#374151")

        # Pooled estimate diamond
        pooled_m = cell_row["pooled_mean_sec"]
        ax.axvline(pooled_m, color="#10b981", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(pooled_m, -0.8, f"Pooled\n{pooled_m:.0f}s",
                ha="center", va="top", fontsize=7, color="#10b981", fontweight="bold")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Mean service time (seconds)", fontsize=7)
        ax.set_title(cell_row["cell_name"], fontsize=9, fontweight="bold", pad=6)
        _style_ax(ax)

    fig.suptitle("Service Time Meta-Analysis — Forest Plots by Geography × Stop Type\n"
                 "(Green dashed = pooled estimate; Blue = individual study with 95% CI)",
                 fontsize=11, fontweight="bold", y=1.01, color="#1e293b")
    plt.savefig(CHARTS_OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Charts saved: {CHARTS_OUT}")


def _style_ax(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.xaxis.label.set_color("#64748b")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("META-ANALYSIS: Service Time Literature Synthesis")
    print("=" * 60)

    # Load extraction data
    print("\n[1] Loading extraction data...")
    try:
        extraction_df = load_extraction(EXTRACT_CSV)
        print(f"    {len(extraction_df)} qualifying rows loaded "
              f"({extraction_df['paper_id'].nunique()} unique papers)")
    except Exception as e:
        print(f"    WARNING: Could not load extraction data: {e}")
        extraction_df = pd.DataFrame(columns=[
            "paper_id", "authors", "year", "country", "city",
            "urban_classification", "network_type", "stop_type",
            "geography_cell", "data_collection_method", "year_of_data",
            "sample_size_stops", "service_time_mean_sec", "service_time_sd_sec",
            "quality_score", "data_status", "distribution_type_best_fit",
        ])

    # Run pooling
    print("\n[2] Running variance-weighted pooling per cell...")
    meta_df = run_meta_analysis(extraction_df)
    for _, r in meta_df.iterrows():
        src = r.get("data_source", "synthetic_fallback")
        i2  = r.get("i_squared_pct", None)
        i2_str = f"I²={i2:.0f}%" if i2 is not None else "I²=n/a"
        print(f"    {r['cell_id']} {r['cell_name']:<25}  "
              f"mean={r['final_mean_sec']:.0f}s  sd={r['final_sd_sec']:.0f}s  "
              f"n={r.get('n_papers',0)} papers  {i2_str}  [{src}]")

    # Save meta results
    print("\n[3] Saving meta_analysis_results.csv...")
    meta_df.to_csv(META_OUT, index=False)
    print(f"    Saved: {META_OUT}")

    # Build and save model params v2
    print("\n[4] Building service_time_params_v2.csv...")
    params_v2 = build_params_v2(meta_df)
    os.makedirs(os.path.dirname(PARAMS_OUT), exist_ok=True)
    params_v2.to_csv(PARAMS_OUT, index=False)
    print(f"    Saved: {PARAMS_OUT}")

    # Charts
    print("\n[5] Generating forest plot charts...")
    try:
        make_charts(meta_df, extraction_df)
    except Exception as e:
        print(f"    WARNING: Chart generation failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    if "data_source" in meta_df.columns:
        lit_cells   = meta_df[meta_df["data_source"] == "literature"]
    else:
        lit_cells   = pd.DataFrame()
    synth_cells = len(meta_df) - len(lit_cells)
    print(f"  Cells grounded in literature : {len(lit_cells)}/6")
    print(f"  Cells using synthetic fallback: {synth_cells}/6")
    print()
    print("  Next step: populate service_time_extraction.csv with")
    print("  actual values from the papers listed in literature_search_log.csv")
    print("  then re-run this script to update the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
