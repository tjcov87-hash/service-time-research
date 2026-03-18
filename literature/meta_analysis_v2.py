"""
meta_analysis_v2.py
-------------------
Reads service_time_extraction_v2.csv and produces:

  1. Pooled BASE HANDOFF parameters per cell   (meta_base_handoff.csv)
  2. Pooled INCREMENT per package parameters   (meta_increment.csv)
  3. Walk time model parameters                (meta_walk_model.csv)
  4. Combined service_time_params_v2.csv       — drop-in for service_time_model_v2.py
  5. Forest plot charts                        (meta_analysis_v2_charts.png)

Model structure
---------------
  T_stop(n, density, stop_type, geo, sig) =
        T_walk(density)                      ← walk component, density-dependent
      + T_base_handoff(stop_type, geo)       ← first-package handoff, per cell
      + (n-1) × T_increment(stop_type)       ← additional packages, per stop type
      + sig × T_signature                    ← signature premium

  Walk time model (Poisson nearest-neighbour):
      d_walk_m = walk_factor × 500 / sqrt(density_stops_per_km2)
      T_walk   = d_walk_m / walk_speed_mps + walk_min_sec

Run: python literature/meta_analysis_v2.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
PARENT     = os.path.dirname(HERE)
EXTRACT    = os.path.join(HERE, "service_time_extraction_v2.csv")
OUT_BASE   = os.path.join(HERE, "meta_base_handoff.csv")
OUT_INCR   = os.path.join(HERE, "meta_increment.csv")
OUT_WALK   = os.path.join(HERE, "meta_walk_model.csv")
PARAMS_OUT = os.path.join(PARENT, "data", "service_time_params_v2.csv")
CHARTS_OUT = os.path.join(HERE, "meta_analysis_v2_charts.png")

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------
CELLS = {
    "C1": {"name": "CBD Residential",    "stop_type": "Residential SFD", "geo": "CBD / Urban Dense"},
    "C2": {"name": "Urban MDU",           "stop_type": "MDU / Apartment",  "geo": "CBD / Urban Dense"},
    "C3": {"name": "CBD Commercial",      "stop_type": "Commercial",       "geo": "CBD / Urban Dense"},
    "C4": {"name": "Suburban Residential","stop_type": "Residential SFD",  "geo": "Suburban / Fringe"},
    "C5": {"name": "Suburban MDU",        "stop_type": "MDU / Apartment",  "geo": "Suburban / Fringe"},
    "C6": {"name": "Suburban Commercial", "stop_type": "Commercial",       "geo": "Suburban / Fringe"},
}

# Stop type groupings for increment model
STOP_TYPES = {
    "Residential SFD": ["C1", "C4"],
    "MDU / Apartment":  ["C2", "C5"],
    "Commercial":       ["C3", "C6"],
}

# ---------------------------------------------------------------------------
# Walk time model parameters
# From Poisson nearest-neighbour theory + empirical calibration
#
#   d_mean_m = walk_factor × (500 / √density)   [mean walk dist metres, one-way]
#   T_walk   = 2 × d_mean_m / walk_speed + walk_fixed
#
#   Calibration targets:
#     density=3000 (Sydney CBD): T_walk ≈ 25s
#     density=1000 (inner suburb): T_walk ≈ 45s
#     density=300  (outer suburb): T_walk ≈ 75s
#     density=50   (rural fringe): T_walk ≈ 130s
# ---------------------------------------------------------------------------
WALK_MODEL = {
    # Calibrated against Oslo unattended study:
    #   Oslo unattended = 119s total = walk + handoff(~75s Winkenbach) → walk ~44s
    #   At Oslo urban density (~1500 stops/km2): model gives 37s → reasonable
    # Walking study (Dalla Chiara 2025): 7min outside-vehicle in urban CBD
    #   Interpreted as total outside-vehicle time (not walk-only) at very dense urban
    "walk_factor":    1.35,    # road/path detour factor (straight-line → actual path)
    "walk_speed_mps": 1.2,     # walking speed metres/second (human walk ~4.3 km/h)
    "walk_fixed_sec": 12.0,    # fixed: vehicle egress, gate/door, approach steps
    "walk_sd_frac":   0.35,    # SD as fraction of mean — higher than handoff due to parking variability
}

# Reference density for each cell (stops/km²)
# Calibrated to Sydney delivery zones
CELL_DENSITY_REF = {
    "C1": 3000,   # Sydney CBD residential — very dense
    "C2": 2500,   # Inner suburb MDU blocks
    "C3": 4000,   # CBD commercial — highest stop density
    "C4": 800,    # Suburban residential
    "C5": 1200,   # Suburban MDU
    "C6": 1500,   # Suburban commercial strip
}

# ---------------------------------------------------------------------------
# Base handoff priors (literature-grounded synthetic fallbacks)
# These are HANDOFF ONLY (door interaction, no walk)
# Updated from literature synthesis:
#   Winkenbach residential handoff: 75s median 49s lognormal
#   MDU: ~2× residential from Winkenbach model
#   Commercial: residential × 0.85 (someone available to receive)
# ---------------------------------------------------------------------------
HANDOFF_FALLBACK = {
    "C1": {"mean": 90,  "sd": 40, "lognorm_mu": 4.30, "lognorm_sigma": 0.43},
    "C2": {"mean": 220, "sd": 95, "lognorm_mu": 5.26, "lognorm_sigma": 0.41},
    "C3": {"mean": 85,  "sd": 38, "lognorm_mu": 4.27, "lognorm_sigma": 0.43},
    "C4": {"mean": 75,  "sd": 32, "lognorm_mu": 4.19, "lognorm_sigma": 0.41},
    "C5": {"mean": 200, "sd": 85, "lognorm_mu": 5.19, "lognorm_sigma": 0.41},
    "C6": {"mean": 70,  "sd": 30, "lognorm_mu": 4.12, "lognorm_sigma": 0.41},
}

# Increment per additional package — by stop type
INCREMENT_FALLBACK = {
    "Residential SFD": {"mean": 14, "sd": 6},
    "MDU / Apartment":  {"mean": 10, "sd": 5},
    "Commercial":       {"mean": 12, "sd": 5},
}

# Signature premium
SIGNATURE_PARAMS = {"mean": 45, "sd": 20}

# Australia calibration adjustments (multiplicative on handoff mean)
AU_ADJUSTMENT = {
    "C1": 1.05,   "C2": 1.15,   "C3": 1.00,
    "C4": 0.95,   "C5": 1.10,   "C6": 0.95,
}


# ---------------------------------------------------------------------------
# Helper: variance-weighted pooling
# ---------------------------------------------------------------------------
def pool_means(means: np.ndarray, sds: np.ndarray,
               ns: np.ndarray) -> dict:
    """Inverse-variance weighted pooling. Returns dict of pooled stats."""
    # Impute missing SD via median CV
    valid = ~np.isnan(sds)
    if valid.sum() > 0:
        cv = np.median(sds[valid] / means[valid])
    else:
        cv = 0.45
    sds = np.where(np.isnan(sds), means * cv, sds)

    se = sds / np.sqrt(ns)
    w  = 1.0 / (se ** 2)
    pooled_mean = np.sum(w * means) / np.sum(w)
    pooled_se   = np.sqrt(1.0 / np.sum(w))
    pooled_sd   = np.average(sds, weights=ns)

    k  = len(means)
    Q  = np.sum(w * (means - pooled_mean) ** 2)
    i2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 and k > 1 else np.nan

    sigma_ln = np.sqrt(np.log(1 + (pooled_sd / pooled_mean) ** 2))
    mu_ln    = np.log(pooled_mean) - 0.5 * sigma_ln ** 2

    return {
        "n_studies":    k,
        "total_n":      int(ns.sum()),
        "pooled_mean":  round(pooled_mean, 1),
        "pooled_sd":    round(pooled_sd, 1),
        "ci_low":       round(pooled_mean - 1.96 * pooled_se, 1),
        "ci_high":      round(pooled_mean + 1.96 * pooled_se, 1),
        "i2":           round(i2, 1) if not np.isnan(i2) else None,
        "lognorm_mu":   round(mu_ln, 4),
        "lognorm_sigma": round(sigma_ln, 4),
    }


# ---------------------------------------------------------------------------
# Step 1 — Load extraction data
# ---------------------------------------------------------------------------
def load_extraction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
    df = df[df["data_status"].isin(["extracted", "partial"])].copy()

    for col in ["total_stop_mean_sec", "total_stop_median_sec", "total_stop_sd_sec",
                "base_handoff_mean_sec", "base_handoff_sd_sec",
                "walk_time_mean_sec", "walk_time_sd_sec",
                "increment_per_pkg_mean_sec", "increment_per_pkg_sd_sec",
                "pkg_count_log_coef", "lognorm_mu", "lognorm_sigma",
                "signature_premium_sec", "sample_size_stops",
                "b2c_adjustment_factor", "quality_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample_size_stops"] = df["sample_size_stops"].fillna(500)
    df["quality_score"]     = df["quality_score"].fillna(3)

    # Where base_handoff not reported but total_stop and walk are, derive handoff
    has_total = df["total_stop_mean_sec"].notna()
    has_walk  = df["walk_time_mean_sec"].notna()
    missing_handoff = df["base_handoff_mean_sec"].isna()
    derive_mask = has_total & missing_handoff & ~has_walk
    # If total_stop only (no walk separated): treat as total stop, derive handoff
    # by subtracting expected walk time (computed later per density)
    # For now: store total_stop as base_handoff with flag
    df.loc[derive_mask, "base_handoff_mean_sec"] = df.loc[derive_mask, "total_stop_mean_sec"]
    df.loc[derive_mask, "handoff_includes_walk"] = True

    # Apply B2C calibration
    b2c_mask = (df["b2c_adjusted"].fillna("NO").astype(str).str.upper() == "YES")
    factor   = df["b2c_adjustment_factor"].fillna(0.45)
    df.loc[b2c_mask, "base_handoff_mean_sec"] = (
        df.loc[b2c_mask, "base_handoff_mean_sec"] * factor[b2c_mask]
    )

    # Derive increment from log coef if not directly reported
    has_coef      = df["pkg_count_log_coef"].notna()
    missing_incr  = df["increment_per_pkg_mean_sec"].isna()
    has_handoff   = df["base_handoff_mean_sec"].notna()
    derive_incr   = has_coef & missing_incr & has_handoff
    df.loc[derive_incr, "increment_per_pkg_mean_sec"] = (
        (np.exp(df.loc[derive_incr, "pkg_count_log_coef"]) - 1)
        * df.loc[derive_incr, "base_handoff_mean_sec"]
    ).round(1)

    return df[df["quality_score"] >= 3]


# ---------------------------------------------------------------------------
# Step 2 — Walk time model
# ---------------------------------------------------------------------------
def walk_time(density: float) -> float:
    """
    Mean walk time (seconds) at given stop density (stops/km²).
    Uses Poisson nearest-neighbour mean distance × round-trip factor.
    """
    p = WALK_MODEL
    d_one_way_m = p["walk_factor"] * 500.0 / np.sqrt(max(density, 1.0))
    t_walk      = 2 * d_one_way_m / p["walk_speed_mps"] + p["walk_fixed_sec"]
    return round(t_walk, 1)


def walk_sd(density: float) -> float:
    t = walk_time(density)
    return round(t * WALK_MODEL["walk_sd_frac"], 1)


# ---------------------------------------------------------------------------
# Step 3 — Pool handoff parameters per cell
# ---------------------------------------------------------------------------
def pool_handoff_per_cell(df: pd.DataFrame) -> dict:
    results = {}
    for cell_id, meta in CELLS.items():
        mask = df["geography_cell"].astype(str).str.contains(cell_id, na=False)
        rows = df[mask].dropna(subset=["base_handoff_mean_sec"])
        rows = rows[rows["base_handoff_mean_sec"] > 0]

        fallback = HANDOFF_FALLBACK[cell_id]
        au_adj   = AU_ADJUSTMENT[cell_id]

        if len(rows) >= 1:
            means = rows["base_handoff_mean_sec"].values
            sds   = rows["base_handoff_sd_sec"].values
            ns    = rows["sample_size_stops"].values
            p     = pool_means(means, sds, ns)

            lit_mean = p["pooled_mean"] * au_adj
            lit_sd   = p["pooled_sd"]   * au_adj
            sigma_ln = np.sqrt(np.log(1 + (lit_sd / lit_mean) ** 2))
            mu_ln    = np.log(lit_mean) - 0.5 * sigma_ln ** 2
            source   = "literature" if len(rows) >= 2 else "literature_single_study"

            results[cell_id] = {
                "cell_id": cell_id, "cell_name": meta["name"],
                "stop_type": meta["stop_type"], "geography": meta["geo"],
                "n_studies": p["n_studies"], "total_n": p["total_n"],
                "handoff_mean_sec": round(lit_mean, 1),
                "handoff_sd_sec":   round(lit_sd, 1),
                "handoff_lognorm_mu":    round(mu_ln, 4),
                "handoff_lognorm_sigma": round(sigma_ln, 4),
                "i2": p["i2"],
                "ci_low": round(p["ci_low"] * au_adj, 1),
                "ci_high": round(p["ci_high"] * au_adj, 1),
                "au_adjustment": au_adj,
                "source": source,
            }
        else:
            results[cell_id] = {
                "cell_id": cell_id, "cell_name": meta["name"],
                "stop_type": meta["stop_type"], "geography": meta["geo"],
                "n_studies": 0, "total_n": 0,
                "handoff_mean_sec": fallback["mean"],
                "handoff_sd_sec":   fallback["sd"],
                "handoff_lognorm_mu":    fallback["lognorm_mu"],
                "handoff_lognorm_sigma": fallback["lognorm_sigma"],
                "i2": None, "ci_low": None, "ci_high": None,
                "au_adjustment": au_adj,
                "source": "synthetic_fallback",
            }
    return results


# ---------------------------------------------------------------------------
# Step 4 — Pool increment parameters by stop type
# ---------------------------------------------------------------------------
def pool_increment_by_stop_type(df: pd.DataFrame) -> dict:
    results = {}
    for st_name, cell_ids in STOP_TYPES.items():
        mask = df["stop_type"].astype(str).str.contains(
            st_name.split("/")[0].strip(), na=False, case=False)
        rows = df[mask].dropna(subset=["increment_per_pkg_mean_sec"])
        rows = rows[rows["increment_per_pkg_mean_sec"] > 0]

        fallback = INCREMENT_FALLBACK[st_name]

        if len(rows) >= 1:
            means = rows["increment_per_pkg_mean_sec"].values
            sds   = rows["increment_per_pkg_sd_sec"].values if "increment_per_pkg_sd_sec" in rows.columns else np.full(len(rows), np.nan)
            ns    = rows["sample_size_stops"].values
            p     = pool_means(means, sds, ns)
            results[st_name] = {
                "stop_type": st_name,
                "increment_mean_sec": p["pooled_mean"],
                "increment_sd_sec":   p["pooled_sd"],
                "n_studies": p["n_studies"],
                "source": "literature",
            }
        else:
            results[st_name] = {
                "stop_type": st_name,
                "increment_mean_sec": fallback["mean"],
                "increment_sd_sec":   fallback["sd"],
                "n_studies": 0,
                "source": "synthetic_fallback",
            }
    return results


# ---------------------------------------------------------------------------
# Step 5 — Build service_time_params_v2.csv
# ---------------------------------------------------------------------------
def build_params_v2(handoff: dict, increment: dict) -> pd.DataFrame:
    rows = []
    for cell_id, h in handoff.items():
        st = h["stop_type"]
        incr = increment.get(st, INCREMENT_FALLBACK.get(st, {"mean": 14, "sd": 6}))
        density = CELL_DENSITY_REF[cell_id]
        t_walk  = walk_time(density)
        t_walk_sd = walk_sd(density)

        total_mean = h["handoff_mean_sec"] + t_walk
        total_sd   = np.sqrt(h["handoff_sd_sec"] ** 2 + t_walk_sd ** 2)
        sigma_total = np.sqrt(np.log(1 + (total_sd / total_mean) ** 2))
        mu_total    = np.log(total_mean) - 0.5 * sigma_total ** 2

        rows.append({
            "segment":                cell_id,
            "cell_name":              h["cell_name"],
            "stop_type":              st,
            "geography":              h["geography"],
            "density_ref_stops_km2":  density,
            # Walk time component
            "walk_time_mean_sec":     t_walk,
            "walk_time_sd_sec":       t_walk_sd,
            # Base handoff (first package, at door)
            "handoff_mean_sec":       h["handoff_mean_sec"],
            "handoff_sd_sec":         h["handoff_sd_sec"],
            "handoff_lognorm_mu":     h["handoff_lognorm_mu"],
            "handoff_lognorm_sigma":  h["handoff_lognorm_sigma"],
            # Increment per additional package
            "increment_per_pkg_mean_sec": incr["increment_mean_sec"],
            "increment_per_pkg_sd_sec":   incr["increment_sd_sec"],
            # Total stop time (walk + handoff, n=1 package)
            "total_n1_mean_sec":      round(total_mean, 1),
            "total_n1_sd_sec":        round(total_sd, 1),
            "total_n1_lognorm_mu":    round(mu_total, 4),
            "total_n1_lognorm_sigma": round(sigma_total, 4),
            # Signature premium
            "signature_premium_mean_sec": SIGNATURE_PARAMS["mean"],
            "signature_premium_sd_sec":   SIGNATURE_PARAMS["sd"],
            # Provenance
            "n_studies_handoff":  h["n_studies"],
            "n_stops_handoff":    h["total_n"],
            "i2_handoff":         h.get("i2"),
            "source_handoff":     h["source"],
            "n_studies_increment": incr["n_studies"],
            "source_increment":   incr["source"],
            "au_adjustment":      h["au_adjustment"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Sliding scale preview table
# ---------------------------------------------------------------------------
def print_sliding_scale(params_df: pd.DataFrame):
    print("\n  Service Time Sliding Scale — T(n) = walk + handoff + (n-1) × increment")
    print("  " + "-" * 80)
    hdr = f"  {'Cell':<20} {'Walk':>6} {'n=1':>6} {'n=2':>6} {'n=3':>6} {'n=5':>7} {'n=10':>7}  Source"
    print(hdr)
    print("  " + "-" * 80)
    for _, r in params_df.iterrows():
        w = r["walk_time_mean_sec"]
        h = r["handoff_mean_sec"]
        i = r["increment_per_pkg_mean_sec"]
        src = r["source_handoff"][:12]
        vals = [w + h + max(0, n - 1) * i for n in [1, 2, 3, 5, 10]]
        print(f"  {r['cell_name']:<20} {w:>5.0f}s "
              f"{vals[0]:>5.0f}s {vals[1]:>5.0f}s {vals[2]:>5.0f}s "
              f"{vals[3]:>6.0f}s {vals[4]:>6.0f}s  [{src}]")
    print("  " + "-" * 80)


# ---------------------------------------------------------------------------
# Step 7 — Charts
# ---------------------------------------------------------------------------
def make_charts(params_df: pd.DataFrame, df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f8fafc")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    # Top row: forest plots (handoff only) for C1-C3
    # Bottom row: forest plots for C4-C6
    cell_order = ["C1", "C2", "C3", "C4", "C5", "C6"]

    for idx, cell_id in enumerate(cell_order):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        cname = CELLS[cell_id]["name"]

        # Individual studies
        mask    = df["geography_cell"].astype(str).str.contains(cell_id, na=False)
        studies = df[mask].dropna(subset=["base_handoff_mean_sec"])

        density = CELL_DENSITY_REF[cell_id]
        t_walk  = walk_time(density)

        # Reference line: synthetic fallback
        fallback_mean = HANDOFF_FALLBACK[cell_id]["mean"] + t_walk

        # Build display data
        study_means  = []
        study_cis    = []
        study_labels = []

        for _, row in studies.iterrows():
            m  = row["base_handoff_mean_sec"]
            sd = row["base_handoff_sd_sec"] if not np.isnan(row.get("base_handoff_sd_sec", np.nan)) else m * 0.45
            n  = row["sample_size_stops"]
            se = sd / np.sqrt(n)
            study_means.append(m + t_walk)
            study_cis.append(1.96 * se)
            author = str(row["authors"]).split(",")[0]
            study_labels.append(f"{author} ({row['year']})")

        if len(study_means) == 0:
            ax.text(0.5, 0.5, "No data yet\n(pending extraction)",
                    ha="center", va="center", fontsize=9, color="#94a3b8",
                    transform=ax.transAxes)
        else:
            y_pos = list(range(len(study_means)))
            for i, (m, ci, lbl) in enumerate(zip(study_means, study_cis, study_labels)):
                ax.errorbar(m, i, xerr=ci, fmt="o", color="#3b82f6",
                            capsize=3, markersize=5, linewidth=1.2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(study_labels, fontsize=6.5)
            ax.set_ylim(-1, max(len(study_means), 1))
            ax.axvline(np.mean(study_means), color="#10b981", linewidth=1.5,
                       linestyle="--", alpha=0.8)

        ax.axvline(fallback_mean, color="#f59e0b", linewidth=1, linestyle=":", alpha=0.7)
        xmax = max(max(study_means) * 1.5 if study_means else 400, fallback_mean * 1.5)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Total stop time n=1 (seconds)", fontsize=7)
        title_src = "LITERATURE" if len(study_means) > 0 else "SYNTHETIC"
        ax.set_title(f"{cname}\n[{title_src}] walk={t_walk:.0f}s",
                     fontsize=8.5, fontweight="bold", pad=5)

        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values(): sp.set_color("#e2e8f0")
        ax.tick_params(colors="#64748b", labelsize=7)
        ax.xaxis.label.set_color("#64748b")

    fig.suptitle(
        "Service Time Meta-Analysis v2 — Base Handoff + Walk Component per Cell\n"
        "Blue = study estimate with 95% CI  |  Green dashed = pooled  |  Yellow dotted = synthetic prior",
        fontsize=10, fontweight="bold", y=1.01, color="#1e293b")
    plt.savefig(CHARTS_OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Charts saved: {CHARTS_OUT}")


# ---------------------------------------------------------------------------
# Step 8 — Walk model summary table
# ---------------------------------------------------------------------------
def walk_model_table() -> pd.DataFrame:
    densities = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 4000]
    rows = []
    for d in densities:
        t = walk_time(d)
        rows.append({
            "density_stops_km2": d,
            "mean_nn_distance_m": round(WALK_MODEL["walk_factor"] * 500 / np.sqrt(d), 1),
            "walk_time_sec": t,
            "walk_time_sd_sec": walk_sd(d),
            "context": (
                "Rural fringe" if d < 100 else
                "Outer suburban" if d < 300 else
                "Suburban" if d < 800 else
                "Inner suburban" if d < 1500 else
                "Urban dense" if d < 3000 else "CBD"
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("META-ANALYSIS v2: Base Handoff + Increment + Walk Model")
    print("=" * 65)

    print("\n[1] Loading extraction data...")
    try:
        df = load_extraction(EXTRACT)
        print(f"    {len(df)} qualifying rows loaded "
              f"({df['paper_id'].nunique()} unique papers)")
    except Exception as e:
        print(f"    WARNING: {e}")
        df = pd.DataFrame(columns=HEADER_COLS)

    print("\n[2] Pooling base handoff parameters per cell...")
    handoff = pool_handoff_per_cell(df if not df.empty else pd.DataFrame())
    for cid, h in handoff.items():
        i2str = f"I²={h['i2']:.0f}%" if h["i2"] is not None else "I²=n/a"
        print(f"    {cid} {h['cell_name']:<22}  handoff={h['handoff_mean_sec']:.0f}s  "
              f"sd={h['handoff_sd_sec']:.0f}s  n={h['n_studies']}  {i2str}  [{h['source']}]")

    print("\n[3] Pooling increment-per-package by stop type...")
    increment = pool_increment_by_stop_type(df if not df.empty else pd.DataFrame())
    for st, inc in increment.items():
        print(f"    {st:<20}  increment={inc['increment_mean_sec']:.1f}s/pkg  "
              f"n={inc['n_studies']}  [{inc['source']}]")

    print("\n[4] Walk time model summary:")
    walk_df = walk_model_table()
    walk_df.to_csv(OUT_WALK, index=False)
    for _, r in walk_df.iterrows():
        print(f"    density={r['density_stops_km2']:>5} stops/km²  "
              f"avg_walk={r['mean_nn_distance_m']:>5.0f}m  "
              f"T_walk={r['walk_time_sec']:>4.0f}s  ({r['context']})")

    print("\n[5] Building service_time_params_v2.csv...")
    params_df = build_params_v2(handoff, increment)
    os.makedirs(os.path.dirname(PARAMS_OUT), exist_ok=True)
    params_df.to_csv(PARAMS_OUT, index=False)
    print(f"    Saved: {PARAMS_OUT}")

    print_sliding_scale(params_df)

    pd.DataFrame(list(handoff.values())).to_csv(OUT_BASE, index=False)
    pd.DataFrame(list(increment.values())).to_csv(OUT_INCR, index=False)

    print("\n[6] Generating charts...")
    try:
        make_charts(params_df, df if not df.empty else pd.DataFrame())
    except Exception as e:
        print(f"    WARNING: Chart failed: {e}")

    print("\n" + "=" * 65)
    lit = sum(1 for h in handoff.values() if "literature" in h["source"])
    syn = sum(1 for h in handoff.values() if "synthetic" in h["source"])
    print(f"  Handoff cells — literature: {lit}/6  synthetic: {syn}/6")
    incr_lit = sum(1 for i in increment.values() if i["source"] == "literature")
    print(f"  Increment     — literature: {incr_lit}/3  synthetic: {3-incr_lit}/3")
    print("=" * 65)


HEADER_COLS = [
    "extraction_id","paper_id","authors","year","country","city",
    "urban_classification","network_type","stop_type","geography_cell",
    "data_collection_method","year_of_data","sample_size_stops",
    "service_time_definition","n_packages_reference",
    "total_stop_mean_sec","total_stop_median_sec","total_stop_sd_sec",
    "base_handoff_mean_sec","base_handoff_sd_sec",
    "walk_time_mean_sec","walk_time_sd_sec","walk_time_density_ref",
    "increment_per_pkg_mean_sec","increment_per_pkg_sd_sec","increment_model_form",
    "pkg_count_log_coef","pkg_count_log_coef_se",
    "density_effect_reported","density_exponent",
    "distribution_type_best_fit","lognorm_mu","lognorm_sigma",
    "gamma_alpha","gamma_beta","goodness_of_fit_stat","goodness_of_fit_value",
    "signature_premium_sec","signature_premium_sd_sec",
    "covariate_model_form","covariate_r_squared",
    "b2c_adjusted","b2c_adjustment_factor","au_relevant",
    "quality_score","data_status","notes"
]

if __name__ == "__main__":
    main()
