"""
pre_route_model.py
==================
Models the pre-route time component: driver floor sort and van load time.

Current state: drivers sort their full run on the DC floor (up to 2 hours)
before loading their van (~30 min to load).

Zone sort:     DC pre-sorts to 15-25 stop zones in delivery sequence.
               No floor sort needed. Driver loads direct in ~25 min.

High-cap van:  DC sorts run to exact delivery sequence. No floor sort.
               But physical loading of a large van takes ~60 min.

This is now the DOMINANT saving component — far larger than the per-stop
savings modelled in Phase 2 and Phase 3.

Outputs (written to ./data/):
  pre_route_params.csv    — scenario parameters for Monte Carlo
  pre_route_charts.png    — floor sort distribution + scenario comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import truncnorm
import os

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Fleet constants ──────────────────────────────────────────────────────────
N_DRIVERS      = 147
OPERATING_DAYS = 264
DRIVER_COST_HR = 32.00

# ── Floor sort distribution (current state) ──────────────────────────────────
FLOOR_SORT_MEAN_MIN = 90.0    # mean minutes (centre of "up to 2 hrs")
FLOOR_SORT_STD_MIN  = 25.0    # standard deviation
FLOOR_SORT_MIN_MIN  = 30.0    # hard lower bound (can't load without any sort)
FLOOR_SORT_MAX_MIN  = 120.0   # hard upper bound ("up to 2 hours")

# ── Load times by scenario (minutes) ─────────────────────────────────────────
LOAD_TIME = {
    "current":   30.0,    # after floor sort, relatively straightforward
    "zone_sort": 25.0,    # small ordered batches, fast staging
    "high_cap":  60.0,    # large van, precise positioning required ("up to 1hr")
}
LOAD_TIME_STD_FRAC = 0.12   # 12% std as fraction of mean (variability in loading)


# ── Helpers ──────────────────────────────────────────────────────────────────

def annual_saving(delta_min: float) -> float:
    """Convert a per-driver-day minute saving to annual fleet dollar saving."""
    return delta_min / 60 * N_DRIVERS * OPERATING_DAYS * DRIVER_COST_HR


def build_truncnorm(mean, std, lo, hi):
    """Return a scipy truncnorm object and its true (truncated) mean/std."""
    a = (lo - mean) / std
    b = (hi - mean) / std
    dist = truncnorm(a=a, b=b, loc=mean, scale=std)
    return dist, a, b


# ── Main functions ────────────────────────────────────────────────────────────

def build_params_table():
    """
    Build the pre-route parameter table for all three scenarios.
    Returns a DataFrame consumed by monte_carlo.py.
    """
    dist, a, b = build_truncnorm(
        FLOOR_SORT_MEAN_MIN, FLOOR_SORT_STD_MIN,
        FLOOR_SORT_MIN_MIN, FLOOR_SORT_MAX_MIN
    )
    truncnorm_mean = dist.mean()
    truncnorm_std  = dist.std()

    rows = []
    for scenario in ["current", "zone_sort", "high_cap"]:
        if scenario == "current":
            fs_dist_type   = "truncnorm"
            fs_mean        = truncnorm_mean
            fs_std         = truncnorm_std
            fs_raw_mean    = FLOOR_SORT_MEAN_MIN
            fs_raw_std     = FLOOR_SORT_STD_MIN
            fs_min         = FLOOR_SORT_MIN_MIN
            fs_max         = FLOOR_SORT_MAX_MIN
            fs_a           = a
            fs_b           = b
        else:
            fs_dist_type   = "fixed"
            fs_mean        = 0.0
            fs_std         = 0.0
            fs_raw_mean    = 0.0
            fs_raw_std     = 0.0
            fs_min         = 0.0
            fs_max         = 0.0
            fs_a           = 0.0
            fs_b           = 0.0

        load_mean = LOAD_TIME[scenario]
        load_std  = load_mean * LOAD_TIME_STD_FRAC

        rows.append({
            "scenario":          scenario,
            "fs_dist_type":      fs_dist_type,
            "fs_mean_min":       round(fs_mean, 4),
            "fs_std_min":        round(fs_std, 4),
            "fs_raw_mean_min":   fs_raw_mean,
            "fs_raw_std_min":    fs_raw_std,
            "fs_min_min":        fs_min,
            "fs_max_min":        fs_max,
            "fs_truncnorm_a":    round(fs_a, 6),
            "fs_truncnorm_b":    round(fs_b, 6),
            "load_mean_min":     load_mean,
            "load_std_min":      round(load_std, 4),
            "total_pre_route_mean_min": round(fs_mean + load_mean, 4),
        })

    return pd.DataFrame(rows), dist, truncnorm_mean


def compute_saving_summary(params_df, truncnorm_mean, report_lines):
    """Print and return the pre-route saving breakdown by scenario."""
    report_lines.append("=" * 65)
    report_lines.append("PRE-ROUTE TIME MODEL — BOTANY DC")
    report_lines.append("=" * 65)

    current_total = params_df[params_df["scenario"] == "current"][
        "total_pre_route_mean_min"].values[0]

    report_lines.append(f"\nCurrent state pre-route time:")
    report_lines.append(f"  Floor sort (mean):   {truncnorm_mean:.1f} min  "
                        f"(truncated normal, range {FLOOR_SORT_MIN_MIN:.0f}–"
                        f"{FLOOR_SORT_MAX_MIN:.0f} min)")
    report_lines.append(f"  Load time:           {LOAD_TIME['current']:.0f} min")
    report_lines.append(f"  Total pre-route:     {current_total:.1f} min / driver / day")
    report_lines.append(f"  Annual cost:         ${annual_saving(-current_total):,.0f}"
                        f"  (before any deliveries start)")

    report_lines.append(f"\nScenario comparison:")
    report_lines.append(f"  {'Scenario':<20} {'Floor sort':>12} {'Load':>8} "
                        f"{'Total':>8} {'vs Current':>12} {'Annual saving':>14}")
    report_lines.append("  " + "-" * 76)

    for _, row in params_df.iterrows():
        delta = current_total - row["total_pre_route_mean_min"]
        ann   = annual_saving(delta)
        report_lines.append(
            f"  {row['scenario']:<20} "
            f"{row['fs_mean_min']:>10.1f}m "
            f"{row['load_mean_min']:>6.0f}m "
            f"{row['total_pre_route_mean_min']:>7.1f}m "
            f"{delta:>+10.1f}m "
            f"  ${ann:>12,.0f}/yr"
        )

    # Summary savings
    zs_row  = params_df[params_df["scenario"] == "zone_sort"].iloc[0]
    hc_row  = params_df[params_df["scenario"] == "high_cap"].iloc[0]

    zs_delta = current_total - zs_row["total_pre_route_mean_min"]
    hc_delta = current_total - hc_row["total_pre_route_mean_min"]
    zs_ann   = annual_saving(zs_delta)
    hc_ann   = annual_saving(hc_delta)
    zs_vs_hc = zs_ann - hc_ann   # positive = zone sort better

    report_lines.append(f"\nPre-route saving vs current state:")
    report_lines.append(f"  Zone sort:       ${zs_ann:>10,.0f}/yr  "
                        f"({zs_delta:.1f} min/driver/day saved)")
    report_lines.append(f"  High-cap van:    ${hc_ann:>10,.0f}/yr  "
                        f"({hc_delta:.1f} min/driver/day saved)")
    report_lines.append(f"\n  Note: High-cap van INCREASES load time by "
                        f"{LOAD_TIME['high_cap'] - LOAD_TIME['current']:.0f} min "
                        f"vs current state = -${annual_saving(-(LOAD_TIME['high_cap'] - LOAD_TIME['current'])):,.0f}/yr load penalty")
    report_lines.append(f"\nZone sort pre-route advantage over high-cap van:")
    report_lines.append(f"  Load time saving: {LOAD_TIME['high_cap'] - LOAD_TIME['zone_sort']:.0f} min/driver/day")
    report_lines.append(f"  Annual advantage: ${abs(zs_vs_hc):,.0f}/yr  "
                        f"(zone sort wins on pre-route alone)")

    return zs_ann, hc_ann, zs_vs_hc, report_lines


def build_charts(params_df, dist, truncnorm_mean):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Pre-Route Time Model — Botany DC\n"
        "Floor sort elimination is the dominant saving driver",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Floor sort distribution (current state) ──────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    x = np.linspace(FLOOR_SORT_MIN_MIN, FLOOR_SORT_MAX_MIN, 500)
    ax1.plot(x, dist.pdf(x), color="#ef4444", linewidth=2.5,
             label=f"Truncated normal\n(mean={truncnorm_mean:.0f} min, "
                   f"std={dist.std():.0f} min)")
    ax1.fill_between(x, dist.pdf(x), alpha=0.15, color="#ef4444")

    # P10, mean, P90
    p10 = dist.ppf(0.10)
    p90 = dist.ppf(0.90)
    for pval, col, lbl in [
        (p10,            "#f97316", f"P10 = {p10:.0f} min"),
        (truncnorm_mean, "#dc2626", f"Mean = {truncnorm_mean:.0f} min"),
        (p90,            "#f97316", f"P90 = {p90:.0f} min"),
    ]:
        ax1.axvline(pval, color=col, linestyle="--", linewidth=1.5, label=lbl)

    ax1.set_xlabel("Floor sort time (minutes)")
    ax1.set_ylabel("Probability density")
    ax1.set_title(
        "Current State: Driver Floor Sort Time Distribution\n"
        "('Up to 2 hours' — modelled as truncated normal)"
    )
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 135)

    # Annotation
    ax1.annotate(
        f"${annual_saving(truncnorm_mean)/1_000_000:.2f}M/yr wasted\n"
        f"before a single parcel is delivered",
        xy=(truncnorm_mean, dist.pdf(truncnorm_mean)),
        xytext=(truncnorm_mean + 15, dist.pdf(truncnorm_mean) * 0.8),
        fontsize=9, color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d"),
        bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5"),
    )

    # ── Panel 2: Pre-route time stacked by scenario ────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    scenarios  = ["Current\nstate", "High-cap\nvan", "Zone\npre-sort"]
    fs_times   = [
        params_df[params_df["scenario"] == "current"]["fs_mean_min"].values[0],
        0, 0
    ]
    load_times = [
        LOAD_TIME["current"],
        LOAD_TIME["high_cap"],
        LOAD_TIME["zone_sort"],
    ]
    x_pos = range(3)
    b1 = ax2.bar(x_pos, fs_times, color="#ef4444", alpha=0.85,
                 label="Floor sort time", width=0.5)
    b2 = ax2.bar(x_pos, load_times, bottom=fs_times, color="#3b82f6",
                 alpha=0.85, label="Load time", width=0.5)

    totals = [f + l for f, l in zip(fs_times, load_times)]
    for i, total in enumerate(totals):
        ax2.text(i, total + 1, f"{total:.0f} min",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.set_ylabel("Minutes per driver per day")
    ax2.set_title("Pre-route time\nby scenario")
    ax2.legend(fontsize=8)

    # ── Panel 3: Annual pre-route saving ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    current_total = params_df[params_df["scenario"] == "current"][
        "total_pre_route_mean_min"].values[0]
    deltas = {
        "Zone sort\nvs current": annual_saving(
            current_total - params_df[params_df["scenario"]=="zone_sort"][
                "total_pre_route_mean_min"].values[0]
        ),
        "High-cap van\nvs current": annual_saving(
            current_total - params_df[params_df["scenario"]=="high_cap"][
                "total_pre_route_mean_min"].values[0]
        ),
    }
    bars = ax3.bar(
        list(deltas.keys()), [v/1000 for v in deltas.values()],
        color=["#10b981", "#f59e0b"], alpha=0.85, width=0.5
    )
    for bar, val in zip(bars, deltas.values()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 8,
                 f"${val/1000:.0f}k/yr",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Annual saving ($000s)")
    ax3.set_title("Pre-route saving\nvs current state ($000s/yr)")
    ax3.set_ylim(0, max(deltas.values())/1000 * 1.2)

    # ── Panel 4: Combined savings - zone sort full waterfall ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    components = [
        "Floor sort\neliminated",
        "Load time\nsaving",
        "Service time\n(per-stop)",
        "Route\nefficiency",
        "TOTAL\nZone Sort",
    ]
    values = [
        annual_saving(truncnorm_mean),       # floor sort
        annual_saving(LOAD_TIME["current"] - LOAD_TIME["zone_sort"]),  # load
        281_000,                              # Phase 2
        61_000,                               # Phase 3
        None,                                 # computed below
    ]
    total = sum(v for v in values[:-1])
    values[-1] = total
    colours = ["#ef4444", "#3b82f6", "#6366f1", "#8b5cf6", "#10b981"]
    b = ax4.bar(components, [v/1000 for v in values],
                color=colours, alpha=0.85, width=0.6)
    for bar, val in zip(b, values):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 15,
                 f"${val/1000:.0f}k",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax4.set_ylabel("Annual saving ($000s)")
    ax4.set_title("Zone sort total saving\ndecomposed ($000s/yr)")
    ax4.tick_params(axis="x", labelsize=7)

    # ── Panel 5: Zone sort vs high-cap van ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    advantage_rows = {
        "Load time\nadvantage": annual_saving(LOAD_TIME["high_cap"] - LOAD_TIME["zone_sort"]),
        "Per-stop\nefficiency": 61_000,      # route/sequencing advantage
        "Capital\navoided*": 1_000_000,      # capex avoided (illustrative)
        "Lease\navoided/yr": 195_000,
    }
    colours2 = ["#10b981", "#6366f1", "#f59e0b", "#f97316"]
    b2 = ax5.bar(
        list(advantage_rows.keys()),
        [v/1000 for v in advantage_rows.values()],
        color=colours2, alpha=0.85, width=0.55
    )
    for bar, val in zip(b2, advantage_rows.values()):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 8,
                 f"${val/1000:.0f}k",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax5.set_ylabel("Advantage ($000s)")
    ax5.set_title("Zone sort advantage\nover high-cap van")
    ax5.tick_params(axis="x", labelsize=8)
    ax5.text(0.5, -0.18, "* One-time capex saving shown at $1M",
             transform=ax5.transAxes, ha="center", fontsize=7, color="gray")

    return fig


def main():
    print("=" * 65)
    print("Pre-Route Time Model")
    print("=" * 65)

    report_lines = []
    params_df, dist, truncnorm_mean = build_params_table()
    zs_ann, hc_ann, zs_vs_hc, report_lines = compute_saving_summary(
        params_df, truncnorm_mean, report_lines
    )

    # Write params
    params_path = os.path.join(DATA_DIR, "pre_route_params.csv")
    params_df.to_csv(params_path, index=False)

    # Charts
    print("\nBuilding charts...")
    fig = build_charts(params_df, dist, truncnorm_mean)
    chart_path = os.path.join(DATA_DIR, "pre_route_charts.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(DATA_DIR, "pre_route_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nOutputs: {params_path}")
    print(f"         {chart_path}")


if __name__ == "__main__":
    main()
