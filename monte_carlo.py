"""
monte_carlo.py  (v2 — three-scenario)
======================================
Phase 5: Monte Carlo simulation across three scenarios.

  Current state   — drivers sort run on DC floor (up to 2hr), load van (30 min),
                    deliver with sub-optimal sequencing (unsorted van)
  Zone sort       — DC pre-sorts to 15-25 stop zones in delivery order,
                    driver loads direct (25 min), follows zone sequence
  High-cap van    — DC sorts run to exact sequence, no floor sort,
                    but physical load of large van takes ~60 min

Simulation structure (per driver-day):
  total_time = floor_sort_time + load_time + on_road_time

  On-road time has two components modelled separately:
    service_time  — Phase 2 log-normal fits (sorted vs unsorted)
    travel_time   — Phase 3 TSP + Gamma gap (sorted removes gap)

Outputs (./data/):
  monte_carlo_scenarios.csv     — 10,000 sims × 3 scenarios
  business_case_summary_v2.csv  — percentile table
  scenario_comparison.csv       — three-way comparison point estimates
  sensitivity_results_v2.csv    — parameter sweep (zone sort vs current)
  monte_carlo_charts_v2.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import gamma as gamma_dist, lognorm, truncnorm as truncnorm_dist
import os, time, warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

N_SIMULATIONS   = 10_000
N_DRIVERS       = 147
OPERATING_DAYS  = 264
DRIVER_COST_HR  = 32.00


# ── 1. LOAD ALL PARAMETERS ───────────────────────────────────────────────────

def load_all_params():
    svc_params   = pd.read_csv(os.path.join(DATA_DIR, "service_time_params.csv"))
    gap_params   = pd.read_csv(os.path.join(DATA_DIR, "gap_distribution_params.csv")).iloc[0]
    zone_summary = pd.read_csv(os.path.join(DATA_DIR, "zone_summary.csv"))
    svc_log      = pd.read_csv(os.path.join(DATA_DIR, "service_time_log.csv"),
                               usecols=["stop_type", "service_class", "n_packages"])
    pre_route    = pd.read_csv(os.path.join(DATA_DIR, "pre_route_params.csv"))
    return svc_params, gap_params, zone_summary, svc_log, pre_route


def build_lookup_tables(svc_params, svc_log):
    param_lookup = {}
    for _, row in svc_params.iterrows():
        param_lookup[(row["stop_type"], row["service_class"], row["sort_state"])] = \
            (row["mu"], row["sigma"])

    mix = svc_log.groupby(["stop_type","service_class"]).size().reset_index(name="count")
    mix["prob"] = mix["count"] / mix["count"].sum()
    mix_probs = mix[["stop_type","service_class","prob"]].values

    pkg_counts = svc_log["n_packages"].value_counts(normalize=True).sort_index()
    return param_lookup, mix_probs, pkg_counts.index.tolist(), pkg_counts.values.tolist()


def build_pre_route_samplers(pre_route_df):
    """Return a dict of callables: scenario_name -> () -> (floor_sort_s, load_s)"""
    samplers = {}
    for _, row in pre_route_df.iterrows():
        s = row["scenario"]
        if row["fs_dist_type"] == "truncnorm":
            a, b = row["fs_truncnorm_a"], row["fs_truncnorm_b"]
            fs_mean, fs_std = row["fs_raw_mean_min"], row["fs_raw_std_min"]
            fs_dist = truncnorm_dist(a=a, b=b, loc=fs_mean, scale=fs_std)
            def make_fs_sampler(d):
                return lambda: d.rvs() * 60
            fs_sampler = make_fs_sampler(fs_dist)
        else:
            fs_sampler = lambda: 0.0

        load_mean = row["load_mean_min"] * 60
        load_std  = row["load_std_min"]  * 60
        lo_load   = max(0, load_mean - 3 * load_std)
        a_l = (lo_load - load_mean) / max(load_std, 0.01)
        b_l = 99.0
        load_dist = truncnorm_dist(a=a_l, b=b_l, loc=load_mean, scale=load_std)
        def make_load_sampler(d):
            return lambda: d.rvs()
        load_sampler = make_load_sampler(load_dist)

        samplers[s] = (fs_sampler, load_sampler)
    return samplers


# ── 2. ON-ROAD SIMULATION (unchanged from Phase 2/3 logic) ──────────────────

def simulate_on_road(
    zone_tsp_times, zone_sizes, param_lookup, mix_probs,
    pkg_values, pkg_probs, gap_alpha, gap_beta,
    stops_per_driver, sort_state_future="sorted",
):
    """
    Simulate one driver's on-road day.
    Returns (current_on_road_s, future_on_road_s).
    current = unsorted service times + gap travel
    future  = sorted service times + TSP-only travel
    """
    current_s = 0.0
    future_s  = 0.0
    stops_covered = 0

    while stops_covered < stops_per_driver:
        zone_idx = np.random.randint(len(zone_tsp_times))
        tsp_time = zone_tsp_times[zone_idx]
        n_stops  = min(int(zone_sizes[zone_idx]), stops_per_driver - stops_covered)
        if n_stops < 1:
            break

        gap              = gamma_dist.rvs(gap_alpha, scale=gap_beta)
        current_s       += tsp_time + gap          # current: TSP + waste
        future_s        += tsp_time                # future: TSP only

        for _ in range(n_stops):
            idx       = np.random.choice(len(mix_probs), p=mix_probs[:,2].astype(float))
            stop_type = mix_probs[idx, 0]
            svc_class = mix_probs[idx, 1]
            n_pkg     = np.random.choice(pkg_values, p=pkg_probs)
            pkg_log   = np.log(1.0 + (n_pkg - 1) * 0.60)

            mu_u, sig_u = param_lookup.get((stop_type, svc_class, "unsorted"), (4.0, 0.35))
            mu_s, sig_s = param_lookup.get((stop_type, svc_class, "sorted"),   (3.88, 0.30))
            current_s += np.random.lognormal(mu_u + pkg_log, sig_u)
            future_s  += np.random.lognormal(mu_s + pkg_log, sig_s)

        stops_covered += n_stops

    return current_s, future_s


# ── 3. FULL SIMULATION LOOP ──────────────────────────────────────────────────

def run_all_scenarios(
    zone_tsp_times, zone_sizes, param_lookup, mix_probs,
    pkg_values, pkg_probs, gap_alpha, gap_beta,
    pre_route_samplers, stops_per_driver=93, n_sims=N_SIMULATIONS,
):
    """
    Run n_sims driver-day simulations.
    For each sim:
      - Draw current and scenario pre-route times
      - Simulate on-road saving once (shared across scenarios)
      - Total saving = pre-route delta + on-road saving
    Returns DataFrame with all three scenario results.
    """
    records = []
    t0 = time.time()

    for i in range(n_sims):
        # Shared on-road simulation
        c_road, f_road = simulate_on_road(
            zone_tsp_times, zone_sizes, param_lookup, mix_probs,
            pkg_values, pkg_probs, gap_alpha, gap_beta, stops_per_driver
        )
        on_road_saving_s = c_road - f_road

        # Current state pre-route
        cur_fs_sampler, cur_load_sampler = pre_route_samplers["current"]
        cur_pre_route_s = cur_fs_sampler() + cur_load_sampler()

        # Zone sort pre-route
        zs_fs_sampler, zs_load_sampler = pre_route_samplers["zone_sort"]
        zs_pre_route_s = zs_fs_sampler() + zs_load_sampler()

        # High-cap van pre-route
        hc_fs_sampler, hc_load_sampler = pre_route_samplers["high_cap"]
        hc_pre_route_s = hc_fs_sampler() + hc_load_sampler()

        # Zone sort vs current
        zs_saving_s = (cur_pre_route_s - zs_pre_route_s) + on_road_saving_s

        # High-cap vs current (high-cap on-road ~ same as zone sort since both sorted)
        # High-cap per-stop saving is same as zone_sort (both sorted); travel slightly
        # less efficient (run-level vs zone-level) — approximate as 60% of on-road saving
        hc_road_saving_s = on_road_saving_s * 0.60
        hc_saving_s = (cur_pre_route_s - hc_pre_route_s) + hc_road_saving_s

        records.append({
            "sim_id":                 i + 1,
            "on_road_saving_s":       round(on_road_saving_s, 1),
            # Zone sort
            "zs_pre_route_saving_s":  round(cur_pre_route_s - zs_pre_route_s, 1),
            "zs_total_saving_s":      round(zs_saving_s, 1),
            "zs_saving_min":          round(zs_saving_s / 60, 2),
            # High-cap van
            "hc_pre_route_saving_s":  round(cur_pre_route_s - hc_pre_route_s, 1),
            "hc_total_saving_s":      round(hc_saving_s, 1),
            "hc_saving_min":          round(hc_saving_s / 60, 2),
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1:,}/{n_sims:,}  |  {elapsed:.0f}s  |  "
                  f"ETA {(n_sims - i - 1)/rate:.0f}s")

    print(f"  Done — {n_sims:,} sims in {time.time()-t0:.1f}s")
    return pd.DataFrame(records)


# ── 4. BUSINESS CASE METRICS ─────────────────────────────────────────────────

def compute_business_case(savings_min, label, lease_cost_yr=0):
    """Compute percentile table + annual fleet saving."""
    def ann(m): return m * N_DRIVERS * OPERATING_DAYS / 60 * DRIVER_COST_HR - lease_cost_yr
    return {
        "scenario":            label,
        "mean_min":            round(savings_min.mean(), 1),
        "std_min":             round(savings_min.std(), 1),
        "p05_min":             round(np.percentile(savings_min, 5), 1),
        "p50_min":             round(np.percentile(savings_min, 50), 1),
        "p95_min":             round(np.percentile(savings_min, 95), 1),
        "annual_mean_$":       round(ann(savings_min.mean())),
        "annual_p05_$":        round(ann(np.percentile(savings_min, 5))),
        "annual_p50_$":        round(ann(np.percentile(savings_min, 50))),
        "annual_p95_$":        round(ann(np.percentile(savings_min, 95))),
        "prob_positive_%":     round((savings_min > 0).mean() * 100, 1),
        "lease_cost_yr_$":     lease_cost_yr,
    }


# ── 5. SENSITIVITY (zone sort vs current only) ───────────────────────────────

def run_sensitivity(
    zone_tsp_times, zone_sizes, param_lookup, mix_probs,
    pkg_values, pkg_probs, gap_alpha, gap_beta,
    pre_route_samplers, n_sims=1500, stops_base=93,
):
    base_cost  = DRIVER_COST_HR
    base_days  = OPERATING_DAYS
    base_fleet = N_DRIVERS
    cases = [
        ("Driver cost ($/hr)",       24,  40,  "cost"),
        ("Operating days/yr",        240, 280, "days"),
        ("Fleet size (drivers)",     110, 180, "fleet"),
        ("Floor sort mean (min)",    70,  110, "floor"),
        ("Zone sort load time (min)",20,  35,  "zs_load"),
        ("Stops/driver/day",         75,  115, "stops"),
    ]
    records = []
    for param, lo_val, hi_val, what in cases:
        for val, direction in [(lo_val, "low"), (hi_val, "high")]:
            # Override pre-route sampler if floor/load parameter is varying
            local_samplers = dict(pre_route_samplers)
            if what == "floor":
                a = (30 - val) / 25.0
                b = (120 - val) / 25.0
                d = truncnorm_dist(a=a, b=b, loc=val, scale=25)
                _, cur_load = pre_route_samplers["current"]
                local_samplers["current"] = (lambda d=d: d.rvs()*60, cur_load)
            if what == "zs_load":
                zs_load_s = val * 60
                zs_std    = zs_load_s * 0.12
                a_l = max(-3, -zs_load_s/max(zs_std,1))
                zs_load_dist = truncnorm_dist(a=a_l, b=99, loc=zs_load_s, scale=zs_std)
                _, old_zs_load = pre_route_samplers["zone_sort"]
                local_samplers["zone_sort"] = (lambda: 0.0,
                                               lambda d=zs_load_dist: d.rvs())

            stops = int(val) if what == "stops" else stops_base
            results = run_all_scenarios(
                zone_tsp_times, zone_sizes, param_lookup, mix_probs,
                pkg_values, pkg_probs, gap_alpha, gap_beta,
                local_samplers, stops, n_sims
            )
            mean_min = results["zs_saving_min"].mean()
            cost   = val  if what == "cost"  else base_cost
            days   = int(val) if what == "days"  else base_days
            fleet  = int(val) if what == "fleet" else base_fleet
            annual = mean_min * fleet * days / 60 * cost
            records.append({
                "parameter": param, "value": val, "direction": direction,
                "mean_saving_min": round(mean_min, 1),
                "annual_saving_$": round(annual),
            })

    df = pd.DataFrame(records)
    swings = []
    for p in df["parameter"].unique():
        sub = df[df["parameter"] == p]
        lo = sub[sub["direction"]=="low"]["annual_saving_$"].values[0]
        hi = sub[sub["direction"]=="high"]["annual_saving_$"].values[0]
        swings.append({"parameter": p, "low_$": lo, "high_$": hi,
                       "swing_$": abs(hi - lo)})
    swing_df = pd.DataFrame(swings).sort_values("swing_$", ascending=True)
    return df, swing_df


# ── 6. CHARTS ────────────────────────────────────────────────────────────────

def build_charts(results_df, biz_zs, biz_hc, swing_df, base_annual):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Botany DC — Zone Sort Monte Carlo (v2: Three-Scenario, Pre-Route Included)\n"
        f"10,000 simulations  |  {N_DRIVERS} drivers  |  "
        f"{OPERATING_DAYS} operating days  |  ${DRIVER_COST_HR}/hr",
        fontsize=12, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    zs_min  = results_df["zs_saving_min"].values
    hc_min  = results_df["hc_saving_min"].values
    zs_ann  = zs_min * N_DRIVERS * OPERATING_DAYS / 60 * DRIVER_COST_HR
    hc_ann  = hc_min * N_DRIVERS * OPERATING_DAYS / 60 * DRIVER_COST_HR
    hc_ann_net = hc_ann - 195_000   # minus lease cost

    # ── P1: Zone sort per-driver saving distribution ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(zs_min, bins=70, density=True, alpha=0.65, color="#10b981",
             label="Zone sort")
    ax1.hist(hc_min, bins=70, density=True, alpha=0.45, color="#f59e0b",
             label="High-cap van")
    for pct, col, ls in [(5, "#f97316","--"), (50, "#10b981","-"), (95, "#f97316","--")]:
        ax1.axvline(np.percentile(zs_min, pct), color=col, linestyle=ls,
                    linewidth=1.5, label=f"ZS P{pct}={np.percentile(zs_min,pct):.0f}m")
    ax1.set_xlabel("Saving per driver per day (min)")
    ax1.set_ylabel("Density")
    ax1.set_title("Saving Distribution\n(per driver/day, both scenarios)")
    ax1.legend(fontsize=7)

    # ── P2: Annual saving — three-scenario overlay ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(zs_ann/1000,  bins=60, alpha=0.55, color="#10b981", label="Zone sort")
    ax2.hist(hc_ann_net/1000, bins=60, alpha=0.45, color="#f59e0b",
             label="High-cap (net of $195k lease)")
    for pct, col, ls in [
        (5,  "#f97316","--"), (50, "#10b981","-"), (95, "#f97316","--")]:
        ax2.axvline(np.percentile(zs_ann,pct)/1000, color=col, linestyle=ls,
                    linewidth=1.5)
    ax2.set_xlabel("Annual saving ($000s)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Annual Saving Distribution\n(fleet, $000s)")
    ax2.legend(fontsize=8)

    # ── P3: Business case waterfall ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    pre_route_zs = biz_zs["annual_mean_$"] - 342_000
    labels  = ["Floor sort\nelim.", "Load time\ndelta", "On-road\nsavings",
               "TOTAL\nZone Sort", "High-cap\n(net)", "Zone sort\nadvantage"]
    floor_sv = pre_route_zs - (DRIVER_COST_HR*N_DRIVERS*OPERATING_DAYS/60*5)
    load_sv  = DRIVER_COST_HR*N_DRIVERS*OPERATING_DAYS/60*5
    vals = [floor_sv, load_sv, 342_000,
            biz_zs["annual_mean_$"],
            biz_hc["annual_mean_$"] - 195_000,
            biz_zs["annual_mean_$"] - (biz_hc["annual_mean_$"] - 195_000)]
    colours = ["#ef4444","#3b82f6","#6366f1","#10b981","#f59e0b","#8b5cf6"]
    b = ax3.bar(labels, [v/1000 for v in vals], color=colours, alpha=0.85)
    for bar, v in zip(b, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                 f"${v/1000:.0f}k", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax3.set_ylabel("Annual saving ($000s)")
    ax3.set_title("Business case waterfall\n($000s/yr)")
    ax3.tick_params(axis="x", labelsize=7)

    # ── P4: Saving decomposition (zone sort) ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    zs_pr_min  = results_df["zs_pre_route_saving_s"].values / 60
    or_min     = results_df["on_road_saving_s"].values / 60
    ax4.scatter(zs_pr_min, or_min, alpha=0.08, s=6, color="#10b981")
    ax4.axhline(or_min.mean(), color="#6366f1", linestyle="--", linewidth=1.5,
                label=f"On-road mean = {or_min.mean():.1f} min")
    ax4.axvline(zs_pr_min.mean(), color="#ef4444", linestyle="--", linewidth=1.5,
                label=f"Pre-route mean = {zs_pr_min.mean():.1f} min")
    ax4.set_xlabel("Pre-route saving (min) — floor sort + load")
    ax4.set_ylabel("On-road saving (min) — service + travel")
    ax4.set_title("Saving decomposition\n(pre-route dominates)")
    ax4.legend(fontsize=8)

    # ── P5: Cumulative confidence curve ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_zs = np.sort(zs_ann)/1000
    sorted_hc = np.sort(hc_ann_net)/1000
    cum = np.arange(1, len(sorted_zs)+1) / len(sorted_zs) * 100
    ax5.plot(sorted_zs, cum, color="#10b981", linewidth=2, label="Zone sort")
    ax5.plot(sorted_hc, cum, color="#f59e0b", linewidth=2, linestyle="--",
             label="High-cap (net)")
    ax5.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax5.set_xlabel("Annual saving ($000s)")
    ax5.set_ylabel("Probability (%)")
    ax5.set_title("Confidence curve\n(probability of achieving >= X)")
    ax5.legend(fontsize=9)
    ax5.invert_xaxis(); ax5.invert_xaxis()

    # ── P6: Pre-route vs high-cap comparison bar ──────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ["Current\nstate", "High-cap\n(net)", "Zone\nsort"]
    annual_vals = [0, biz_hc["annual_mean_$"] - 195_000, biz_zs["annual_mean_$"]]
    bar_cols = ["#9ca3af", "#f59e0b", "#10b981"]
    b6 = ax6.bar(categories, [v/1000 for v in annual_vals], color=bar_cols, alpha=0.85)
    for bar, v in zip(b6, annual_vals):
        if v > 0:
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                     f"${v/1000:.0f}k", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    ax6.set_ylabel("Annual saving vs current ($000s)")
    ax6.set_title("Three-scenario comparison\n(annual, fleet-wide)")
    ax6.annotate(
        f"Zone sort wins\nby ${(biz_zs['annual_mean_$']-(biz_hc['annual_mean_$']-195_000))/1000:.0f}k/yr\n+ avoids $1M capex",
        xy=(2, biz_zs["annual_mean_$"]/1000),
        xytext=(1.2, biz_zs["annual_mean_$"]/1000 * 0.7),
        fontsize=8, color="#065f46",
        arrowprops=dict(arrowstyle="->", color="#065f46"),
        bbox=dict(boxstyle="round", fc="#d1fae5", ec="#6ee7b7"),
    )

    # ── P7+8+9: Tornado chart ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    base_k = base_annual / 1000
    params   = swing_df["parameter"].tolist()
    y_pos    = range(len(params))
    for i, p in enumerate(params):
        lo = swing_df[swing_df["parameter"]==p]["low_$"].values[0] / 1000
        hi = swing_df[swing_df["parameter"]==p]["high_$"].values[0] / 1000
        col_lo = "#ef4444" if lo < base_k else "#10b981"
        col_hi = "#10b981" if hi > base_k else "#ef4444"
        ax7.barh(i, lo - base_k, left=base_k, color=col_lo, alpha=0.75, height=0.6)
        ax7.barh(i, hi - base_k, left=base_k, color=col_hi, alpha=0.75, height=0.6)
        ax7.text(lo - 20, i, f"${lo:.0f}k", ha="right", va="center", fontsize=8)
        ax7.text(hi + 20, i, f"${hi:.0f}k", ha="left",  va="center", fontsize=8)
    ax7.axvline(base_k, color="black", linewidth=1.5,
                label=f"Base case = ${base_k:.0f}k/yr")
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(params, fontsize=9)
    ax7.set_xlabel("Annual saving ($000s)")
    ax7.set_title(f"Sensitivity Tornado — Zone Sort vs Current\n"
                  f"Base = ${base_k:.0f}k/yr")
    red_p   = mpatches.Patch(color="#ef4444", alpha=0.75, label="Reduces saving")
    green_p = mpatches.Patch(color="#10b981", alpha=0.75, label="Increases saving")
    ax7.legend(handles=[red_p, green_p], fontsize=9, loc="lower right")

    return fig


# ── 7. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Monte Carlo v2 — Three-Scenario")
    print(f"  {N_SIMULATIONS:,} sims  |  {N_DRIVERS} drivers  |  "
          f"{OPERATING_DAYS} days  |  ${DRIVER_COST_HR}/hr")
    print("=" * 65)

    print("\nLoading parameters...")
    svc_params, gap_params_row, zone_summary, svc_log, pre_route_df = load_all_params()
    param_lookup, mix_probs, pkg_values, pkg_probs = build_lookup_tables(svc_params, svc_log)
    pre_route_samplers = build_pre_route_samplers(pre_route_df)

    zone_tsp_times = zone_summary["tsp_estimate_s"].values
    zone_sizes     = zone_summary["n_stops"].values
    gap_alpha      = float(gap_params_row["alpha"])
    gap_beta       = float(gap_params_row["beta"])
    stops_per_driver = int(13780 / N_DRIVERS)

    print(f"  Pre-route floor sort mean: "
          f"{pre_route_df[pre_route_df['scenario']=='current']['fs_mean_min'].values[0]:.1f} min")
    print(f"  Load times — current: "
          f"{pre_route_df[pre_route_df['scenario']=='current']['load_mean_min'].values[0]:.0f} min  |  "
          f"zone sort: "
          f"{pre_route_df[pre_route_df['scenario']=='zone_sort']['load_mean_min'].values[0]:.0f} min  |  "
          f"high-cap: "
          f"{pre_route_df[pre_route_df['scenario']=='high_cap']['load_mean_min'].values[0]:.0f} min")

    # ── Base case ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Running base case ({N_SIMULATIONS:,} simulations)...")
    results_df = run_all_scenarios(
        zone_tsp_times, zone_sizes, param_lookup, mix_probs,
        pkg_values, pkg_probs, gap_alpha, gap_beta,
        pre_route_samplers, stops_per_driver, N_SIMULATIONS,
    )

    biz_zs = compute_business_case(results_df["zs_saving_min"], "zone_sort")
    biz_hc = compute_business_case(results_df["hc_saving_min"], "high_cap_van",
                                   lease_cost_yr=195_000)

    # ── Sensitivity ────────────────────────────────────────────────────────
    print("\n[2/3] Running sensitivity (6 params × 2 levels)...")
    sensitivity_df, swing_df = run_sensitivity(
        zone_tsp_times, zone_sizes, param_lookup, mix_probs,
        pkg_values, pkg_probs, gap_alpha, gap_beta,
        pre_route_samplers, n_sims=1500,
    )
    base_annual = biz_zs["annual_mean_$"]

    # ── Write outputs ──────────────────────────────────────────────────────
    print("\n[3/3] Writing outputs and building charts...")
    results_df.to_csv(os.path.join(DATA_DIR, "monte_carlo_scenarios.csv"), index=False)
    pd.DataFrame([biz_zs, biz_hc]).to_csv(
        os.path.join(DATA_DIR, "business_case_summary_v2.csv"), index=False)
    sensitivity_df.to_csv(
        os.path.join(DATA_DIR, "sensitivity_results_v2.csv"), index=False)

    fig = build_charts(results_df, biz_zs, biz_hc, swing_df, base_annual)
    chart_path = os.path.join(DATA_DIR, "monte_carlo_charts_v2.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Print results ──────────────────────────────────────────────────────
    pre_route_row = pre_route_df[pre_route_df["scenario"]=="current"]
    fs_mean  = pre_route_row["fs_mean_min"].values[0]
    load_cur = pre_route_df[pre_route_df["scenario"]=="current"]["load_mean_min"].values[0]
    load_zs  = pre_route_df[pre_route_df["scenario"]=="zone_sort"]["load_mean_min"].values[0]
    load_hc  = pre_route_df[pre_route_df["scenario"]=="high_cap"]["load_mean_min"].values[0]

    floor_ann  = fs_mean / 60 * N_DRIVERS * OPERATING_DAYS * DRIVER_COST_HR
    load_zs_ann = (load_cur - load_zs) / 60 * N_DRIVERS * OPERATING_DAYS * DRIVER_COST_HR
    on_road_ann = 342_000

    print("\n" + "=" * 65)
    print("RESULTS — ZONE SORT VS CURRENT STATE")
    print("=" * 65)
    print(f"  Per driver per day (minutes saved):")
    print(f"    Mean:  {biz_zs['mean_min']:>7.1f} min")
    print(f"    P05:   {biz_zs['p05_min']:>7.1f} min   (downside)")
    print(f"    P50:   {biz_zs['p50_min']:>7.1f} min   (median)")
    print(f"    P95:   {biz_zs['p95_min']:>7.1f} min   (upside)")
    print(f"    Prob. positive: {biz_zs['prob_positive_%']}%")

    print(f"\n  Annual fleet saving — decomposed:")
    print(f"    Floor sort eliminated:    ${floor_ann:>10,.0f}/yr")
    print(f"    Load time saving:         ${load_zs_ann:>10,.0f}/yr")
    print(f"    On-road (Phase 2+3):      ${on_road_ann:>10,.0f}/yr")
    print(f"    {'-'*40}")
    print(f"    Bottom-up total:          ${floor_ann+load_zs_ann+on_road_ann:>10,.0f}/yr")
    print(f"    Monte Carlo mean:         ${biz_zs['annual_mean_$']:>10,.0f}/yr")
    print(f"    90% CI:      ${biz_zs['annual_p05_$']:>8,.0f} – ${biz_zs['annual_p95_$']:>8,.0f}/yr")

    print(f"\n{'='*65}")
    print("THREE-SCENARIO COMPARISON")
    print(f"{'='*65}")
    print(f"  {'':30} {'Zone Sort':>12} {'High-Cap Van':>14}")
    print(f"  {'-'*56}")
    print(f"  {'Floor sort saving':30} ${floor_ann:>10,.0f}   ${floor_ann:>10,.0f}")
    load_hc_ann = -(load_hc - load_cur)/60 * N_DRIVERS * OPERATING_DAYS * DRIVER_COST_HR
    print(f"  {'Load time delta':30} ${load_zs_ann:>+10,.0f}   ${load_hc_ann:>+10,.0f}")
    print(f"  {'On-road savings':30} ${on_road_ann:>10,.0f}   ${on_road_ann*0.6:>10,.0f}")
    print(f"  {'-'*56}")
    print(f"  {'Operational total':30} ${floor_ann+load_zs_ann+on_road_ann:>10,.0f}   "
          f"${floor_ann+load_hc_ann+on_road_ann*0.6:>10,.0f}")
    print(f"  {'Lease cost (high-cap)':30} {'n/a':>12}   ${'195,000':>12}")
    print(f"  {'-'*56}")
    print(f"  {'NET annual benefit':30} ${biz_zs['annual_mean_$']:>10,.0f}   "
          f"${biz_hc['annual_mean_$']:>10,.0f}")
    print(f"\n  Zone sort advantage vs high-cap:")
    advantage = biz_zs['annual_mean_$'] - biz_hc['annual_mean_$']
    print(f"    Operational: ${advantage:,.0f}/yr  +  avoids ~$1M+ capex")
    print(f"\n  Sensitivity — top drivers of uncertainty:")
    for _, row in swing_df.sort_values("swing_$", ascending=False).iterrows():
        print(f"    {row['parameter']:<35}  swing = ${row['swing_$']:>9,.0f}")


if __name__ == "__main__":
    main()
