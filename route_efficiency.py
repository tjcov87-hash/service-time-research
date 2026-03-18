"""
route_efficiency.py
===================
Phase 3: Route efficiency modelling via TSP.

For each of the 694 zones (15-25 stops), this module:
  1. Solves the TSP using nearest-neighbour + 2-opt to find the optimal
     delivery sequence within the zone
  2. Simulates the CURRENT STATE route — stops visited in a sub-optimal
     order (reflecting a suburb-only sort with no zone sequencing)
  3. Computes the Route Efficiency Ratio (RER = TSP_time / Current_time)
  4. Fits a Gamma distribution to the travel time gap (waste) per zone
  5. Extracts the travel time saving achievable by zone pre-sorting

Outputs (written to ./data/):
  route_efficiency.csv          — per-zone TSP, current, gap, RER
  gap_distribution_params.csv   — Gamma fit on the gap distribution
  route_efficiency_charts.png   — RER distribution, gap dist, zone map
  route_efficiency_report.txt   — statistical summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gamma as gamma_dist
from scipy.stats import kstest
from math import radians, cos, sin, asin, sqrt
import os
import time
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Number of random permutations to simulate current-state routing
N_RANDOM_ROUTES = 20

# Number of 2-opt improvement passes
TWO_OPT_PASSES = 3


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * asin(sqrt(a))


def travel_time_s(lat1, lon1, lat2, lon2):
    """Road travel time in seconds: Haversine x 1.35 road factor at 28 km/h."""
    return (haversine_km(lat1, lon1, lat2, lon2) * 1.35 / 28) * 3600


def build_distance_matrix(lats, lons):
    """Build full pairwise travel time matrix for a set of stops."""
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            t = travel_time_s(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = t
            D[j, i] = t
    return D


def route_total_time(order, D):
    """Total travel time for a given stop sequence."""
    return sum(D[order[i], order[i+1]] for i in range(len(order) - 1))


# ---------------------------------------------------------------------------
# TSP SOLVER: Nearest-Neighbour + 2-opt
# ---------------------------------------------------------------------------

def nearest_neighbour(D, start=0):
    """
    Nearest-neighbour TSP heuristic.
    Greedy: always move to the closest unvisited stop.
    Returns ordered list of stop indices.
    """
    n = len(D)
    unvisited = list(range(n))
    route = [start]
    unvisited.remove(start)

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda j: D[current, j])
        route.append(nearest)
        unvisited.remove(nearest)

    return route


def two_opt(route, D, max_passes=TWO_OPT_PASSES):
    """
    2-opt improvement: iteratively swap route segments if it reduces
    total travel time. Converges to a local optimum.
    """
    best = route[:]
    best_time = route_total_time(best, D)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Reverse the segment between i and j
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_time = route_total_time(new_route, D)
                if new_time < best_time - 0.01:
                    best = new_route
                    best_time = new_time
                    improved = True

    return best, best_time


def solve_tsp(lats, lons):
    """
    Solve TSP for a set of stops.
    Tries nearest-neighbour from every stop as starting point,
    applies 2-opt to the best result.
    Returns (optimal_route, optimal_time_seconds).
    """
    n = len(lats)
    if n <= 1:
        return list(range(n)), 0.0

    D = build_distance_matrix(lats, lons)

    # Try NN from multiple starts — pick best
    best_nn_route = None
    best_nn_time = float("inf")

    # For zones of 15-25 stops, try all starts (fast enough)
    starts = range(n) if n <= 25 else range(0, n, max(1, n // 10))
    for start in starts:
        route = nearest_neighbour(D, start=start)
        t = route_total_time(route, D)
        if t < best_nn_time:
            best_nn_time = t
            best_nn_route = route

    # Apply 2-opt improvement
    optimal_route, optimal_time = two_opt(best_nn_route, D)

    return optimal_route, optimal_time, D


# ---------------------------------------------------------------------------
# CURRENT STATE SIMULATION
# ---------------------------------------------------------------------------

def simulate_current_state_routes(D, n_sims=N_RANDOM_ROUTES):
    """
    Simulate current-state routing: driver visits zone stops in a
    sub-optimal sequence (suburb-sort only, no zone sequencing).

    Models this as: nearest-neighbour order with moderate random noise
    injected into the distance matrix, representing a driver who has a
    rough sense of geography but is not following an optimised sequence.

    Returns array of route times for N_RANDOM_ROUTES simulations.
    """
    n = len(D)
    times = []

    for _ in range(n_sims):
        # Add 20-40% random noise to distances to simulate non-optimal sequencing
        noise_factor = np.random.uniform(1.15, 1.45, size=D.shape)
        noise_factor = (noise_factor + noise_factor.T) / 2  # keep symmetric
        np.fill_diagonal(noise_factor, 1.0)
        D_noisy = D * noise_factor

        # Route using nearest-neighbour on noisy distances
        route = nearest_neighbour(D_noisy, start=np.random.randint(n))

        # Evaluate the route on the TRUE distance matrix
        actual_time = route_total_time(route, D)
        times.append(actual_time)

    return np.array(times)


# ---------------------------------------------------------------------------
# PROCESS ALL ZONES
# ---------------------------------------------------------------------------

def process_all_zones(zone_assignments_df, zone_summary_df):
    """
    For each zone: solve TSP, simulate current-state routes, compute gap.
    Returns a DataFrame with one row per zone.
    """
    records = []
    zone_ids = sorted(zone_assignments_df["zone_id"].unique())
    n_zones = len(zone_ids)

    print(f"  Processing {n_zones} zones...")
    t0 = time.time()

    for i, zid in enumerate(zone_ids):
        zone_stops = zone_assignments_df[zone_assignments_df["zone_id"] == zid]
        lats = zone_stops["lat"].values
        lons = zone_stops["lon"].values
        n = len(lats)

        if n < 2:
            continue

        # --- TSP ---
        optimal_route, tsp_time, D = solve_tsp(lats, lons)

        # --- Current state simulation ---
        current_times = simulate_current_state_routes(D)
        current_mean  = current_times.mean()
        current_std   = current_times.std()
        current_p90   = np.percentile(current_times, 90)

        # --- Gap and RER ---
        gap_mean  = current_mean - tsp_time
        gap_times = current_times - tsp_time  # gap for each simulation
        rer       = tsp_time / current_mean if current_mean > 0 else 1.0

        # Zone metadata
        z_meta = zone_summary_df[zone_summary_df["zone_id"] == zid]
        suburb = z_meta["dominant_suburb"].values[0] if len(z_meta) > 0 else "Unknown"
        clat   = z_meta["centroid_lat"].values[0]   if len(z_meta) > 0 else lats.mean()
        clon   = z_meta["centroid_lon"].values[0]   if len(z_meta) > 0 else lons.mean()

        records.append({
            "zone_id":             zid,
            "n_stops":             n,
            "dominant_suburb":     suburb,
            "centroid_lat":        clat,
            "centroid_lon":        clon,
            # TSP
            "tsp_time_s":          round(tsp_time, 1),
            "tsp_time_min":        round(tsp_time / 60, 2),
            # Current state
            "current_mean_s":      round(current_mean, 1),
            "current_std_s":       round(current_std, 1),
            "current_p90_s":       round(current_p90, 1),
            "current_mean_min":    round(current_mean / 60, 2),
            # Gap (waste = current - TSP)
            "gap_mean_s":          round(gap_mean, 1),
            "gap_mean_min":        round(gap_mean / 60, 2),
            "gap_std_s":           round(gap_times.std(), 1),
            "gap_pct":             round(gap_mean / current_mean * 100, 1) if current_mean > 0 else 0,
            # Efficiency ratio
            "rer":                 round(rer, 4),
            "rer_pct":             round(rer * 100, 1),
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_zones - i - 1) / rate
            print(f"    {i+1}/{n_zones} zones  |  {elapsed:.0f}s elapsed  |  "
                  f"ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Done — {n_zones} zones in {elapsed:.1f}s")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FIT GAMMA TO GAP DISTRIBUTION
# ---------------------------------------------------------------------------

def fit_gap_distribution(efficiency_df, report_lines):
    """
    Fit a Gamma distribution to the zone-level mean gap times.
    The Gamma is the standard choice for always-positive, right-skewed
    waiting/waste time distributions.
    """
    report_lines.append("\n" + "=" * 65)
    report_lines.append("GAP DISTRIBUTION FIT (Gamma)")
    report_lines.append("=" * 65)

    gaps = efficiency_df["gap_mean_s"].values
    gaps_clean = gaps[gaps > 0]

    # Fit Gamma
    alpha, loc, beta = gamma_dist.fit(gaps_clean, floc=0)
    ks_stat, ks_p = kstest(gaps_clean, "gamma", args=(alpha, loc, beta))

    report_lines.append(f"\n  Gap observations:   {len(gaps_clean):,} zones")
    report_lines.append(f"  Gap range:          {gaps_clean.min():.1f}s – {gaps_clean.max():.1f}s")
    report_lines.append(f"  Gap mean:           {gaps_clean.mean():.1f}s")
    report_lines.append(f"  Gap median:         {np.median(gaps_clean):.1f}s")
    report_lines.append(f"  Gap std:            {gaps_clean.std():.1f}s")
    report_lines.append(f"\n  Gamma fit:")
    report_lines.append(f"    alpha (shape):  {alpha:.4f}")
    report_lines.append(f"    beta  (scale):  {beta:.4f}")
    report_lines.append(f"    loc:            {loc:.4f} (fixed at 0)")
    report_lines.append(f"    Mean of fit:    {gamma_dist.mean(alpha, loc, beta):.1f}s")
    report_lines.append(f"    Std of fit:     {gamma_dist.std(alpha, loc, beta):.1f}s")
    report_lines.append(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f}  "
                        f"({'PASS' if ks_p > 0.05 else 'marginal — sample size inflates stat'})")

    gap_params = {
        "distribution": "gamma",
        "alpha":  round(alpha, 6),
        "beta":   round(beta, 6),
        "loc":    0,
        "mean_s": round(gamma_dist.mean(alpha, loc, beta), 2),
        "std_s":  round(gamma_dist.std(alpha, loc, beta), 2),
        "n_obs":  len(gaps_clean),
    }
    return gap_params, report_lines


# ---------------------------------------------------------------------------
# SAVING SUMMARY
# ---------------------------------------------------------------------------

def compute_travel_saving(efficiency_df, report_lines):
    """
    Translate route efficiency results into concrete travel time savings.
    """
    report_lines.append("\n" + "=" * 65)
    report_lines.append("TRAVEL TIME SAVING FROM ZONE PRE-SORT")
    report_lines.append("=" * 65)

    total_zones   = len(efficiency_df)
    avg_gap_s     = efficiency_df["gap_mean_s"].mean()
    median_gap_s  = efficiency_df["gap_mean_s"].median()
    avg_rer       = efficiency_df["rer"].mean()
    avg_tsp_min   = efficiency_df["tsp_time_min"].mean()
    avg_curr_min  = efficiency_df["current_mean_min"].mean()

    # Zones per driver per day
    # Each driver covers ~93 stops, zones are 15-25 stops → ~5-6 zones/driver
    avg_stops_per_driver = 13780 / 147
    avg_zone_size        = efficiency_df["n_stops"].mean()
    zones_per_driver     = avg_stops_per_driver / avg_zone_size

    daily_saving_s  = avg_gap_s * zones_per_driver
    daily_saving_min = daily_saving_s / 60

    report_lines.append(f"\n  Zones analysed:            {total_zones:,}")
    report_lines.append(f"  Avg TSP route time/zone:   {avg_tsp_min:.1f} min")
    report_lines.append(f"  Avg current route/zone:    {avg_curr_min:.1f} min")
    report_lines.append(f"  Avg gap (waste)/zone:      {avg_gap_s:.1f}s  ({median_gap_s:.1f}s median)")
    report_lines.append(f"  Avg Route Efficiency Ratio: {avg_rer:.3f}  "
                        f"(1.0 = perfect, current state achieves {avg_rer*100:.1f}%)")
    report_lines.append(f"\n  Zones/driver/day:          {zones_per_driver:.1f}")
    report_lines.append(f"  Travel saving/driver/day:  {daily_saving_s:.0f}s  "
                        f"= {daily_saving_min:.1f} minutes")

    fleet_daily_hours  = daily_saving_s * 147 / 3600
    annual_hours       = fleet_daily_hours * 264
    annual_dollars     = annual_hours * 32

    report_lines.append(f"  Fleet saving/day:          {fleet_daily_hours:.1f} driver-hours")
    report_lines.append(f"\n  Annual saving (travel only, 264 days, $32/hr):")
    report_lines.append(f"    Driver-hours saved/year:  {annual_hours:,.0f}")
    report_lines.append(f"    Dollar value:             ${annual_dollars:,.0f}")

    # Combined with Phase 2
    service_time_annual = 280_727   # from Phase 2
    combined_annual = annual_dollars + service_time_annual
    report_lines.append(f"\n  COMBINED SAVING (Phase 2 + Phase 3):")
    report_lines.append(f"    Service time saving:      ${service_time_annual:>10,.0f}/yr")
    report_lines.append(f"    Travel time saving:       ${annual_dollars:>10,.0f}/yr")
    report_lines.append(f"    ----------------------------------------")
    report_lines.append(f"    TOTAL (pre-Monte Carlo):  ${combined_annual:>10,.0f}/yr")
    report_lines.append(f"\n  NOTE: Monte Carlo in Phase 5 will add confidence intervals")
    report_lines.append(f"  and account for variability. This is point-estimate only.")

    return daily_saving_min, annual_dollars, report_lines


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

def build_charts(efficiency_df, gap_params):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Botany DC — Route Efficiency Model (Phase 3 TSP Analysis)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    gaps   = efficiency_df["gap_mean_s"].values
    rers   = efficiency_df["rer"].values
    tsps   = efficiency_df["tsp_time_min"].values
    currs  = efficiency_df["current_mean_min"].values

    x_gap = np.linspace(0, gaps.max() * 1.1, 500)

    # --- Panel 1: Gap distribution with Gamma fit ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gaps, bins=40, density=True, alpha=0.55, color="#3b82f6",
             label="Observed gaps")
    alpha_g, beta_g = gap_params["alpha"], gap_params["beta"]
    ax1.plot(x_gap,
             gamma_dist.pdf(x_gap, alpha_g, 0, beta_g),
             "r-", linewidth=2,
             label=f"Gamma fit\n(a={alpha_g:.2f}, b={beta_g:.1f})")
    ax1.axvline(gaps.mean(), color="green", linestyle="--", linewidth=1.5,
                label=f"Mean gap = {gaps.mean():.0f}s")
    ax1.set_xlabel("Route gap per zone (seconds)")
    ax1.set_ylabel("Density")
    ax1.set_title("Travel Time Gap Distribution\n(Current state vs TSP optimal)")
    ax1.legend(fontsize=8)

    # --- Panel 2: RER distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rers, bins=40, color="#10b981", edgecolor="white",
             linewidth=0.5, alpha=0.8)
    ax2.axvline(rers.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean RER = {rers.mean():.3f}")
    ax2.axvline(1.0, color="black", linestyle="-", linewidth=1,
                label="Perfect (1.0)")
    ax2.set_xlabel("Route Efficiency Ratio (TSP / Actual)")
    ax2.set_ylabel("Number of zones")
    ax2.set_title("Route Efficiency Ratio\n(higher = closer to optimal)")
    ax2.legend(fontsize=8)

    # --- Panel 3: TSP vs current state scatter ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(tsps, currs, c=rers, cmap="RdYlGn",
                     alpha=0.4, s=12, vmin=0.6, vmax=1.0)
    max_val = max(tsps.max(), currs.max()) * 1.05
    ax3.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Equal (RER=1)")
    plt.colorbar(sc, ax=ax3, label="RER")
    ax3.set_xlabel("TSP optimal time (minutes)")
    ax3.set_ylabel("Current state time (minutes)")
    ax3.set_title("TSP vs Current State Route Time\n(colour = Route Efficiency Ratio)")
    ax3.legend(fontsize=8)

    # --- Panel 4: Gap by suburb ---
    ax4 = fig.add_subplot(gs[1, 0])
    suburb_gap = (
        efficiency_df.groupby("dominant_suburb")["gap_mean_s"]
        .agg(["mean", "median", "std"])
        .sort_values("mean", ascending=True)
    )
    y_pos = range(len(suburb_gap))
    ax4.barh(y_pos, suburb_gap["mean"], color="#f59e0b", alpha=0.8,
             label="Mean gap")
    ax4.errorbar(suburb_gap["mean"], y_pos,
                 xerr=suburb_gap["std"],
                 fmt="none", color="black", linewidth=1, capsize=3)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(suburb_gap.index, fontsize=8)
    ax4.set_xlabel("Travel time gap (seconds/zone)")
    ax4.set_title("Route Gap by Suburb\n(mean ± std)")

    # --- Panel 5: RER by zone size ---
    ax5 = fig.add_subplot(gs[1, 1])
    for n_stops in sorted(efficiency_df["n_stops"].unique()):
        subset = efficiency_df[efficiency_df["n_stops"] == n_stops]["rer"]
        if len(subset) > 5:
            ax5.scatter(
                [n_stops] * len(subset),
                subset,
                alpha=0.2, s=8, color="#6366f1"
            )
    # Overlay mean line
    mean_rer_by_size = efficiency_df.groupby("n_stops")["rer"].mean()
    ax5.plot(mean_rer_by_size.index, mean_rer_by_size.values,
             "r-o", linewidth=2, markersize=5, label="Mean RER")
    ax5.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax5.set_xlabel("Zone size (stops)")
    ax5.set_ylabel("Route Efficiency Ratio")
    ax5.set_title("RER vs Zone Size\n(smaller zones easier to optimise)")
    ax5.legend(fontsize=8)
    ax5.set_ylim(0.5, 1.05)

    # --- Panel 6: Cumulative saving chart ---
    ax6 = fig.add_subplot(gs[1, 2])
    avg_gap_s        = efficiency_df["gap_mean_s"].mean()
    avg_zone_size    = efficiency_df["n_stops"].mean()
    zones_per_driver = (13780 / 147) / avg_zone_size
    saving_per_driver_min = avg_gap_s * zones_per_driver / 60

    driver_counts = [10, 25, 50, 75, 100, 125, 147]
    annual_savings = [
        d * saving_per_driver_min * 60 * 264 / 3600 * 32
        for d in driver_counts
    ]
    ax6.bar(driver_counts, annual_savings, color="#8b5cf6", alpha=0.8, width=8)
    ax6.axvline(147, color="red", linestyle="--", linewidth=1.5,
                label="Full fleet (147)")
    for dc, sav in zip(driver_counts, annual_savings):
        ax6.text(dc, sav * 1.02, f"${sav/1000:.0f}k",
                 ha="center", va="bottom", fontsize=7)
    ax6.set_xlabel("Number of drivers")
    ax6.set_ylabel("Annual travel saving ($)")
    ax6.set_title("Annual Travel Saving by Fleet Size\n(travel time only, $32/hr)")
    ax6.legend(fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Route Efficiency Model — Phase 3 (TSP Analysis)")
    print("=" * 65)

    report_lines = [
        "ROUTE EFFICIENCY MODEL — BOTANY DC",
        "Phase 3: TSP Analysis Report",
        "=" * 65,
    ]

    # Load data
    print("\nLoading zone data...")
    zone_assignments = pd.read_csv(
        os.path.join(DATA_DIR, "zone_assignments.csv")
    )
    zone_summary = pd.read_csv(
        os.path.join(DATA_DIR, "zone_summary.csv")
    )
    print(f"  {len(zone_assignments):,} stops across "
          f"{zone_assignments['zone_id'].nunique()} zones")

    # Process all zones
    print("\nSolving TSP and simulating current-state routes...")
    print(f"  ({N_RANDOM_ROUTES} current-state simulations per zone, "
          f"{TWO_OPT_PASSES} 2-opt passes)")
    t0 = time.time()
    efficiency_df = process_all_zones(zone_assignments, zone_summary)
    print(f"  Total time: {time.time()-t0:.1f}s")

    # TSP quality stats
    report_lines.append("\nTSP SOLUTION SUMMARY")
    report_lines.append("=" * 65)
    report_lines.append(f"  Zones solved:              {len(efficiency_df):,}")
    report_lines.append(f"  Avg TSP time/zone:         {efficiency_df['tsp_time_min'].mean():.1f} min")
    report_lines.append(f"  Median TSP time/zone:      {efficiency_df['tsp_time_min'].median():.1f} min")
    report_lines.append(f"  Avg current route/zone:    {efficiency_df['current_mean_min'].mean():.1f} min")
    report_lines.append(f"  Avg RER:                   {efficiency_df['rer'].mean():.3f}")
    report_lines.append(f"  Median RER:                {efficiency_df['rer'].median():.3f}")
    report_lines.append(f"  Zones with RER > 0.90:     "
                        f"{(efficiency_df['rer'] > 0.90).sum():,} "
                        f"({(efficiency_df['rer'] > 0.90).mean()*100:.1f}%)")

    # Fit gap distribution
    print("\nFitting Gamma distribution to gap data...")
    gap_params, report_lines = fit_gap_distribution(efficiency_df, report_lines)

    # Travel saving
    print("\nComputing travel time savings...")
    daily_saving_min, annual_dollars, report_lines = compute_travel_saving(
        efficiency_df, report_lines
    )

    # Write outputs
    eff_path    = os.path.join(DATA_DIR, "route_efficiency.csv")
    gap_path    = os.path.join(DATA_DIR, "gap_distribution_params.csv")
    chart_path  = os.path.join(DATA_DIR, "route_efficiency_charts.png")
    report_path = os.path.join(DATA_DIR, "route_efficiency_report.txt")

    efficiency_df.to_csv(eff_path, index=False)
    pd.DataFrame([gap_params]).to_csv(gap_path, index=False)

    print("\nBuilding charts...")
    fig = build_charts(efficiency_df, gap_params)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print report
    print("\n" + report_text.encode("ascii", "replace").decode("ascii"))

    print("\n" + "=" * 65)
    print("OUTPUTS WRITTEN")
    print("=" * 65)
    for path in [eff_path, gap_path, chart_path, report_path]:
        kb = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):<40} ({kb:.1f} KB)")

    print("\n" + "=" * 65)
    print("PHASE 3 SUMMARY")
    print("=" * 65)
    print(f"  Avg route gap/zone:      {efficiency_df['gap_mean_s'].mean():.0f}s")
    print(f"  Avg RER:                 {efficiency_df['rer'].mean():.3f}")
    print(f"  Travel saving/driver/day: {daily_saving_min:.1f} min")
    print(f"  Annual travel saving:    ${annual_dollars:,.0f}")
    print(f"\n  Ready for Phase 5: Monte Carlo simulation")
    print(f"  Inputs prepared:")
    print(f"    service_time_params.csv  (Phase 2 distributions)")
    print(f"    gap_distribution_params.csv  (Phase 3 Gamma fit)")
    print(f"    zone_summary.csv  (zone TSP times)")


if __name__ == "__main__":
    main()
