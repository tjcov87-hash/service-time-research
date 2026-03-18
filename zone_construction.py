"""
zone_construction.py
====================
Phase 1: Constrained spatial clustering for delivery zone design.

Partitions 13,780 stops into zones of 15-25 stops each, minimising
intra-zone travel time (spatial compactness).

Algorithm:
  1. K-Means seeding  — fast initial partition into ~700 zones
  2. Size enforcement — split oversized zones, merge undersized zones
  3. Refinement       — reassign boundary stops to improve compactness
  4. Quality analysis — intra-zone travel stats, zone stability metrics
  5. Visualisation    — interactive folium map + matplotlib charts

Outputs (written to ./data/):
  zone_assignments.csv   — stop_id → zone_id
  zone_summary.csv       — per-zone stats (size, centroid, avg intra-travel)
  zone_map.html          — interactive map (open in browser)
  zone_charts.png        — distribution charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import folium
import os
import time
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Reproducibility
np.random.seed(42)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = DATA_DIR   # write outputs alongside input data

# Zone size constraints
ZONE_MIN = 15
ZONE_MAX = 25
ZONE_TARGET = 19   # ideal size — used to set initial k

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * asin(sqrt(a))


def travel_time_seconds(lat1, lon1, lat2, lon2):
    """Estimate road travel time: Haversine × 1.35 road factor at 28km/h urban."""
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return (dist_km * 1.35 / 28) * 3600


def zone_centroid(df_zone):
    """Weighted centroid of a zone (simple mean lat/lon)."""
    return df_zone["lat"].mean(), df_zone["lon"].mean()


def intra_zone_travel_stats(df_zone):
    """
    Compute intra-zone travel statistics for a set of stops.
    Samples pairwise travel times (not full O(n²) — uses a representative sample).
    Returns: mean_tt, median_tt, max_tt, total_tsp_estimate
    """
    n = len(df_zone)
    if n < 2:
        return {"mean_tt": 0, "median_tt": 0, "max_tt": 0, "tsp_estimate_s": 0}

    lats = df_zone["lat"].values
    lons = df_zone["lon"].values

    # Sample up to 50 pairs (enough to characterise distribution without full O(n²))
    pairs = min(n * (n-1) // 2, 50)
    tts = []
    for _ in range(pairs):
        i, j = np.random.choice(n, 2, replace=False)
        tts.append(travel_time_seconds(lats[i], lons[i], lats[j], lons[j]))

    # TSP estimate: nearest-neighbour heuristic travel through all stops
    # Approximation: sum of nearest-neighbour distances from centroid
    clat, clon = lats.mean(), lons.mean()
    unvisited = list(range(n))
    cur_lat, cur_lon = clat, clon
    tsp_time = 0
    while unvisited:
        dists = [travel_time_seconds(cur_lat, cur_lon, lats[i], lons[i]) for i in unvisited]
        nearest_idx = unvisited[np.argmin(dists)]
        tsp_time += min(dists)
        cur_lat, cur_lon = lats[nearest_idx], lons[nearest_idx]
        unvisited.remove(nearest_idx)

    return {
        "mean_tt_s":    round(np.mean(tts), 1),
        "median_tt_s":  round(np.median(tts), 1),
        "max_tt_s":     round(np.max(tts), 1),
        "tsp_estimate_s": round(tsp_time, 1),
    }


# ---------------------------------------------------------------------------
# STEP 1: INITIAL K-MEANS PARTITION
# ---------------------------------------------------------------------------

def initial_kmeans(stops_df):
    """
    Run K-Means on lat/lon coords to create initial zone seeds.
    k = n_stops // ZONE_TARGET gives ~19 stops per zone on average.
    Uses MiniBatchKMeans for speed at ~14k stops.
    """
    k = len(stops_df) // ZONE_TARGET
    print(f"  K-Means: k={k} clusters ({ZONE_TARGET} stops/zone target)")

    # Scale coordinates — lat/lon degrees aren't equal distance units
    # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km × cos(lat) ≈ 91km at -33°
    # Scale lon to equalise with lat in distance terms
    coords = stops_df[["lat", "lon"]].copy()
    coords["lon_scaled"] = coords["lon"] * cos(radians(-33.9))  # Sydney latitude

    X = coords[["lat", "lon_scaled"]].values

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=2000,
        n_init=5,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)

    stops_df = stops_df.copy()
    stops_df["zone_id_raw"] = labels
    sizes = stops_df["zone_id_raw"].value_counts()
    print(f"  Initial zone sizes: min={sizes.min()}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}, std={sizes.std():.1f}")
    violations_over  = (sizes > ZONE_MAX).sum()
    violations_under = (sizes < ZONE_MIN).sum()
    print(f"  Violations: {violations_over} oversized (>{ZONE_MAX}), "
          f"{violations_under} undersized (<{ZONE_MIN})")
    return stops_df


# ---------------------------------------------------------------------------
# STEP 2: SPLIT OVERSIZED ZONES
# ---------------------------------------------------------------------------

def split_oversized_zones(stops_df):
    """
    Any zone with more than ZONE_MAX stops is recursively split using
    K-Means(2) until all sub-zones satisfy the constraint.
    """
    stops_df = stops_df.copy()
    next_zone_id = stops_df["zone_id_raw"].max() + 1
    stops_df["zone_id"] = stops_df["zone_id_raw"].copy()

    zones_to_check = list(stops_df["zone_id"].unique())
    split_count = 0

    while zones_to_check:
        zid = zones_to_check.pop()
        mask = stops_df["zone_id"] == zid
        zone_stops = stops_df[mask]
        n = len(zone_stops)

        if n <= ZONE_MAX:
            continue

        # Need to split — how many sub-zones?
        n_sub = int(np.ceil(n / ZONE_TARGET))
        n_sub = max(2, n_sub)

        X = zone_stops[["lat", "lon"]].values
        sub_kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init=3)
        sub_labels = sub_kmeans.fit_predict(X)

        # Assign new zone IDs to the split zones
        new_ids = []
        for sub_label in range(n_sub):
            new_id = next_zone_id
            next_zone_id += 1
            sub_mask = sub_labels == sub_label
            stops_df.loc[zone_stops.index[sub_mask], "zone_id"] = new_id
            new_ids.append(new_id)
            split_count += 1

        # Add new zones to check list (they may still be too large)
        zones_to_check.extend(new_ids)

    print(f"  Split {split_count} oversized zones")
    sizes = stops_df["zone_id"].value_counts()
    print(f"  Post-split sizes: min={sizes.min()}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}")
    return stops_df


# ---------------------------------------------------------------------------
# STEP 3: MERGE UNDERSIZED ZONES
# ---------------------------------------------------------------------------

def merge_undersized_zones(stops_df):
    """
    Zones with fewer than ZONE_MIN stops are merged into their
    nearest neighbour zone (by centroid distance), provided the
    merged size does not exceed ZONE_MAX.
    Iterates until no more merges are possible.
    """
    stops_df = stops_df.copy()
    merged_count = 0

    for iteration in range(20):  # max iterations to prevent infinite loop
        sizes = stops_df["zone_id"].value_counts()
        undersized = sizes[sizes < ZONE_MIN].index.tolist()

        if not undersized:
            break

        # Compute centroids for all zones
        centroids = stops_df.groupby("zone_id")[["lat", "lon"]].mean()

        made_merge = False
        for zid in undersized:
            if zid not in stops_df["zone_id"].values:
                continue  # already merged in this iteration

            z_size = (stops_df["zone_id"] == zid).sum()
            if z_size >= ZONE_MIN:
                continue

            z_lat, z_lon = centroids.loc[zid, "lat"], centroids.loc[zid, "lon"]

            # Find nearest zone that can absorb these stops
            other_zones = centroids.drop(index=zid)
            if len(other_zones) == 0:
                continue

            dists = other_zones.apply(
                lambda r: haversine_km(z_lat, z_lon, r["lat"], r["lon"]), axis=1
            )
            dists_sorted = dists.sort_values()

            for candidate_id, dist in dists_sorted.items():
                candidate_size = (stops_df["zone_id"] == candidate_id).sum()
                if candidate_size + z_size <= ZONE_MAX:
                    # Merge: reassign all stops from zid → candidate_id
                    stops_df.loc[stops_df["zone_id"] == zid, "zone_id"] = candidate_id
                    merged_count += 1
                    made_merge = True
                    break

        if not made_merge:
            # Can't merge any more without violating constraints — leave as-is
            # These will be flagged in the quality report
            break

    sizes = stops_df["zone_id"].value_counts()
    still_under = (sizes < ZONE_MIN).sum()
    print(f"  Merged {merged_count} undersized zones (iterations: {iteration+1})")
    print(f"  Post-merge sizes: min={sizes.min()}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}")
    if still_under > 0:
        print(f"  WARNING: {still_under} zones still below {ZONE_MIN} stops "
              f"(cannot merge without violating max constraint)")
    return stops_df


# ---------------------------------------------------------------------------
# STEP 4: RENUMBER ZONES AND ASSIGN SUBURB LABEL
# ---------------------------------------------------------------------------

def finalise_zone_ids(stops_df):
    """
    Renumber zones sequentially (Z0001, Z0002, ...) and annotate each
    zone with its dominant suburb.
    """
    stops_df = stops_df.copy()

    # Map old IDs → new sequential IDs
    unique_ids = sorted(stops_df["zone_id"].unique())
    id_map = {old: f"Z{new+1:04d}" for new, old in enumerate(unique_ids)}
    stops_df["zone_id"] = stops_df["zone_id"].map(id_map)

    # Dominant suburb per zone
    dominant_suburb = (
        stops_df.groupby("zone_id")["suburb"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("dominant_suburb")
    )
    stops_df = stops_df.merge(dominant_suburb, on="zone_id", how="left")

    n_zones = stops_df["zone_id"].nunique()
    print(f"  Finalised {n_zones:,} zones (IDs: Z0001 – Z{n_zones:04d})")
    return stops_df


# ---------------------------------------------------------------------------
# STEP 5: BOUNDARY REFINEMENT
# ---------------------------------------------------------------------------

def refine_boundary_stops(stops_df, n_passes=2):
    """
    One or two passes of boundary refinement:
    For each stop, check if reassigning it to a neighbouring zone
    (whose centroid is closer) would improve compactness without
    violating size constraints.

    This tightens zones that have straggler stops on their edges.
    """
    stops_df = stops_df.copy()
    reassigned_total = 0

    for pass_num in range(n_passes):
        # Compute centroids
        centroids = stops_df.groupby("zone_id")[["lat", "lon"]].mean()
        sizes = stops_df["zone_id"].value_counts()

        reassigned = 0
        # Sample a random order to avoid directional bias
        indices = stops_df.index.tolist()
        np.random.shuffle(indices)

        for idx in indices:
            row = stops_df.loc[idx]
            current_zone = row["zone_id"]
            current_centroid = centroids.loc[current_zone]
            dist_current = haversine_km(
                row["lat"], row["lon"],
                current_centroid["lat"], current_centroid["lon"]
            )

            # Only consider reassigning if stop is in the furthest 30% of its zone
            zone_stops = stops_df[stops_df["zone_id"] == current_zone]
            zone_centroid_lat, zone_centroid_lon = (
                current_centroid["lat"], current_centroid["lon"]
            )
            zone_dists = zone_stops.apply(
                lambda r: haversine_km(
                    r["lat"], r["lon"], zone_centroid_lat, zone_centroid_lon
                ), axis=1
            )
            threshold = zone_dists.quantile(0.70)
            if dist_current < threshold:
                continue

            # Find closest other zone centroid
            other_centroids = centroids.drop(index=current_zone)
            dists_to_others = other_centroids.apply(
                lambda r: haversine_km(row["lat"], row["lon"], r["lat"], r["lon"]),
                axis=1
            )
            nearest_zone_id = dists_to_others.idxmin()
            dist_nearest = dists_to_others.min()

            if dist_nearest >= dist_current:
                continue  # current zone is already closest

            # Check size constraints
            current_size = sizes.get(current_zone, 0)
            nearest_size = sizes.get(nearest_zone_id, 0)

            if current_size - 1 < ZONE_MIN:
                continue  # would make current zone too small
            if nearest_size + 1 > ZONE_MAX:
                continue  # would make target zone too large

            # Reassign
            stops_df.at[idx, "zone_id"] = nearest_zone_id
            sizes[current_zone] -= 1
            sizes[nearest_zone_id] = sizes.get(nearest_zone_id, 0) + 1
            # Update centroid (approximate — recalculated each pass)
            reassigned += 1

        reassigned_total += reassigned
        print(f"  Refinement pass {pass_num+1}: {reassigned} stops reassigned")

    return stops_df


# ---------------------------------------------------------------------------
# STEP 6: BUILD ZONE SUMMARY
# ---------------------------------------------------------------------------

def build_zone_summary(stops_df):
    """
    Compute per-zone statistics:
    - Stop count
    - Centroid lat/lon
    - Intra-zone travel stats (sampled pairwise + TSP estimate)
    - Suburb composition
    - Dominant suburb
    """
    print("  Computing intra-zone travel stats (this takes ~30s)...")
    records = []

    zone_ids = stops_df["zone_id"].unique()
    for i, zid in enumerate(sorted(zone_ids)):
        zone_stops = stops_df[stops_df["zone_id"] == zid]
        clat, clon = zone_centroid(zone_stops)
        tt_stats = intra_zone_travel_stats(zone_stops)
        suburb_counts = zone_stops["suburb"].value_counts()
        dominant_suburb = suburb_counts.index[0]
        suburb_mix = ", ".join(
            f"{sub}({cnt})" for sub, cnt in suburb_counts.items()
        )

        # Zone diameter: max distance between any two stops in zone
        lats = zone_stops["lat"].values
        lons = zone_stops["lon"].values
        if len(lats) >= 2:
            # Sample max distance from a few random pairs
            sample_dists = [
                haversine_km(lats[a], lons[a], lats[b], lons[b])
                for a, b in [
                    np.random.choice(len(lats), 2, replace=False)
                    for _ in range(min(20, len(lats)*(len(lats)-1)//2))
                ]
            ]
            zone_diameter_km = round(max(sample_dists), 3)
        else:
            zone_diameter_km = 0.0

        records.append({
            "zone_id":           zid,
            "n_stops":           len(zone_stops),
            "centroid_lat":      round(clat, 6),
            "centroid_lon":      round(clon, 6),
            "dominant_suburb":   dominant_suburb,
            "suburb_mix":        suburb_mix,
            "zone_diameter_km":  zone_diameter_km,
            "mean_intra_tt_s":   tt_stats["mean_tt_s"],
            "median_intra_tt_s": tt_stats["median_tt_s"],
            "max_intra_tt_s":    tt_stats["max_tt_s"],
            "tsp_estimate_s":    tt_stats["tsp_estimate_s"],
            "tsp_estimate_min":  round(tt_stats["tsp_estimate_s"] / 60, 1),
        })

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(zone_ids)} zones processed...")

    df = pd.DataFrame(records)
    print(f"  Zone summary: {len(df)} zones")
    return df


# ---------------------------------------------------------------------------
# STEP 7: VISUALISATION — FOLIUM MAP
# ---------------------------------------------------------------------------

def build_zone_map(stops_df, zone_summary_df, depot):
    """
    Interactive folium map showing:
    - Each stop coloured by zone
    - Zone centroids with summary popup
    - Depot marker
    """
    # Centre map on Sydney
    m = folium.Map(location=[-33.90, 151.22], zoom_start=13, tiles="CartoDB positron")

    # Colour palette — cycle through distinct colours for adjacent zones
    n_zones = stops_df["zone_id"].nunique()
    # Use a set of visually distinct colours
    COLOURS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
    ]
    unique_zones = sorted(stops_df["zone_id"].unique())
    zone_colour = {
        zid: COLOURS[i % len(COLOURS)] for i, zid in enumerate(unique_zones)
    }

    # Plot stops (small circles)
    for _, row in stops_df.iterrows():
        colour = zone_colour[row["zone_id"]]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            weight=0,
            popup=folium.Popup(
                f"<b>{row['stop_id']}</b><br>"
                f"Zone: {row['zone_id']}<br>"
                f"Suburb: {row['suburb']}<br>"
                f"Type: {row['stop_type']}<br>"
                f"{row['address']}",
                max_width=220,
            ),
            tooltip=row["zone_id"],
        ).add_to(m)

    # Plot zone centroids (larger markers with summary)
    for _, z in zone_summary_df.iterrows():
        colour = zone_colour.get(z["zone_id"], "#666666")
        folium.CircleMarker(
            location=[z["centroid_lat"], z["centroid_lon"]],
            radius=8,
            color="#222222",
            fill=True,
            fill_color=colour,
            fill_opacity=1.0,
            weight=1.5,
            popup=folium.Popup(
                f"<b>Zone {z['zone_id']}</b><br>"
                f"Stops: {z['n_stops']}<br>"
                f"Suburb: {z['dominant_suburb']}<br>"
                f"Diameter: {z['zone_diameter_km']} km<br>"
                f"TSP est: {z['tsp_estimate_min']} min<br>"
                f"Avg intra-travel: {z['mean_intra_tt_s']:.0f}s",
                max_width=220,
            ),
            tooltip=f"{z['zone_id']} ({z['n_stops']} stops)",
        ).add_to(m)

    # Depot marker
    folium.Marker(
        location=[depot["lat"], depot["lon"]],
        popup=folium.Popup(
            f"<b>{depot['name']}</b><br>{depot['address']}", max_width=200
        ),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        tooltip="Botany DC",
    ).add_to(m)

    # Add layer control and title
    title_html = """
    <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                background: white; padding: 10px 15px; border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <b>Botany DC — Zone Sort Map</b><br>
        <span style="color:#555; font-size:12px;">
            Each colour = one zone (15–25 stops)<br>
            Large dots = zone centroids (click for stats)<br>
            Small dots = individual stops
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# ---------------------------------------------------------------------------
# STEP 8: MATPLOTLIB CHARTS
# ---------------------------------------------------------------------------

def build_charts(stops_df, zone_summary_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Botany DC — Zone Construction Quality Report", fontsize=14, fontweight="bold")

    sizes = zone_summary_df["n_stops"]

    # 1. Zone size distribution
    ax = axes[0, 0]
    ax.hist(sizes, bins=range(ZONE_MIN - 2, ZONE_MAX + 4), color="#3b82f6",
            edgecolor="white", linewidth=0.5)
    ax.axvline(ZONE_MIN, color="red",   linestyle="--", linewidth=1.5, label=f"Min ({ZONE_MIN})")
    ax.axvline(ZONE_MAX, color="red",   linestyle="--", linewidth=1.5, label=f"Max ({ZONE_MAX})")
    ax.axvline(sizes.mean(), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean ({sizes.mean():.1f})")
    ax.set_xlabel("Stops per zone")
    ax.set_ylabel("Number of zones")
    ax.set_title("Zone Size Distribution")
    ax.legend(fontsize=8)

    # 2. Zones per suburb
    ax = axes[0, 1]
    suburb_zone_counts = zone_summary_df["dominant_suburb"].value_counts()
    ax.barh(suburb_zone_counts.index, suburb_zone_counts.values, color="#10b981")
    ax.set_xlabel("Number of zones")
    ax.set_title("Zones per Suburb\n(dominant suburb)")
    ax.tick_params(axis="y", labelsize=8)

    # 3. Intra-zone TSP travel time distribution
    ax = axes[0, 2]
    tsp_min = zone_summary_df["tsp_estimate_min"]
    ax.hist(tsp_min, bins=25, color="#f59e0b", edgecolor="white", linewidth=0.5)
    ax.axvline(tsp_min.median(), color="red", linestyle="--", linewidth=1.5,
               label=f"Median ({tsp_min.median():.0f} min)")
    ax.set_xlabel("TSP travel time (minutes)")
    ax.set_ylabel("Number of zones")
    ax.set_title("Intra-Zone Route Time\n(TSP estimate)")
    ax.legend(fontsize=8)

    # 4. Zone diameter distribution
    ax = axes[1, 0]
    ax.hist(zone_summary_df["zone_diameter_km"], bins=25,
            color="#8b5cf6", edgecolor="white", linewidth=0.5)
    ax.axvline(zone_summary_df["zone_diameter_km"].median(), color="red",
               linestyle="--", linewidth=1.5,
               label=f"Median ({zone_summary_df['zone_diameter_km'].median():.2f} km)")
    ax.set_xlabel("Zone diameter (km)")
    ax.set_ylabel("Number of zones")
    ax.set_title("Zone Diameter Distribution\n(max sampled stop distance)")
    ax.legend(fontsize=8)

    # 5. Stops per zone by suburb (box plot)
    ax = axes[1, 1]
    suburbs_in_data = sorted(zone_summary_df["dominant_suburb"].unique())
    box_data = [
        zone_summary_df[zone_summary_df["dominant_suburb"] == s]["n_stops"].values
        for s in suburbs_in_data
    ]
    bp = ax.boxplot(box_data, patch_artist=True, vert=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#60a5fa")
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(suburbs_in_data) + 1))
    ax.set_xticklabels(suburbs_in_data, rotation=45, ha="right", fontsize=7)
    ax.axhline(ZONE_MIN, color="red",   linestyle="--", linewidth=1, label=f"Min ({ZONE_MIN})")
    ax.axhline(ZONE_MAX, color="red",   linestyle="--", linewidth=1, label=f"Max ({ZONE_MAX})")
    ax.set_ylabel("Stops per zone")
    ax.set_title("Zone Size by Suburb")
    ax.legend(fontsize=7)

    # 6. Constraint compliance summary
    ax = axes[1, 2]
    compliant     = ((sizes >= ZONE_MIN) & (sizes <= ZONE_MAX)).sum()
    over_sized    = (sizes > ZONE_MAX).sum()
    under_sized   = (sizes < ZONE_MIN).sum()
    total         = len(sizes)
    compliance_pct = compliant / total * 100

    wedges, texts, autotexts = ax.pie(
        [compliant, over_sized, under_sized],
        labels=[
            f"Compliant\n({compliant}, {compliance_pct:.1f}%)",
            f"Oversized\n({over_sized})",
            f"Undersized\n({under_sized})",
        ],
        colors=["#10b981", "#ef4444", "#f59e0b"],
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax.set_title(f"Constraint Compliance\n({total} total zones)")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Zone Construction — Botany DC")
    print(f"Target zone size: {ZONE_MIN}–{ZONE_MAX} stops")
    print("=" * 65)

    # Load data
    stops_df = pd.read_csv(os.path.join(DATA_DIR, "stop_master.csv"))
    depot_df = pd.read_csv(os.path.join(DATA_DIR, "depot.csv"))
    depot = depot_df.iloc[0].to_dict()
    print(f"\nLoaded {len(stops_df):,} stops across "
          f"{stops_df['suburb'].nunique()} suburbs\n")

    # --- Build zones ---
    t0 = time.time()

    print("[1/4] Initial K-Means partition...")
    stops_df = initial_kmeans(stops_df)

    print("\n[2/4] Splitting oversized zones...")
    stops_df = split_oversized_zones(stops_df)

    print("\n[3/4] Merging undersized zones...")
    stops_df = merge_undersized_zones(stops_df)

    print("\n[4/4] Boundary refinement (2 passes)...")
    stops_df = refine_boundary_stops(stops_df, n_passes=2)

    # Finalise IDs
    stops_df = finalise_zone_ids(stops_df)

    t_cluster = time.time() - t0
    print(f"\nClustering complete in {t_cluster:.1f}s")

    # --- Zone summary ---
    print("\nBuilding zone summary statistics...")
    zone_summary_df = build_zone_summary(stops_df)

    # --- Constraint check ---
    sizes = zone_summary_df["n_stops"]
    compliant = ((sizes >= ZONE_MIN) & (sizes <= ZONE_MAX)).sum()
    print(f"\nConstraint compliance: {compliant}/{len(sizes)} zones "
          f"({compliant/len(sizes)*100:.1f}%) within [{ZONE_MIN}, {ZONE_MAX}]")

    # --- Write outputs ---
    assignments_path = os.path.join(OUTPUT_DIR, "zone_assignments.csv")
    summary_path     = os.path.join(OUTPUT_DIR, "zone_summary.csv")
    map_path         = os.path.join(OUTPUT_DIR, "zone_map.html")
    chart_path       = os.path.join(OUTPUT_DIR, "zone_charts.png")

    stops_df[["stop_id", "suburb", "lat", "lon", "stop_type",
              "address", "zone_id", "dominant_suburb"]].to_csv(
        assignments_path, index=False
    )
    zone_summary_df.to_csv(summary_path, index=False)
    print(f"\nzone_assignments.csv written ({len(stops_df):,} rows)")
    print(f"zone_summary.csv     written ({len(zone_summary_df):,} rows)")

    print("\nBuilding zone map...")
    m = build_zone_map(stops_df, zone_summary_df, depot)
    m.save(map_path)
    print(f"zone_map.html        written — open in browser to explore")

    print("\nBuilding charts...")
    fig = build_charts(stops_df, zone_summary_df)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"zone_charts.png      written")

    # --- Final summary ---
    print("\n" + "=" * 65)
    print("ZONE CONSTRUCTION RESULTS")
    print("=" * 65)
    print(f"Total zones created:        {len(zone_summary_df):>6,}")
    print(f"Total stops assigned:       {len(stops_df):>6,}")
    print(f"Avg stops/zone:             {sizes.mean():>6.1f}")
    print(f"Std stops/zone:             {sizes.std():>6.1f}")
    print(f"Min stops/zone:             {sizes.min():>6}")
    print(f"Max stops/zone:             {sizes.max():>6}")
    print(f"Compliant zones:            {compliant:>6,} ({compliant/len(sizes)*100:.1f}%)")
    print(f"Avg zone diameter:          {zone_summary_df['zone_diameter_km'].mean():>6.2f} km")
    print(f"Median TSP route time:      {zone_summary_df['tsp_estimate_min'].median():>6.1f} min")
    print(f"Avg intra-zone travel:      {zone_summary_df['mean_intra_tt_s'].mean():>6.0f} s")
    print()
    print("Zones by suburb (dominant):")
    for suburb, count in zone_summary_df["dominant_suburb"].value_counts().items():
        pct = count / len(zone_summary_df) * 100
        bar = "|" * int(pct / 2)
        print(f"  {suburb:<20} {count:>4} zones  {bar}")
    print()
    print(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
