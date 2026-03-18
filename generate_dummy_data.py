"""
generate_dummy_data.py
======================
Generates realistic dummy delivery data for Botany DC zone-sort modelling.

Botany DC: 25,000 packages/day
Delivery areas: Bondi, Rose Bay, Coogee, Kensington, Maroubra, Waterloo,
                Alexandria, Surry Hills, Darlinghurst, Paddington, Double Bay,
                Glebe, Sydney CBD

Outputs (written to ./data/):
  depot.csv               - Depot location
  suburbs.csv             - Suburb reference data
  stop_master.csv         - All unique delivery stops (~9,500 stops)
  package_manifest.csv    - One day's 25,000 packages with cube + service class
  service_time_log.csv    - 20 days of historical service time observations
  route_history.csv       - 20 days of historical route sequences (current unsorted state)
  travel_time_sample.csv  - Travel times between stop pairs (sampled, not full matrix)
"""

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from math import radians, cos, sin, asin, sqrt
import os
from datetime import date, timedelta
import itertools

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DEPOT
# ---------------------------------------------------------------------------

DEPOT = {
    "depot_id": "BOT001",
    "name": "Botany DC",
    "lat": -33.9457,
    "lon": 151.1954,
    "address": "1 Botany Rd, Botany NSW 2019",
}

# ---------------------------------------------------------------------------
# 2. SUBURB DEFINITIONS
#    lat/lon = approximate suburb centroid
#    lat_std/lon_std = spread for stop generation (~500m-1km radius)
#    packages = daily package volume
#    avg_pkgs_per_stop = controls how many stops are generated
#    stop_type_mix = proportion of business / apartment / residential stops
# ---------------------------------------------------------------------------

SUBURBS = {
    "Sydney CBD": {
        "lat": -33.8688, "lon": 151.2093,
        "lat_std": 0.012, "lon_std": 0.015,
        "packages": 4200,
        "avg_pkgs_per_stop": 2.8,
        "stop_type_mix": {"business": 0.55, "apartment": 0.35, "residential": 0.10},
        "streets": [
            "George St", "Pitt St", "Kent St", "York St", "Clarence St",
            "Sussex St", "Market St", "King St", "Hunter St", "Bond St",
            "Castlereagh St", "Elizabeth St", "Phillip St", "Macquarie St",
        ],
    },
    "Surry Hills": {
        "lat": -33.8885, "lon": 151.2100,
        "lat_std": 0.008, "lon_std": 0.009,
        "packages": 2400,
        "avg_pkgs_per_stop": 1.7,
        "stop_type_mix": {"business": 0.20, "apartment": 0.40, "residential": 0.40},
        "streets": [
            "Crown St", "Cleveland St", "Bourke St", "Foveaux St",
            "Albion St", "Devonshire St", "Riley St", "Fitzroy St",
            "Campbell St", "Oxford St",
        ],
    },
    "Darlinghurst": {
        "lat": -33.8760, "lon": 151.2210,
        "lat_std": 0.007, "lon_std": 0.008,
        "packages": 1800,
        "avg_pkgs_per_stop": 1.6,
        "stop_type_mix": {"business": 0.15, "apartment": 0.55, "residential": 0.30},
        "streets": [
            "Oxford St", "Victoria St", "Burton St", "Liverpool St",
            "William St", "Darlinghurst Rd", "Forbes St", "Palmer St",
        ],
    },
    "Paddington": {
        "lat": -33.8848, "lon": 151.2290,
        "lat_std": 0.008, "lon_std": 0.009,
        "packages": 1700,
        "avg_pkgs_per_stop": 1.5,
        "stop_type_mix": {"business": 0.15, "apartment": 0.25, "residential": 0.60},
        "streets": [
            "Oxford St", "Glenmore Rd", "Paddington St", "Jersey Rd",
            "Cascade St", "William St", "Heeley St", "Goodhope St",
        ],
    },
    "Double Bay": {
        "lat": -33.8762, "lon": 151.2455,
        "lat_std": 0.006, "lon_std": 0.007,
        "packages": 700,
        "avg_pkgs_per_stop": 1.4,
        "stop_type_mix": {"business": 0.25, "apartment": 0.35, "residential": 0.40},
        "streets": [
            "New South Head Rd", "Bay St", "Cross St", "Knox St",
            "Manning Rd", "Ocean Ave", "William St",
        ],
    },
    "Rose Bay": {
        "lat": -33.8721, "lon": 151.2681,
        "lat_std": 0.007, "lon_std": 0.008,
        "packages": 900,
        "avg_pkgs_per_stop": 1.4,
        "stop_type_mix": {"business": 0.10, "apartment": 0.30, "residential": 0.60},
        "streets": [
            "New South Head Rd", "Dover Rd", "Vickery Ave", "Plumer Rd",
            "Beaumont St", "O'Sullivan Rd", "Woollahra Ave",
        ],
    },
    "Bondi": {
        "lat": -33.8914, "lon": 151.2743,
        "lat_std": 0.009, "lon_std": 0.009,
        "packages": 2000,
        "avg_pkgs_per_stop": 1.6,
        "stop_type_mix": {"business": 0.15, "apartment": 0.45, "residential": 0.40},
        "streets": [
            "Campbell Pde", "Hall St", "Curlewis St", "Gould St",
            "O'Brien St", "Blair St", "Notts Ave", "Pacific Ave",
            "Beach Rd", "Lamrock Ave",
        ],
    },
    "Coogee": {
        "lat": -33.9200, "lon": 151.2567,
        "lat_std": 0.008, "lon_std": 0.008,
        "packages": 1500,
        "avg_pkgs_per_stop": 1.5,
        "stop_type_mix": {"business": 0.10, "apartment": 0.40, "residential": 0.50},
        "streets": [
            "Coogee Bay Rd", "Beach St", "Arden St", "Carr St",
            "Brook St", "Mount St", "Dolphin St", "Dudley St",
        ],
    },
    "Maroubra": {
        "lat": -33.9500, "lon": 151.2500,
        "lat_std": 0.010, "lon_std": 0.010,
        "packages": 2100,
        "avg_pkgs_per_stop": 1.5,
        "stop_type_mix": {"business": 0.10, "apartment": 0.30, "residential": 0.60},
        "streets": [
            "Maroubra Rd", "Marine Pde", "Moverly Rd", "Fitzgerald Ave",
            "Malabar Rd", "Beauchamp Rd", "Storey St", "Eva St",
        ],
    },
    "Kensington": {
        "lat": -33.9160, "lon": 151.2250,
        "lat_std": 0.007, "lon_std": 0.007,
        "packages": 1500,
        "avg_pkgs_per_stop": 1.5,
        "stop_type_mix": {"business": 0.10, "apartment": 0.35, "residential": 0.55},
        "streets": [
            "Anzac Pde", "Abbotford St", "Doncaster Ave", "High St",
            "Day Ave", "Barker St", "Boronia St", "Meeks St",
        ],
    },
    "Waterloo": {
        "lat": -33.9000, "lon": 151.2050,
        "lat_std": 0.007, "lon_std": 0.007,
        "packages": 2000,
        "avg_pkgs_per_stop": 2.0,
        "stop_type_mix": {"business": 0.15, "apartment": 0.60, "residential": 0.25},
        "streets": [
            "Botany Rd", "Bourke St", "Cope St", "Raglan St",
            "Young St", "Wellington St", "Henderson Rd", "McEvoy St",
        ],
    },
    "Alexandria": {
        "lat": -33.9050, "lon": 151.1950,
        "lat_std": 0.008, "lon_std": 0.008,
        "packages": 1900,
        "avg_pkgs_per_stop": 2.0,
        "stop_type_mix": {"business": 0.40, "apartment": 0.40, "residential": 0.20},
        "streets": [
            "Botany Rd", "O'Riordan St", "Innes Rd", "Doody St",
            "Mitchell Rd", "Bourke Rd", "Wyndham St", "Garden St",
        ],
    },
    "Glebe": {
        "lat": -33.8800, "lon": 151.1870,
        "lat_std": 0.008, "lon_std": 0.008,
        "packages": 1300,
        "avg_pkgs_per_stop": 1.5,
        "stop_type_mix": {"business": 0.15, "apartment": 0.30, "residential": 0.55},
        "streets": [
            "Glebe Point Rd", "Broadway", "Bridge Rd", "St Johns Rd",
            "Cowper St", "Wigram Rd", "Forest Lodge Rd", "Arundel St",
        ],
    },
}

TOTAL_PACKAGES = sum(s["packages"] for s in SUBURBS.values())
print(f"Total packages configured: {TOTAL_PACKAGES:,}")

# ---------------------------------------------------------------------------
# 3. PACKAGE TYPE DEFINITIONS
#    Realistic AusPost parcel mix by volume segment
# ---------------------------------------------------------------------------

PACKAGE_TYPES = [
    # (type_name, length_cm, width_cm, height_cm, weight_kg, proportion, service_class_mix)
    # service_class: ATL=authority to leave, SIG=signature, CARD=card left
    ("small_satchel",   25, 18,  3,  0.4, 0.30, {"ATL": 0.70, "SIG": 0.15, "CARD": 0.15}),
    ("medium_satchel",  32, 22,  6,  0.9, 0.22, {"ATL": 0.65, "SIG": 0.20, "CARD": 0.15}),
    ("small_parcel",    32, 27, 18,  2.0, 0.20, {"ATL": 0.55, "SIG": 0.25, "CARD": 0.20}),
    ("medium_parcel",   45, 35, 25,  5.0, 0.14, {"ATL": 0.50, "SIG": 0.30, "CARD": 0.20}),
    ("large_parcel",    60, 45, 35, 10.0, 0.09, {"ATL": 0.40, "SIG": 0.35, "CARD": 0.25}),
    ("extra_large",     80, 60, 40, 18.0, 0.05, {"ATL": 0.30, "SIG": 0.40, "CARD": 0.30}),
]

# Service time parameters by stop type and service class (seconds)
# Base service time: walk to door + scan
# These are used to parameterise the log-normal distribution
SERVICE_TIME_PARAMS = {
    # (stop_type, service_class): (mu_lognormal, sigma_lognormal)
    # Derived from: base=35s residential, +20s apt (intercom/lift), +15s business (reception)
    # +20s signature, +30s card-left (no-answer), multiplied by pkg count at stop
    ("residential", "ATL"):  (3.80, 0.30),   # ~45s median
    ("residential", "SIG"):  (4.05, 0.28),   # ~57s median
    ("residential", "CARD"): (4.20, 0.32),   # ~67s median
    ("apartment",   "ATL"):  (4.10, 0.35),   # ~60s median (intercom + lift)
    ("apartment",   "SIG"):  (4.30, 0.32),   # ~74s median
    ("apartment",   "CARD"): (4.45, 0.35),   # ~86s median
    ("business",    "ATL"):  (3.90, 0.28),   # ~49s median (reception)
    ("business",    "SIG"):  (4.15, 0.25),   # ~63s median
    ("business",    "CARD"): (3.75, 0.30),   # ~42s (left at reception)
}

# Unsorted van penalty: multiplier on service time (driver rummages for parcel)
# Current state: driver loses time finding parcel in unsorted van
UNSORTED_PENALTY_MU = 0.12   # adds ~12% to log-mean (≈13% uplift on median time)
UNSORTED_PENALTY_SIGMA = 0.05 # adds variance from unpredictable search times

# Average road speed for travel time estimation (km/h by area type)
ROAD_SPEED_URBAN = 28    # inner city / mixed
ROAD_SPEED_SUBURBAN = 38  # outer suburbs

# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def travel_time_seconds(lat1, lon1, lat2, lon2, suburb1=None, suburb2=None):
    """
    Estimate road travel time in seconds between two stops.
    Uses Haversine distance × road factor (1.35 for urban routing inefficiency)
    × speed based on suburb type. Adds realistic random noise.
    """
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    road_dist_km = dist_km * 1.35  # road network vs straight-line
    speed = ROAD_SPEED_URBAN
    base_time = (road_dist_km / speed) * 3600  # seconds
    # Add 15% random noise to simulate congestion/traffic variation
    noise = np.random.normal(1.0, 0.12)
    return max(20, base_time * noise)  # minimum 20s (adjacent stops)


def sample_service_class(stop_type, pkg_type_name):
    """Sample a service class for a package given stop type and package type."""
    pkg = next(p for p in PACKAGE_TYPES if p[0] == pkg_type_name)
    mix = pkg[6]
    return np.random.choice(list(mix.keys()), p=list(mix.values()))


def generate_stop_service_time(stop_type, service_class, n_packages, sorted_state=False):
    """
    Generate a realistic service time for a stop (seconds).
    Accounts for: stop type, service class, number of packages, sort state.
    """
    key = (stop_type, service_class)
    mu, sigma = SERVICE_TIME_PARAMS.get(key, (3.90, 0.30))

    # Multi-parcel stops: each additional parcel adds ~60% of base time
    # (driver already at door, just additional scan+carry)
    pkg_factor = 1.0 + (n_packages - 1) * 0.60
    effective_mu = mu + np.log(pkg_factor)

    # Unsorted van penalty: driver spends extra time finding parcels
    if not sorted_state:
        effective_mu += UNSORTED_PENALTY_MU
        sigma = min(sigma + UNSORTED_PENALTY_SIGMA, 0.60)

    return np.random.lognormal(effective_mu, sigma)


# ---------------------------------------------------------------------------
# 5. GENERATE STOPS
# ---------------------------------------------------------------------------

def generate_stops():
    """
    Generate unique delivery stops for each suburb.
    Number of stops = packages / avg_pkgs_per_stop
    """
    stops = []
    stop_id = 1

    for suburb_name, s in SUBURBS.items():
        n_stops = int(round(s["packages"] / s["avg_pkgs_per_stop"]))
        type_names = list(s["stop_type_mix"].keys())
        type_probs = list(s["stop_type_mix"].values())

        lats = np.random.normal(s["lat"], s["lat_std"], n_stops)
        lons = np.random.normal(s["lon"], s["lon_std"], n_stops)
        stop_types = np.random.choice(type_names, size=n_stops, p=type_probs)
        street_names = np.random.choice(s["streets"], size=n_stops)
        house_numbers = np.random.randint(1, 350, size=n_stops)
        # Even numbers on one side of street
        house_numbers = house_numbers * 2

        for i in range(n_stops):
            stops.append({
                "stop_id": f"S{stop_id:05d}",
                "suburb": suburb_name,
                "lat": round(lats[i], 6),
                "lon": round(lons[i], 6),
                "stop_type": stop_types[i],
                "address": f"{house_numbers[i]} {street_names[i]}, {suburb_name} NSW",
            })
            stop_id += 1

    df = pd.DataFrame(stops)
    print(f"Generated {len(df):,} unique stops across {df['suburb'].nunique()} suburbs")
    return df


# ---------------------------------------------------------------------------
# 6. GENERATE PACKAGE MANIFEST (one day)
# ---------------------------------------------------------------------------

def generate_package_manifest(stop_master_df, day_date=None):
    """
    Generate 25,000 packages assigned to stops, with cube dimensions,
    weight, package type, and service class.
    Packages per stop follows a negative binomial distribution (realistic
    household delivery — most stops get 1-2, some get 5+).
    """
    if day_date is None:
        day_date = date(2025, 6, 2)

    # Package type proportions
    type_names = [p[0] for p in PACKAGE_TYPES]
    type_props = [p[5] for p in PACKAGE_TYPES]

    packages = []
    pkg_id = 1

    # Assign packages to stops by suburb, respecting daily volume targets
    for suburb_name, s in SUBURBS.items():
        suburb_stops = stop_master_df[stop_master_df["suburb"] == suburb_name].copy()
        n_pkgs_suburb = s["packages"]

        # Distribute packages across stops using negative binomial
        # This gives realistic skew: most stops 1-2 pkgs, some stops 5-10+
        raw_counts = np.random.negative_binomial(1.5, 0.45, size=len(suburb_stops))
        raw_counts = np.clip(raw_counts, 1, 12)  # min 1, max 12 per stop
        # Scale to hit target volume
        scale = n_pkgs_suburb / raw_counts.sum()
        counts = np.round(raw_counts * scale).astype(int)
        counts = np.clip(counts, 1, 12)
        # Adjust last stop to hit exact total
        diff = n_pkgs_suburb - counts.sum()
        counts[-1] = max(1, counts[-1] + diff)

        for idx, (_, stop_row) in enumerate(suburb_stops.iterrows()):
            n = counts[idx]
            for _ in range(n):
                pkg_type = np.random.choice(type_names, p=type_props)
                ptype = next(p for p in PACKAGE_TYPES if p[0] == pkg_type)
                svc_class = sample_service_class(stop_row["stop_type"], pkg_type)

                # Add dimensional variation ±10%
                l = round(ptype[1] * np.random.uniform(0.92, 1.10), 1)
                w = round(ptype[2] * np.random.uniform(0.92, 1.10), 1)
                h = round(ptype[3] * np.random.uniform(0.90, 1.12), 1)
                wt = round(ptype[4] * np.random.uniform(0.85, 1.15), 2)
                cube_litres = round(l * w * h / 1000, 2)

                packages.append({
                    "parcel_id": f"AP{day_date.strftime('%Y%m%d')}{pkg_id:06d}",
                    "stop_id": stop_row["stop_id"],
                    "suburb": suburb_name,
                    "package_type": pkg_type,
                    "service_class": svc_class,
                    "length_cm": l,
                    "width_cm": w,
                    "height_cm": h,
                    "weight_kg": wt,
                    "cube_litres": cube_litres,
                    "manifest_date": day_date.isoformat(),
                })
                pkg_id += 1

    df = pd.DataFrame(packages)
    print(f"Generated {len(df):,} packages for {day_date}")
    print(f"  Package type breakdown:\n{df['package_type'].value_counts().to_string()}")
    print(f"  Service class breakdown:\n{df['service_class'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 7. GENERATE DRIVERS
# ---------------------------------------------------------------------------

def generate_drivers(n_drivers=147):
    """Generate driver master list."""
    drivers = []
    for i in range(1, n_drivers + 1):
        drivers.append({
            "driver_id": f"D{i:03d}",
            "name": f"Driver {i:03d}",
            "vehicle_type": np.random.choice(
                ["LCV_medium", "LCV_large", "van_small"],
                p=[0.45, 0.35, 0.20]
            ),
            "van_capacity_litres": np.random.choice([800, 1100, 600], p=[0.45, 0.35, 0.20]),
        })
    return pd.DataFrame(drivers)


# ---------------------------------------------------------------------------
# 8. GENERATE ROUTE HISTORY (current unsorted state, 20 days)
# ---------------------------------------------------------------------------

def generate_route_history(stop_master_df, package_manifest_df, n_days=20):
    """
    Simulate 20 days of historical routes in the CURRENT (unsorted) state.

    Each driver is assigned a set of stops. The delivery sequence is NOT
    optimal — it reflects the current practice of rough sort by suburb only,
    with no zone-level sequencing.

    Returns:
        route_history_df   - one row per stop visit with timing data
        service_time_df    - extracted service time observations
    """
    start_date = date(2025, 5, 5)
    n_drivers = 147
    route_records = []
    service_records = []

    # Suburb-to-driver mapping (each driver owns a set of suburbs/areas)
    # In current state: drivers cover broad areas, no zone pre-sort
    suburb_list = list(SUBURBS.keys())
    # Assign suburbs to drivers in blocks
    stops_per_driver = len(stop_master_df) // n_drivers

    for day_num in range(n_days):
        current_date = start_date + timedelta(days=day_num)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Shuffle stops and assign to drivers
        stops_shuffled = stop_master_df.sample(frac=1).reset_index(drop=True)

        for driver_num in range(n_drivers):
            driver_id = f"D{driver_num + 1:03d}"
            start_idx = driver_num * stops_per_driver
            end_idx = min(start_idx + stops_per_driver, len(stops_shuffled))
            driver_stops = stops_shuffled.iloc[start_idx:end_idx].copy()

            if len(driver_stops) == 0:
                continue

            # Current state: sequence is roughly suburb-ordered but NOT zone-optimised
            # Simulate by sorting on suburb then adding positional noise
            driver_stops["sort_key"] = driver_stops["suburb"].astype("category").cat.codes
            driver_stops["sort_noise"] = np.random.uniform(0, 3, len(driver_stops))
            driver_stops = driver_stops.sort_values(
                ["sort_key", "sort_noise"]
            ).reset_index(drop=True)

            # Simulate packages at each stop on this day
            # Sample from packages for this suburb (rough approximation)
            current_time = 8 * 3600  # 08:00 start
            prev_lat = DEPOT["lat"]
            prev_lon = DEPOT["lon"]

            for seq_pos, (_, stop_row) in enumerate(driver_stops.iterrows()):
                # Number of packages at this stop today (1-4, occasionally more)
                n_pkgs_today = np.random.choice(
                    [1, 1, 1, 2, 2, 3, 4],
                    p=[0.35, 0.20, 0.15, 0.15, 0.08, 0.05, 0.02]
                )

                # Dominant service class at this stop
                stop_type = stop_row["stop_type"]
                svc_class = np.random.choice(
                    ["ATL", "SIG", "CARD"], p=[0.60, 0.25, 0.15]
                )

                # Travel time from previous stop (CURRENT state — sub-optimal routing)
                t_travel = travel_time_seconds(
                    prev_lat, prev_lon,
                    stop_row["lat"], stop_row["lon"]
                )
                # Route inefficiency penalty (current state: ~8-15% extra travel)
                route_inefficiency = np.random.lognormal(np.log(1.10), 0.12)
                t_travel_actual = t_travel * route_inefficiency

                # Service time at stop (UNSORTED = True → includes search penalty)
                t_service = generate_stop_service_time(
                    stop_type, svc_class, n_pkgs_today, sorted_state=False
                )

                arrive_time = current_time + t_travel_actual
                depart_time = arrive_time + t_service
                current_time = depart_time

                route_records.append({
                    "route_date": current_date.isoformat(),
                    "driver_id": driver_id,
                    "sequence_position": seq_pos + 1,
                    "stop_id": stop_row["stop_id"],
                    "suburb": stop_row["suburb"],
                    "stop_type": stop_type,
                    "n_packages": n_pkgs_today,
                    "service_class": svc_class,
                    "travel_time_seconds": round(t_travel_actual, 1),
                    "service_time_seconds": round(t_service, 1),
                    "arrive_time_hhmm": f"{int(arrive_time//3600):02d}:{int((arrive_time%3600)//60):02d}",
                    "depart_time_hhmm": f"{int(depart_time//3600):02d}:{int((depart_time%3600)//60):02d}",
                    "sort_state": "unsorted",
                    "lat": stop_row["lat"],
                    "lon": stop_row["lon"],
                })

                service_records.append({
                    "route_date": current_date.isoformat(),
                    "stop_id": stop_row["stop_id"],
                    "suburb": stop_row["suburb"],
                    "stop_type": stop_type,
                    "service_class": svc_class,
                    "n_packages": n_pkgs_today,
                    "sequence_position": seq_pos + 1,
                    "service_time_seconds": round(t_service, 1),
                    "sort_state": "unsorted",
                })

                prev_lat = stop_row["lat"]
                prev_lon = stop_row["lon"]

        print(f"  Day {day_num+1}/{n_days}: {current_date} — route history generated")

    route_df = pd.DataFrame(route_records)
    service_df = pd.DataFrame(service_records)

    print(f"\nRoute history: {len(route_df):,} stop visits across {n_days} days")
    print(f"Service time log: {len(service_df):,} observations")
    print(f"Avg service time (unsorted): {service_df['service_time_seconds'].mean():.1f}s")

    return route_df, service_df


# ---------------------------------------------------------------------------
# 9. GENERATE TRAVEL TIME SAMPLE
#    Full OD matrix for ~9,500 stops is 90M pairs — not practical.
#    Instead generate a representative sample: within-suburb pairs and
#    cross-suburb pairs. This is enough to fit distributions.
# ---------------------------------------------------------------------------

def generate_travel_time_sample(stop_master_df, n_within=5000, n_cross=3000):
    """
    Sample travel times between stop pairs.
    within-suburb: pairs of stops in the same suburb (for zone-level routing)
    cross-suburb: pairs across suburb boundaries (for inter-zone travel)
    """
    records = []

    # Within-suburb pairs
    for suburb_name in SUBURBS.keys():
        suburb_stops = stop_master_df[stop_master_df["suburb"] == suburb_name]
        if len(suburb_stops) < 2:
            continue
        n_sample = min(n_within // len(SUBURBS), len(suburb_stops) * (len(suburb_stops) - 1) // 2)
        idx_pairs = [
            np.random.choice(len(suburb_stops), 2, replace=False)
            for _ in range(n_sample)
        ]
        for a, b in idx_pairs:
            sa = suburb_stops.iloc[a]
            sb = suburb_stops.iloc[b]
            tt = travel_time_seconds(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
            dist = haversine_km(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
            records.append({
                "stop_id_from": sa["stop_id"],
                "stop_id_to": sb["stop_id"],
                "suburb_from": suburb_name,
                "suburb_to": suburb_name,
                "pair_type": "within_suburb",
                "haversine_km": round(dist, 3),
                "travel_time_seconds": round(tt, 1),
            })

    # Cross-suburb pairs
    suburb_names = list(SUBURBS.keys())
    for _ in range(n_cross):
        s1, s2 = np.random.choice(suburb_names, 2, replace=False)
        stops_s1 = stop_master_df[stop_master_df["suburb"] == s1]
        stops_s2 = stop_master_df[stop_master_df["suburb"] == s2]
        if len(stops_s1) == 0 or len(stops_s2) == 0:
            continue
        sa = stops_s1.sample(1).iloc[0]
        sb = stops_s2.sample(1).iloc[0]
        tt = travel_time_seconds(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
        dist = haversine_km(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
        records.append({
            "stop_id_from": sa["stop_id"],
            "stop_id_to": sb["stop_id"],
            "suburb_from": s1,
            "suburb_to": s2,
            "pair_type": "cross_suburb",
            "haversine_km": round(dist, 3),
            "travel_time_seconds": round(tt, 1),
        })

    df = pd.DataFrame(records)
    print(f"\nTravel time sample: {len(df):,} pairs")
    print(f"  Within-suburb median: {df[df['pair_type']=='within_suburb']['travel_time_seconds'].median():.0f}s")
    print(f"  Cross-suburb median:  {df[df['pair_type']=='cross_suburb']['travel_time_seconds'].median():.0f}s")
    return df


# ---------------------------------------------------------------------------
# 10. MAIN — run all generators and write output
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Botany DC — Dummy Data Generator")
    print(f"Target: {TOTAL_PACKAGES:,} packages/day, 13 suburbs")
    print("=" * 60)

    # --- Depot ---
    depot_df = pd.DataFrame([DEPOT])
    depot_df.to_csv(os.path.join(OUTPUT_DIR, "depot.csv"), index=False)
    print("\n[1/6] depot.csv written")

    # --- Suburb reference ---
    suburb_rows = []
    for name, s in SUBURBS.items():
        suburb_rows.append({
            "suburb": name,
            "centroid_lat": s["lat"],
            "centroid_lon": s["lon"],
            "daily_packages": s["packages"],
            "pct_business": s["stop_type_mix"]["business"],
            "pct_apartment": s["stop_type_mix"]["apartment"],
            "pct_residential": s["stop_type_mix"]["residential"],
        })
    suburb_df = pd.DataFrame(suburb_rows)
    suburb_df.to_csv(os.path.join(OUTPUT_DIR, "suburbs.csv"), index=False)
    print("[2/6] suburbs.csv written")

    # --- Stops ---
    print("\n[3/6] Generating stops...")
    stop_master_df = generate_stops()
    stop_master_df.to_csv(os.path.join(OUTPUT_DIR, "stop_master.csv"), index=False)
    print(f"      stop_master.csv written ({len(stop_master_df):,} rows)")

    # --- Package manifest (single day) ---
    print("\n[4/6] Generating package manifest...")
    manifest_df = generate_package_manifest(stop_master_df, date(2025, 6, 2))
    manifest_df.to_csv(os.path.join(OUTPUT_DIR, "package_manifest.csv"), index=False)
    print(f"      package_manifest.csv written ({len(manifest_df):,} rows)")
    print(f"      Total cube: {manifest_df['cube_litres'].sum():,.0f} litres")
    print(f"      Avg cube per parcel: {manifest_df['cube_litres'].mean():.2f} litres")

    # --- Route history (20 days) ---
    print("\n[5/6] Generating route history (20 days — this may take a minute)...")
    route_history_df, service_time_df = generate_route_history(stop_master_df, manifest_df)
    route_history_df.to_csv(os.path.join(OUTPUT_DIR, "route_history.csv"), index=False)
    service_time_df.to_csv(os.path.join(OUTPUT_DIR, "service_time_log.csv"), index=False)
    print(f"      route_history.csv written ({len(route_history_df):,} rows)")
    print(f"      service_time_log.csv written ({len(service_time_df):,} rows)")

    # --- Travel time sample ---
    print("\n[6/6] Generating travel time sample...")
    tt_df = generate_travel_time_sample(stop_master_df)
    tt_df.to_csv(os.path.join(OUTPUT_DIR, "travel_time_sample.csv"), index=False)
    print(f"      travel_time_sample.csv written ({len(tt_df):,} rows)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles written:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        rows = len(pd.read_csv(fpath))
        print(f"  {fname:<35} {rows:>8,} rows  ({size_kb:>7.1f} KB)")

    # --- Quick stats summary ---
    print("\nKEY STATISTICS")
    print("-" * 40)
    print(f"Total stops:           {len(stop_master_df):>8,}")
    print(f"Packages (day):        {len(manifest_df):>8,}")
    print(f"Avg pkgs/stop:         {len(manifest_df)/len(stop_master_df):>8.2f}")
    print(f"Drivers:               {147:>8,}")
    print(f"Avg pkgs/driver/day:   {len(manifest_df)/147:>8.0f}")
    svc = service_time_df
    print(f"Avg service time (s):  {svc['service_time_seconds'].mean():>8.1f}")
    print(f"Median service time:   {svc['service_time_seconds'].median():>8.1f}")
    print(f"P90 service time:      {svc['service_time_seconds'].quantile(0.90):>8.1f}")
    print(f"Route history days:    {service_time_df['route_date'].nunique():>8}")


if __name__ == "__main__":
    main()
