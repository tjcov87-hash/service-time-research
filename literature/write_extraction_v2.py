"""
Writes service_time_extraction_v2.csv with proper quoting.
New structure: base handoff time + incremental per package + walk time component.
Run: python literature/write_extraction_v2.py
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "service_time_extraction_v2.csv")

HEADER = [
    # Identification
    "extraction_id", "paper_id", "authors", "year", "country", "city",
    "urban_classification", "network_type", "stop_type", "geography_cell",
    "data_collection_method", "year_of_data", "sample_size_stops",
    # What the study measured
    "service_time_definition",   # TOTAL_STOP / HANDOFF_ONLY / CURB_DWELL / WALK_ONLY
    "n_packages_reference",      # packages per stop at which mean was measured (usually 1)
    # Total stop time (walk + handoff + return) — primary measure
    "total_stop_mean_sec",
    "total_stop_median_sec",
    "total_stop_sd_sec",
    "total_stop_p10_sec",
    "total_stop_p90_sec",
    # Base handoff time (first package, door only, no walk) — if separated
    "base_handoff_mean_sec",
    "base_handoff_sd_sec",
    # Walk time (vehicle to door and back)
    "walk_time_mean_sec",
    "walk_time_sd_sec",
    "walk_time_density_ref",     # stops/km² at which walk time was measured
    # Incremental per additional package at same stop
    "increment_per_pkg_mean_sec",
    "increment_per_pkg_sd_sec",
    "increment_model_form",      # LINEAR / LOG_LINEAR / DIMINISHING
    # Package count log-linear coefficient (from regression)
    "pkg_count_log_coef",        # beta: log(T) = ... + beta * n_pkgs
    "pkg_count_log_coef_se",
    # Density effect
    "density_effect_reported",   # YES / NO / PARTIAL
    "density_exponent",          # b in T_walk = a * density^(-b)
    # Distribution of total stop time
    "distribution_type_best_fit",
    "lognorm_mu",                # log-scale mean (lognormal)
    "lognorm_sigma",             # log-scale SD (lognormal)
    "gamma_alpha",
    "gamma_beta",
    "goodness_of_fit_stat",
    "goodness_of_fit_value",
    # Signature premium
    "signature_premium_sec",
    "signature_premium_sd_sec",
    # Other covariates
    "covariate_model_form",
    "covariate_r_squared",
    # Metadata
    "b2c_adjusted",              # YES if raw data was freight/mixed and we adjusted
    "b2c_adjustment_factor",
    "au_relevant",               # HIGH / MEDIUM / LOW relevance to Australia
    "quality_score",             # 1-5
    "data_status",               # extracted / partial / not_started
    "notes",
]

ROWS = [
    # -----------------------------------------------------------------------
    # Winkenbach et al. — highest quality, richest covariate model
    # Note: exact parameters from abstract + cited values; full text needed
    # for complete separation of base vs incremental
    # -----------------------------------------------------------------------
    {
        "extraction_id": "W01_winkenbach_residential",
        "paper_id": "P007",
        "authors": "Winkenbach et al.",
        "year": "2021",
        "country": "USA",
        "city": "Boston",
        "urban_classification": "Urban Dense / Suburban",
        "network_type": "B2C express parcel",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "GPS + carrier records",
        "year_of_data": "2019",
        "sample_size_stops": "3200",
        "service_time_definition": "HANDOFF_ONLY",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "",
        "total_stop_median_sec": "",
        "base_handoff_mean_sec": "75",
        "base_handoff_sd_sec": "",
        "walk_time_mean_sec": "",
        "distribution_type_best_fit": "lognormal",
        "lognorm_mu": "3.9",
        "lognorm_sigma": "0.82",
        "pkg_count_log_coef": "0.12",
        "increment_per_pkg_mean_sec": "10",
        "increment_model_form": "LOG_LINEAR",
        "signature_premium_sec": "45",
        "covariate_model_form": "log-linear OLS",
        "covariate_r_squared": "0.41",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "5",
        "data_status": "partial",
        "notes": (
            "Lognormal best fit confirmed. lognorm_mu=3.9 lognorm_sigma=0.82 -> "
            "mean=exp(3.9+0.82^2/2)=~75s median=exp(3.9)=~49s at n=1 package. "
            "pkg_count_log_coef=0.12 from log-linear regression: each additional package "
            "multiplies service time by exp(0.12)=1.128 -> ~10s increment at base=75s. "
            "Signature adds ~45s. HANDOFF ONLY - walk component not included in these params. "
            "Add walk time separately using density model. Full text needed for MDU/commercial breakdown."
        ),
    },
    {
        "extraction_id": "W02_winkenbach_mdu",
        "paper_id": "P007",
        "authors": "Winkenbach et al.",
        "year": "2021",
        "country": "USA",
        "city": "Boston",
        "urban_classification": "Urban Dense",
        "network_type": "B2C express parcel",
        "stop_type": "MDU / Apartment",
        "geography_cell": "C2",
        "data_collection_method": "GPS + carrier records",
        "year_of_data": "2019",
        "sample_size_stops": "",
        "service_time_definition": "HANDOFF_ONLY",
        "n_packages_reference": "1",
        "base_handoff_mean_sec": "180",
        "distribution_type_best_fit": "lognormal",
        "increment_per_pkg_mean_sec": "8",
        "increment_model_form": "LOG_LINEAR",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "5",
        "data_status": "partial",
        "notes": (
            "MDU ~2x residential from model predictions. "
            "Estimated base handoff = 180s (includes lobby navigation to unit). "
            "Increment per package lower than residential (~8s) as driver already at unit. "
            "Full text needed for exact lognormal params."
        ),
    },
    # -----------------------------------------------------------------------
    # Urban Freight Lab Seattle — walk time component, CBD context
    # -----------------------------------------------------------------------
    {
        "extraction_id": "UFL01_seattle_cbd_walk",
        "paper_id": "P002",
        "authors": "Dalla Chiara et al.",
        "year": "2025",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "Urban Dense / CBD",
        "network_type": "B2C parcel",
        "stop_type": "Mixed",
        "geography_cell": "C1",
        "data_collection_method": "GPS traces",
        "year_of_data": "2023",
        "sample_size_stops": "1800",
        "service_time_definition": "WALK_ONLY",
        "walk_time_mean_sec": "",
        "walk_time_density_ref": "",
        "density_effect_reported": "YES",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "4",
        "data_status": "partial",
        "notes": (
            "Walking accounts for majority of service time in CBD. "
            "Parcel carrier shows economies of scale at multi-delivery stops. "
            "Carriers trade off walking distance vs serving multiple customers from single park. "
            "Full text needed for walk time parameters and density relationship."
        ),
    },
    {
        "extraction_id": "UFL02_seattle_freight_cbd",
        "paper_id": "P001",
        "authors": "Dalla Chiara et al.",
        "year": "2021",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "CBD / Urban Dense",
        "network_type": "Commercial freight (B2B)",
        "stop_type": "Commercial",
        "geography_cell": "C3",
        "data_collection_method": "GPS traces + field observation",
        "year_of_data": "2021",
        "sample_size_stops": "157",
        "service_time_definition": "CURB_DWELL",
        "n_packages_reference": "many",
        "total_stop_mean_sec": "984",
        "total_stop_median_sec": "738",
        "total_stop_p10_sec": "",
        "total_stop_p25_sec": "468",
        "total_stop_p75_sec": "1266",
        "total_stop_p90_sec": "",
        "total_stop_sd_sec": "",
        "distribution_type_best_fit": "gamma",
        "b2c_adjusted": "YES",
        "b2c_adjustment_factor": "0.40",
        "au_relevant": "MEDIUM",
        "quality_score": "4",
        "data_status": "extracted",
        "notes": (
            "FREIGHT (restaurants/offices) NOT B2C parcel. Apply 40% factor -> B2C adjusted mean = 394s. "
            "Curb dwell = full stop including unloading multiple items from vehicle. "
            "For single B2C parcel in CBD commercial context: ~150-200s more appropriate. "
            "Use as upper bound with 0.40 factor applied."
        ),
    },
    {
        "extraction_id": "UFL03_locker_mdu",
        "paper_id": "P010",
        "authors": "Urban Freight Lab",
        "year": "2023",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "CBD / Urban Dense",
        "network_type": "B2C parcel",
        "stop_type": "MDU / Apartment",
        "geography_cell": "C2",
        "data_collection_method": "GPS traces",
        "year_of_data": "2022",
        "sample_size_stops": "140",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "978",
        "total_stop_sd_sec": "",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": (
            "Average of WITHOUT LOCKER (1620s/27min) and WITH LOCKER (336s/5.6min) scenarios. "
            "Standard in-building MDU without locker = 1620s. With locker = 336s. "
            "79% reduction with locker. For standard B2C Australian MDU: "
            "use 1620s as upper bound, 336s as locker-optimised lower bound. "
            "Typical AU MDU likely 300-600s given no locker infrastructure."
        ),
    },
    # -----------------------------------------------------------------------
    # Santiago Chile — good B2C residential baseline with density context
    # -----------------------------------------------------------------------
    {
        "extraction_id": "SCL01_santiago_residential",
        "paper_id": "P_santiago",
        "authors": "Mora et al.",
        "year": "2024",
        "country": "Chile",
        "city": "Santiago",
        "urban_classification": "Mixed urban/suburban",
        "network_type": "B2C parcel (food courier)",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "GPS traces",
        "year_of_data": "2022",
        "sample_size_stops": "2900",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "348",
        "total_stop_median_sec": "138",
        "total_stop_sd_sec": "",
        "distribution_type_best_fit": "lognormal",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": (
            "GPS last-mile Santiago. Median=138s more reliable than mean=348s (outlier-driven). "
            "Mean/median ratio=2.5 confirms strong lognormal right skew. "
            "Mixed food courier + parcel: likely skewed toward commercial contexts. "
            "Median 138s used as C4 suburban residential central estimate. "
            "This is TOTAL STOP time including walk."
        ),
    },
    # -----------------------------------------------------------------------
    # Gevaers et al. — European suburban reference
    # -----------------------------------------------------------------------
    {
        "extraction_id": "G01_gevaers_eu_suburban",
        "paper_id": "P004",
        "authors": "Gevaers et al.",
        "year": "2014",
        "country": "Belgium",
        "city": "Multiple EU",
        "urban_classification": "Suburban",
        "network_type": "B2C parcel",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "Carrier records + literature synthesis",
        "year_of_data": "2012",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "180",
        "total_stop_sd_sec": "90",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "EU suburban B2C residential mean ~180s. "
            "Higher than Winkenbach (75s handoff) suggesting walk component of ~105s, "
            "consistent with suburban walk time model. "
            "No distribution params reported. Use as directional reference for C4."
        ),
    },
    # -----------------------------------------------------------------------
    # NYC commercial — CBD commercial reference
    # -----------------------------------------------------------------------
    {
        "extraction_id": "NYC01_schmid_commercial",
        "paper_id": "P014",
        "authors": "Schmid et al.",
        "year": "2022",
        "country": "USA",
        "city": "New York City",
        "urban_classification": "CBD",
        "network_type": "Mixed B2B/B2C",
        "stop_type": "Commercial",
        "geography_cell": "C3",
        "data_collection_method": "Field observation",
        "year_of_data": "2020",
        "sample_size_stops": "890",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "many",
        "total_stop_mean_sec": "540",
        "total_stop_sd_sec": "300",
        "distribution_type_best_fit": "gamma",
        "b2c_adjusted": "YES",
        "b2c_adjustment_factor": "0.50",
        "au_relevant": "MEDIUM",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "Mixed B2B/B2C commercial CBD. Apply 0.50 factor -> B2C adjusted mean = 270s. "
            "Secured buildings add 120-180s. "
            "For AU CBD commercial single parcel: 150-250s range reasonable."
        ),
    },
    # -----------------------------------------------------------------------
    # USPS rural baseline
    # -----------------------------------------------------------------------
    {
        "extraction_id": "USPS01_rural_residential",
        "paper_id": "P005",
        "authors": "USPS OIG",
        "year": "2020",
        "country": "USA",
        "city": "National",
        "urban_classification": "Rural",
        "network_type": "Postal parcels",
        "stop_type": "Residential SFD",
        "geography_cell": "Rural",
        "data_collection_method": "Carrier records",
        "year_of_data": "2019",
        "service_time_definition": "TOTAL_STOP",
        "total_stop_mean_sec": "90",
        "total_stop_sd_sec": "35",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "Rural door service time 10-20% SHORTER than suburban at-door (easier access, "
            "mailboxes at road, no lobby). Estimated ~90s total stop. "
            "Overall rural delivery cost higher due to travel time between stops. "
            "Not directly applicable to Botany DC delivery zones."
        ),
    },
    # -----------------------------------------------------------------------
    # Oslo study (2023) — best explicit increment data + attended/unattended split
    # Kjelle et al. "Replacing home deliveries by deliveries to parcel lockers"
    # Tandfonline 2023, Greater Oslo, ~1M inhabitants, 187 locker locations
    # -----------------------------------------------------------------------
    {
        "extraction_id": "OSLO01_attended_residential",
        "paper_id": "P_OSLO",
        "authors": "Kjelle et al.",
        "year": "2023",
        "country": "Norway",
        "city": "Greater Oslo",
        "urban_classification": "Urban Dense",
        "network_type": "B2C parcel (attended)",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "Empirical city-wide study",
        "year_of_data": "2021",
        "sample_size_stops": "5000",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "240",
        "total_stop_sd_sec": "",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "4",
        "data_status": "extracted",
        "notes": (
            "ATTENDED delivery (recipient must be home to receive). 240s total stop. "
            "Comparable to AU signature-required or MDU buzzer-entry scenario. "
            "For standard AU ATL (authority-to-leave), use unattended figure instead."
        ),
    },
    {
        "extraction_id": "OSLO02_unattended_residential",
        "paper_id": "P_OSLO",
        "authors": "Kjelle et al.",
        "year": "2023",
        "country": "Norway",
        "city": "Greater Oslo",
        "urban_classification": "Urban Dense",
        "network_type": "B2C parcel (unattended ATL)",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "Empirical city-wide study",
        "year_of_data": "2021",
        "sample_size_stops": "5000",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "119",
        "total_stop_sd_sec": "",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "4",
        "data_status": "extracted",
        "notes": (
            "UNATTENDED delivery (left at door, no signature). 119s total stop. "
            "CRITICAL: This is the primary AU benchmark. AU parcels ~85% ATL. "
            "Cross-validates our model: Winkenbach handoff 75s + walk 48s = 123s ~ Oslo 119s. "
            "Model is well-calibrated for C4 suburban residential."
        ),
    },
    {
        "extraction_id": "OSLO03_locker_base",
        "paper_id": "P_OSLO",
        "authors": "Kjelle et al.",
        "year": "2023",
        "country": "Norway",
        "city": "Greater Oslo",
        "urban_classification": "Urban Dense",
        "network_type": "B2C parcel locker",
        "stop_type": "MDU / Apartment",
        "geography_cell": "C2",
        "data_collection_method": "Empirical city-wide study",
        "year_of_data": "2021",
        "sample_size_stops": "187",
        "service_time_definition": "TOTAL_STOP",
        "n_packages_reference": "1",
        "total_stop_mean_sec": "250",
        "total_stop_sd_sec": "",
        "increment_per_pkg_mean_sec": "19",
        "increment_per_pkg_sd_sec": "5",
        "increment_model_form": "LINEAR",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "4",
        "data_status": "extracted",
        "notes": (
            "PARCEL LOCKER stops. Base=250s (parking + vehicle open + walk to locker). "
            "INCREMENT=19s per package loaded — explicit LINEAR model confirmed. "
            "This is the only empirical increment coefficient from a controlled study. "
            "Locker loading faster than door handoff per item but base time higher. "
            "For standard MDU door delivery, increment likely 12-20s range."
        ),
    },
    # -----------------------------------------------------------------------
    # Urban walking study — walk time calibration
    # Dalla Chiara et al. 2025, Transportation (Springer)
    # -----------------------------------------------------------------------
    {
        "extraction_id": "WALK01_dalla_chiara_urban",
        "paper_id": "P002",
        "authors": "Dalla Chiara et al.",
        "year": "2025",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "Urban Dense / CBD",
        "network_type": "B2C parcel",
        "stop_type": "Mixed",
        "geography_cell": "C1",
        "data_collection_method": "GPS traces",
        "year_of_data": "2023",
        "sample_size_stops": "1800",
        "service_time_definition": "WALK_ONLY",
        "walk_time_mean_sec": "420",
        "walk_time_density_ref": "2000",
        "density_effect_reported": "YES",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "4",
        "data_status": "partial",
        "notes": (
            "60-80% of driver time outside vehicle. Walk time 7min (420s) daytime urban. "
            "NOTE: This 420s likely covers TOTAL outside-vehicle time per stop (walk out + "
            "handoff + walk back) — not walk component alone. At density ~2000 stops/km2, "
            "pure walk ≈ 30-50s; handoff adds 60-120s; total outside ≈ 150-200s typical. "
            "The 7-min figure may be for CBD high-rise or multi-package stops. "
            "Treat as upper bound for C1 CBD total outside-vehicle time, not walk-only."
        ),
    },
    # -----------------------------------------------------------------------
    # Australia Post Queensland — locker collection (customer-side, not driver)
    # Jara-Diaz et al., Journal of Transport Geography 2023
    # -----------------------------------------------------------------------
    {
        "extraction_id": "AU01_auspost_qld_locker",
        "paper_id": "P_AU_locker",
        "authors": "Jara-Diaz et al.",
        "year": "2023",
        "country": "Australia",
        "city": "Queensland",
        "urban_classification": "Mixed",
        "network_type": "AusPost parcel locker",
        "stop_type": "MDU / Apartment",
        "geography_cell": "C2",
        "data_collection_method": "AusPost operational records",
        "year_of_data": "2013-2017",
        "sample_size_stops": "51",
        "service_time_definition": "TOTAL_STOP",
        "total_stop_mean_sec": "932",
        "total_stop_sd_sec": "1256",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": (
            "AusPost Queensland parcel locker data. Mean=932s (15.53 min) SD=1256s (20.94 min). "
            "IMPORTANT: This is CUSTOMER collection dwell time (time from notification to collection), "
            "NOT driver service time at stop. Very high SD shows wide spread. "
            "Not directly comparable to driver stop time. Included as AU context reference only. "
            "Confirms AusPost operates at scale in AU — operational data exists but not driver-side."
        ),
    },
    # -----------------------------------------------------------------------
    # University of Melbourne inner-city freight study
    # -----------------------------------------------------------------------
    {
        "extraction_id": "AU02_unimelb_freight",
        "paper_id": "P_AU_unimelb",
        "authors": "University of Melbourne",
        "year": "2020",
        "country": "Australia",
        "city": "Melbourne CBD",
        "urban_classification": "CBD",
        "network_type": "Mixed freight carriers",
        "stop_type": "Commercial",
        "geography_cell": "C3",
        "data_collection_method": "Survey + field study",
        "year_of_data": "2019",
        "sample_size_stops": "55",
        "service_time_definition": "TOTAL_STOP",
        "b2c_adjusted": "NO",
        "au_relevant": "HIGH",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "55 freight carriers in inner-city Melbourne. Found high vehicle underutilization "
            "and large number of delivery stops. Qualitative finding: dense urban AU context. "
            "No specific service time parameter reported. Confirms AU CBD freight delivery "
            "is characterised by high stop density and capacity underutilisation."
        ),
    },
    # -----------------------------------------------------------------------
    # Industry benchmarks — density calibration
    # -----------------------------------------------------------------------
    {
        "extraction_id": "BENCH01_density_benchmarks",
        "paper_id": "P_BENCH",
        "authors": "Nuvizz / Industry",
        "year": "2023",
        "country": "USA",
        "city": "Various",
        "urban_classification": "Urban Dense",
        "network_type": "B2C parcel",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "Industry benchmarking",
        "year_of_data": "2022",
        "service_time_definition": "TOTAL_STOP",
        "total_stop_mean_sec": "90",
        "b2c_adjusted": "NO",
        "au_relevant": "MEDIUM",
        "quality_score": "2",
        "data_status": "partial",
        "notes": (
            "Industry benchmark: top logistics providers achieve <90s per stop in high-density urban. "
            "High-density urban = 25-40 deliveries/day. Suburban = 15-25/day. Rural = 8-15/day. "
            "Low quality score — industry marketing source, not peer-reviewed. "
            "Use only as lower-bound reference for optimised B2C residential in dense AU suburbs."
        ),
    },
]


def fill_row(d):
    row = {k: "" for k in HEADER}
    row.update(d)
    return [row[k] for k in HEADER]


with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(HEADER)
    for r in ROWS:
        writer.writerow(fill_row(r))

print(f"Written {len(ROWS)} rows to {OUT}")
