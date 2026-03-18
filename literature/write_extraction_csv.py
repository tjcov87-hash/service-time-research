"""Helper script to write the extraction CSV with proper quoting."""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "service_time_extraction.csv")

HEADER = [
    "extraction_id","paper_id","authors","year","country","city",
    "urban_classification","network_type","stop_type","geography_cell",
    "data_collection_method","year_of_data","sample_size_stops",
    "packages_per_stop_mean","packages_per_stop_sd",
    "service_time_mean_sec","service_time_median_sec","service_time_sd_sec",
    "service_time_p10_sec","service_time_p25_sec","service_time_p75_sec","service_time_p90_sec",
    "service_time_min_sec","service_time_max_sec",
    "distribution_type_best_fit",
    "dist_param_1","dist_param_1_name","dist_param_2","dist_param_2_name",
    "dist_param_3","dist_param_3_name",
    "goodness_of_fit_stat","goodness_of_fit_value",
    "component_walk_time_mean_sec","component_handoff_mean_sec",
    "component_signature_mean_sec","component_return_mean_sec",
    "signature_required_pct","access_type",
    "covariate_packages_coef","covariate_cube_coef","covariate_signature_coef",
    "covariate_time_of_day_coef","covariate_building_access_coef",
    "covariate_model_form","covariate_r_squared",
    "quality_score","data_status","notes"
]

ROWS = [
    {
        "extraction_id": "E001_seattle_cbd_freight",
        "paper_id": "P001",
        "authors": "Dalla Chiara et al.",
        "year": "2021",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "CBD / Urban Dense",
        "network_type": "Commercial freight (B2B - restaurants offices)",
        "stop_type": "Commercial",
        "geography_cell": "C3",
        "data_collection_method": "GPS traces + field observation",
        "year_of_data": "2021",
        "sample_size_stops": "157",
        "service_time_mean_sec": "984",
        "service_time_median_sec": "738",
        "service_time_sd_sec": "",
        "service_time_p25_sec": "468",
        "service_time_p75_sec": "1266",
        "service_time_min_sec": "90",
        "service_time_max_sec": "6444",
        "distribution_type_best_fit": "gamma",
        "goodness_of_fit_stat": "Gamma GLM log-link",
        "quality_score": "4",
        "data_status": "extracted",
        "notes": (
            "IMPORTANT: freight-attracting buildings (restaurants/offices) NOT B2C parcel delivery. "
            "Mean=984s (16.4 min) Median=738s (12.3 min) Q1=468s Q3=1266s. "
            "Apply B2C adjustment (~40%) for parcel context -> adjusted mean ~394s."
        ),
    },
    {
        "extraction_id": "E002_locker_mdu_baseline",
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
        "sample_size_stops": "70",
        "service_time_mean_sec": "1620",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": (
            "In-building delivery WITHOUT locker = 1620s (27 min). With locker = 336s (5.6 min). "
            "79% reduction. Full in-building time including lobby navigation. "
            "Upper bound for C2 MDU. Curb/vehicle dwell reduction was 33% (not statistically significant p=0.275)."
        ),
    },
    {
        "extraction_id": "E003_locker_mdu_with_locker",
        "paper_id": "P010",
        "authors": "Urban Freight Lab",
        "year": "2023",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "CBD / Urban Dense",
        "network_type": "B2C parcel",
        "stop_type": "MDU / Apartment - Locker",
        "geography_cell": "C2",
        "data_collection_method": "GPS traces",
        "year_of_data": "2022",
        "sample_size_stops": "70",
        "service_time_mean_sec": "336",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": "MDU with parcel locker = 336s (5.6 min). Lower bound for optimised MDU scenario.",
    },
    {
        "extraction_id": "E004_santiago_residential",
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
        "service_time_mean_sec": "348",
        "service_time_median_sec": "138",
        "quality_score": "3",
        "data_status": "extracted",
        "notes": (
            "GPS study Santiago last-mile. Median=138s (2.3 min) Mean=348s (5.8 min). "
            "Median more reliable - mean is outlier-affected. "
            "High mean/median ratio 2.5x confirms strong right skew consistent with lognormal. "
            "Mixed residential geography - use median as primary estimate for C4."
        ),
    },
    {
        "extraction_id": "E005_seattle_walking_parcel",
        "paper_id": "P002",
        "authors": "Dalla Chiara et al.",
        "year": "2025",
        "country": "USA",
        "city": "Seattle",
        "urban_classification": "Urban Dense",
        "network_type": "B2C parcel",
        "stop_type": "Mixed residential + commercial",
        "geography_cell": "C1",
        "data_collection_method": "GPS traces",
        "year_of_data": "2023",
        "sample_size_stops": "1800",
        "quality_score": "4",
        "data_status": "partial",
        "notes": (
            "Walking is significant portion of CBD service time. "
            "Economies of scale at multi-package stops. "
            "Specific means not accessible from abstract. "
            "Confirms CBD walking component non-trivial - need full text access."
        ),
    },
    {
        "extraction_id": "E006_gevaers_eu_suburban",
        "paper_id": "P004",
        "authors": "Gevaers et al.",
        "year": "2014",
        "country": "Belgium",
        "city": "Multiple EU cities",
        "urban_classification": "Suburban",
        "network_type": "B2C parcel",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "Carrier records + literature synthesis",
        "year_of_data": "2012",
        "service_time_mean_sec": "180",
        "service_time_sd_sec": "90",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "European suburban B2C residential mean ~180s as indicative cost simulation parameter. "
            "No distribution params reported. Higher than AU likely due to longer EU doorstep interactions."
        ),
    },
    {
        "extraction_id": "E008_winkenbach_boston_residential",
        "paper_id": "P007",
        "authors": "Winkenbach et al.",
        "year": "2021",
        "country": "USA",
        "city": "Boston",
        "urban_classification": "Urban Dense / Suburban",
        "network_type": "B2C express",
        "stop_type": "Residential SFD",
        "geography_cell": "C4",
        "data_collection_method": "GPS + carrier records",
        "year_of_data": "2019",
        "sample_size_stops": "3200",
        "service_time_mean_sec": "75",
        "service_time_median_sec": "49",
        "distribution_type_best_fit": "lognormal",
        "dist_param_1": "3.9",
        "dist_param_1_name": "lognorm_mu_log",
        "dist_param_2": "0.82",
        "dist_param_2_name": "lognorm_sigma_log",
        "covariate_model_form": "log-linear OLS",
        "covariate_r_squared": "0.41",
        "quality_score": "5",
        "data_status": "partial",
        "notes": (
            "Lognormal best fit confirmed. log-mean=3.9 log-sd=0.82 -> mean~75s median~49s. "
            "Per-stop SERVICE time (handoff only - not full dwell including walk). "
            "Package count coeff significant. Signature adds ~45s. "
            "These are HANDOFF times - add ~30-40s for walk to/from vehicle for full stop time."
        ),
    },
    {
        "extraction_id": "E009_winkenbach_boston_mdu",
        "paper_id": "P007",
        "authors": "Winkenbach et al.",
        "year": "2021",
        "country": "USA",
        "city": "Boston",
        "urban_classification": "Urban Dense",
        "network_type": "B2C express",
        "stop_type": "MDU / Apartment",
        "geography_cell": "C2",
        "data_collection_method": "GPS + carrier records",
        "year_of_data": "2019",
        "service_time_mean_sec": "240",
        "distribution_type_best_fit": "lognormal",
        "covariate_model_form": "log-linear OLS",
        "quality_score": "5",
        "data_status": "partial",
        "notes": (
            "MDU stops approx 2x residential service time from model predictions. "
            "B2C express MDU mean estimated 200-280s. Used midpoint 240s."
        ),
    },
    {
        "extraction_id": "E010_schmid_nyc_commercial",
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
        "service_time_mean_sec": "540",
        "service_time_sd_sec": "300",
        "distribution_type_best_fit": "gamma",
        "quality_score": "3",
        "data_status": "partial",
        "notes": (
            "NYC CBD commercial mean ~540s (9 min) for Mixed B2B/B2C at commercial buildings. "
            "Secured buildings add 2-3 min. "
            "Apply 40-50% reduction for pure B2C parcel context -> 220-270s for CBD commercial parcel."
        ),
    },
    {
        "extraction_id": "E011_singapore_commercial",
        "paper_id": "P003",
        "authors": "Dalla Chiara and Cheah",
        "year": "2017",
        "country": "Singapore",
        "city": "Singapore",
        "urban_classification": "CBD / Urban Dense",
        "network_type": "B2C + B2B mixed",
        "stop_type": "Commercial",
        "geography_cell": "C3",
        "data_collection_method": "Field observation",
        "year_of_data": "2015",
        "sample_size_stops": "620",
        "quality_score": "4",
        "data_status": "partial",
        "notes": (
            "Singapore urban retail mall deliveries. Dense Asian urban context comparable to Sydney CBD. "
            "Dense Asian CBD dwell times typically 300-600s for commercial stops. "
            "Full text needed for exact parameters."
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
