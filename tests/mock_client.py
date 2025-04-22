import warnings
import os
import pandas as pd
from pathlib import Path

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

warnings.filterwarnings("ignore")

from strata_fit_v6_km_py.types import DEFAULT_INTERVAL_START_COLUMN, DEFAULT_CUMULATIVE_INCIDENCE_COLUMN

def plot_km_curve(km_df):
    import matplotlib.pyplot as plt
    # convert months → years
    years = km_df["interval_start"] / 12

    plt.figure(figsize=(8, 5))
    plt.step(
        years,
        km_df["cumulative_incidence"],   # or use 1 - survival_probability if you only stored survival
        where="post",
        lw=2
    )
    plt.xlabel("Years from diagnosis")
    plt.ylabel("Cumulative incidence of D2T‑RA")
    plt.title("Cumulative incidence of difficult‑to‑treat RA")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 1. Define the per‐node datasets (raw STRATA‑FIT CSVs) ---
data_directory = Path("tests/data/data_times")
dataset1 = {"database": data_directory / "alpha.csv", "db_type": "csv"}
dataset2 = {"database": data_directory / "beta.csv",  "db_type": "csv"}
dataset3 = {"database": data_directory / "gamma.csv", "db_type": "csv"}

# We have three “organizations” in this mock run:
org_ids = [0, 1, 2]

# --- 2. Instantiate the mock client with our module name ---
#    Make sure `module` here matches the name in your setup.py (i.e. the package name).
client = MockAlgorithmClient(
    datasets=[[dataset1], [dataset2], [dataset3]],
    organization_ids=org_ids,
    module="strata_fit_v6_km_py"
)

# --- 3. Trigger the central orchestration ---
# Only send the “master” task to one org; the central function will fan out
# to all three under the hood.
task = client.task.create(
    input_={
        "method": "kaplan_meier_central",
        "kwargs": {
            # you can override noise parameters here if you like,
            # e.g. "noise_type": "GAUSSIAN", "snr": 10, "random_seed": 42
        }
    },
    organizations=[org_ids[0]]
)

# --- 4. Collect and parse the result ---
results_json = client.result.get(task["id"])
df_km = pd.read_json(results_json)

# --- 5. Inspect / assert ---
print("Kaplan–Meier curve (first 5 rows):")
print(df_km.head(), "\n")

print("Summary statistics:")
print(df_km[["at_risk", "observed", "censored", "interval", "hazard", DEFAULT_CUMULATIVE_INCIDENCE_COLUMN]].describe())

# Example assertion (ensure we have at least one time‐point and survival_cdf is ≤1):
assert not df_km.empty
assert df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN].max() <= 1.0

# plotting
plot_km_curve(df_km)

print("\n✅ Central Kaplan–Meier test completed successfully.")
