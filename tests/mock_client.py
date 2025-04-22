import warnings
import os
import pandas as pd

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

warnings.filterwarnings("ignore")

# --- 1. Define the per‐node datasets (raw STRATA‑FIT CSVs) ---
path = "data_times/alpha.csv"
dataset1 = {"database": os.path.join("tests", "data", path), "db_type": "csv"}
dataset2 = {"database": os.path.join("tests", "data", path),  "db_type": "csv"}
dataset3 = {"database": os.path.join("tests", "data", path), "db_type": "csv"}

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
print(df_km[["at_risk", "observed", "censored", "interval", "hazard", "survival_cdf"]].describe())

# Example assertion (ensure we have at least one time‐point and survival_cdf is ≤1):
assert not df_km.empty
assert df_km["survival_cdf"].max() <= 1.0

print("\n✅ Central Kaplan–Meier test completed successfully.")
