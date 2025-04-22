import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

# Number of rows
n = 160

# Define event type distribution
event_types = ["exact", "censored", "interval"]
probs = [0.6, 0.3, 0.1]

# Generate start times uniformly between 0 and 50
interval_start = np.round(np.random.uniform(0, 50, size=n), 2)

# Generate event types
event_type = np.random.choice(event_types, size=n, p=probs)

# Generate end times: for exact & censored, end = start; for interval, end > start
interval_end = []
for start, et in zip(interval_start, event_type):
    if et == "interval":
        # draw a positive increment up to 10
        increment = np.round(np.random.uniform(0.01, 10.0), 2)
        interval_end.append(np.round(start + increment, 2))
    else:
        interval_end.append(start)

df = pd.DataFrame({
    "interval_start": interval_start,
    "interval_end": interval_end,
    "event_type": event_type
}).to_csv("tests/data/synthetic_times.csv")

