import pandas as pd
import numpy as np

from typing import List, Optional
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.exceptions import InputError

from .types import NoiseType, EventType
from .utils import add_noise_to_event_times

@data(1)
def get_unique_event_times(
    df: pd.DataFrame,
    interval_start_column_name: str,
    interval_end_column_name: str,
    noise_type: NoiseType = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> List[float]:
    """
    Collect unique event times from both interval start and end columns.
    """
    info("Collecting unique event times.")
    df = add_noise_to_event_times(df, interval_start_column_name, noise_type, snr, random_seed)
    df = add_noise_to_event_times(df, interval_end_column_name, noise_type, snr, random_seed)

    unique_times = pd.concat([
        df[interval_start_column_name],
        df[interval_end_column_name]
    ]).dropna().unique()

    return unique_times.tolist()


@data(1)
def get_km_event_table(
    df: pd.DataFrame,
    interval_start_column_name: str,
    interval_end_column_name: str,
    event_indicator_column_name: str,
    unique_event_times: List[float],
    noise_type: NoiseType = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> str:
    """
    Generate event table for Kaplan-Meier calculation with interval censoring.
    """
    info("Generating KM event table with interval censoring support.")

    df = add_noise_to_event_times(df, interval_start_column_name, noise_type, snr, random_seed)
    df = add_noise_to_event_times(df, interval_end_column_name, noise_type, snr, random_seed)

    event_table = pd.DataFrame({interval_start_column_name: unique_event_times})

    # Exact events
    exact_events = df[df[event_indicator_column_name] == EventType.EXACT.value]
    event_counts = exact_events[interval_start_column_name].value_counts().reindex(unique_event_times, fill_value=0)

    # Right censored
    censored_events = df[df[event_indicator_column_name] == EventType.CENSORED.value]
    censored_counts = censored_events[interval_start_column_name].value_counts().reindex(unique_event_times, fill_value=0)

    # Interval censored
    interval_events = df[df[event_indicator_column_name] == EventType.INTERVAL.value]
    interval_counts = interval_events[interval_end_column_name].value_counts().reindex(unique_event_times, fill_value=0)

    # Assemble the table
    event_table["removed"] = event_counts + censored_counts + interval_counts
    event_table["observed"] = event_counts
    event_table["interval"] = interval_counts
    event_table["censored"] = censored_counts

    # Calculate at risk using cumulative sum from last time downwards
    event_table["at_risk"] = event_table["removed"].iloc[::-1].cumsum().iloc[::-1]

    return event_table.to_json()
