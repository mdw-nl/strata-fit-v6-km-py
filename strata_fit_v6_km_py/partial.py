import pandas as pd
import numpy as np

from typing import List, Literal, Optional
from vantage6.algorithm.tools.util import info, warn
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.exceptions import InputError

NoiseType = Optional[Literal["NONE", "GAUSSIAN", "POISSON"]]


@data(1)
def get_unique_event_times(
    df: pd.DataFrame,
    time_column_name: str,
    noise_type: NoiseType = "NONE",
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> List[float]:
    """
    Get unique event times from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_column_name : str
        Column name representing time.
    noise_type : str, optional
        Noise type: 'NONE', 'GAUSSIAN', or 'POISSON' (default: 'NONE').
    snr : float, optional
        Signal-to-noise ratio for Gaussian noise.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[float]
        Unique event times.
    """
    info(f"Getting unique event times from '{time_column_name}'.")
    df = _add_noise_to_event_times(df, time_column_name, noise_type, snr, random_seed)
    return df[time_column_name].unique().tolist()


@data(1)
def get_km_event_table(
    df: pd.DataFrame,
    time_column_name: str,
    censor_column_name: str,
    unique_event_times: List[float],
    noise_type: NoiseType = "NONE",
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> str:
    """
    Generate event table for Kaplan-Meier calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_column_name : str
        Time column.
    censor_column_name : str
        Censoring indicator column.
    unique_event_times : list of float
        List of global unique event times.
    noise_type : str, optional
        Noise type.
    snr : float, optional
        Signal-to-noise ratio.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    str
        Event table as JSON string.
    """
    info("Generating Kaplan-Meier event table.")
    df = _add_noise_to_event_times(df, time_column_name, noise_type, snr, random_seed)

    # Aggregate counts
    km_df = (
        df.groupby(time_column_name)
        .agg(
            removed=(censor_column_name, "count"),
            observed=(censor_column_name, "sum")
        )
        .reset_index()
    )
    km_df["censored"] = km_df["removed"] - km_df["observed"]

    # Ensure all global times are present
    km_df = (
        pd.DataFrame({time_column_name: unique_event_times})
        .merge(km_df, on=time_column_name, how="left")
        .fillna(0)
        .sort_values(by=time_column_name)
    )

    # Calculate at-risk counts
    km_df["at_risk"] = km_df["removed"].iloc[::-1].cumsum().iloc[::-1]

    return km_df.to_json()


# === Internal utilities ===

def _add_noise_to_event_times(
    df: pd.DataFrame,
    time_column_name: str,
    noise_type: NoiseType,
    snr: Optional[float],
    random_seed: Optional[int]
) -> pd.DataFrame:
    if noise_type == "NONE" or noise_type is None:
        info("No noise applied.")
        return df

    if random_seed is not None:
        np.random.seed(random_seed)
        info(f"Random seed set to {random_seed}.")

    if noise_type == "GAUSSIAN":
        return _apply_gaussian_noise(df, time_column_name, snr)
    elif noise_type == "POISSON":
        return _apply_poisson_noise(df, time_column_name)
    else:
        raise InputError(f"Unknown noise type: {noise_type}")


def _apply_gaussian_noise(df: pd.DataFrame, time_column_name: str, snr: Optional[float]) -> pd.DataFrame:
    if snr is None or snr <= 0:
        raise InputError("For Gaussian noise, 'snr' (signal-to-noise ratio) must be provided and > 0.")

    variance = np.var(df[time_column_name])
    std_dev = np.sqrt(variance / snr)
    noise = np.random.normal(0, std_dev, size=len(df))
    info(f"Applying Gaussian noise with std dev {std_dev:.4f}.")

    df[time_column_name] += np.round(noise)
    df[time_column_name] = df[time_column_name].clip(lower=0.0)
    return df


def _apply_poisson_noise(df: pd.DataFrame, time_column_name: str) -> pd.DataFrame:
    info("Applying Poisson noise.")
    df[time_column_name] = df[time_column_name].apply(
        lambda x: np.random.poisson(lam=x) if x > 0 else 0
    )
    return df
