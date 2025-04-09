import pandas as pd
from typing import Dict, List, Union, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation

@algorithm_client
def kaplan_meier_central(
    client: AlgorithmClient,
    interval_start_column_name: str,
    interval_end_column_name: str,
    event_indicator_column_name: str,
    organizations_to_include: Optional[List[int]] = None,
    noise_type: Optional[str] = None,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Central orchestration of federated Kaplan-Meier algorithm with interval censoring.

    Parameters
    ----------
    client : AlgorithmClient
        Vantage6 client.
    interval_start_column_name : str
        Column name for interval start time.
    interval_end_column_name : str
        Column name for interval end time.
    event_indicator_column_name : str
        Column name indicating event type ('exact', 'censored', 'interval').
    organizations_to_include : list of int, optional
        List of organization IDs to include.
    noise_type : str, optional
        Noise type ('NONE', 'GAUSSIAN', 'POISSON').
    snr : float, optional
        Signal-to-noise ratio for Gaussian noise.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Kaplan-Meier event table as JSON.
    """
    if not organizations_to_include:
        organizations_to_include = [org["id"] for org in client.organization.list()]

    MINIMUM_ORGANIZATIONS = 3
    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(
            f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS})."
        )

    info("Step 1: Collecting unique event times.")
    unique_event_times_results = _start_partial_and_collect_results(
        client,
        method="get_unique_event_times",
        organizations_to_include=organizations_to_include,
        interval_start_column_name=interval_start_column_name,
        interval_end_column_name=interval_end_column_name,
        noise_type=noise_type,
        snr=snr,
        random_seed=random_seed,
    )

    unique_event_times = set()
    for result in unique_event_times_results:
        unique_event_times.update(result)
    unique_event_times = sorted(unique_event_times)

    info("Step 2: Collecting local event tables.")
    local_event_tables_results = _start_partial_and_collect_results(
        client,
        method="get_km_event_table",
        organizations_to_include=organizations_to_include,
        interval_start_column_name=interval_start_column_name,
        interval_end_column_name=interval_end_column_name,
        event_indicator_column_name=event_indicator_column_name,
        unique_event_times=unique_event_times,
        noise_type=noise_type,
        snr=snr,
        random_seed=random_seed,
    )

    local_event_tables = [pd.read_json(result) for result in local_event_tables_results]

    info("Step 3: Aggregating local event tables.")
    km_df = pd.concat(local_event_tables).groupby(interval_start_column_name, as_index=False).sum()

    # Calculate hazard rate with half-contribution for interval-censored events
    km_df["hazard"] = (km_df["observed"] + km_df["interval"] * 0.5) / km_df["at_risk"]

    # Cumulative survival (Kaplan-Meier curve)
    km_df["survival_cdf"] = (1 - km_df["hazard"]).cumprod()

    info("Kaplan-Meier curve with interval censoring computed.")
    return km_df.to_json()

def _start_partial_and_collect_results(
    client: AlgorithmClient,
    method: str,
    organizations_to_include: List[int],
    **kwargs,
) -> List[Dict]:
    """
    Helper to start a partial task and collect the results.

    Parameters
    ----------
    client : AlgorithmClient
        Vantage6 client.
    method : str
        Method name to execute.
    organizations_to_include : list of int
        Target organizations.
    **kwargs
        Additional parameters for the partial task.

    Returns
    -------
    list of dict
        Results from organizations.
    """
    info(f"Starting partial task '{method}' with {len(organizations_to_include)} organizations.")
    task = client.task.create(
        input_={"method": method, "kwargs": kwargs},
        organizations=organizations_to_include,
    )

    info("Waiting for results...")
    results = client.wait_for_results(task_id=task["id"])
    info(f"Results for '{method}' received.")
    return results
