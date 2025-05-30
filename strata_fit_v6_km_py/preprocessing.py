"""
Preprocessing functions to transform raw STRATA-FIT data
into standardized interval survival data for federated KM.
"""

import pandas as pd
import numpy as np
from .types import (
    EventType, 
    DEFAULT_INTERVAL_START_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_EVENT_INDICATOR_COLUMN
)

def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    # Sort data by patient ID and follow-up time
    df.sort_values(['pat_ID', 'Visit_months_from_diagnosis'], inplace=True)

    # Step 1: Cumulative treatment counts
    df['cum_bDMARD'] = df.groupby('pat_ID')['bDMARD'].cumsum().fillna(0)
    df['cum_tsDMARD'] = df.groupby('pat_ID')['tsDMARD'].cumsum().fillna(0)
    df['cum_btsDMARD'] = df['cum_bDMARD'] + df['cum_tsDMARD']
    df['cum_btsDMARDmin'] = df.groupby('pat_ID')['cum_btsDMARD'].cummin()

    # Rolling average DAS28 (optional improvement)
    df['rolling_avg_DAS28'] = df.groupby('pat_ID')['DAS28'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Step 2: Define criteria for D2T RA
    df['D2T_crit1'] = df['cum_btsDMARD'] > 1
    df['D2T_crit2'] = (df['DAS28'] > 3.2) | (df['rolling_avg_DAS28'] > 3.2)
    df['D2T_crit3'] = (df['Pat_global'] > 50) | (df['Ph_global'] > 50)

    df['D2T_RA'] = df['D2T_crit1'] & df['D2T_crit2'] & df['D2T_crit3']

    # Step 3: Per-patient summary
    summary = df.groupby('pat_ID').agg(
        Year_diagnosis=('Year_diagnosis', 'first'),
        D2T_RA_Ever=('D2T_RA', 'max'),
        cum_btsDMARDmin=('cum_btsDMARDmin', 'max'),
        minFU=('Visit_months_from_diagnosis', 'min'),
        TTE=('Visit_months_from_diagnosis', lambda x: x[df.loc[x.index, 'D2T_RA']].min() if any(df.loc[x.index, 'D2T_RA']) else np.nan),
        maxFU=('Visit_months_from_diagnosis', 'max')
    ).reset_index()

    summary['D2T_RA_Ever'] = summary['D2T_RA_Ever'].fillna(0)
    summary['cum_btsDMARDmin'] = summary['cum_btsDMARDmin'].fillna(0)
    summary['TTE'] = summary['TTE'].fillna(summary['maxFU'])

    # Step 4: Define censoring type
    summary['cens'] = np.select(
        condlist=[
            (summary['D2T_RA_Ever'] == 1) & (summary['cum_btsDMARDmin'] > 2),
            (summary['D2T_RA_Ever'] == 0)
        ],
        choicelist=['interval', 'right'],
        default='no'
    )

    # Step 5: Define interval start, interval end, and event type
    summary[DEFAULT_INTERVAL_START_COLUMN] = np.where(summary['cens'] == 'interval', 0, summary['TTE'])
    summary[DEFAULT_INTERVAL_END_COLUMN] = np.where(summary['cens'] == 'interval', summary['minFU'], summary['TTE'])
    summary[DEFAULT_EVENT_INDICATOR_COLUMN] = np.select(
        condlist=[
            (summary['cens'] == 'interval'),
            (summary['cens'] == 'no')
        ],
        choicelist=[EventType.INTERVAL.value, EventType.EXACT.value],
        default=EventType.CENSORED.value
    )

    return summary
