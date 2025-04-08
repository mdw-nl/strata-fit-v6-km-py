import pandas as pd
from typing import List, Dict


def compute_event_row(group):
    group = group.copy()
    group['event_time'] = group.loc[group['D2T_RA'], 'Visit_months_from_diagnosis'].min()
    group['censored_time'] = group['Visit_months_from_diagnosis'].max()

    group['time'] = group['Visit_months_from_diagnosis'].min()

    if pd.notna(group['event_time'].iloc[0]):
        group['time2'] = group['event_time'].iloc[0]
        group['status'] = 1
    else:
        group['time2'] = group['censored_time'].iloc[0]
        group['status'] = 0  # right censored

    return group.iloc[[0]]  # take one row per patient


def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    # Sort data
    df.sort_values(['pat_ID', 'Visit_months_from_diagnosis'], inplace=True)

    # Step 1: Cumulative treatment counts
    df['cum_bDMARD'] = df.groupby('pat_ID')['bDMARD'].cumsum().fillna(0)
    df['cum_tsDMARD'] = df.groupby('pat_ID')['tsDMARD'].cumsum().fillna(0)
    df['cum_btsDMARD'] = df['cum_bDMARD'] + df['cum_tsDMARD']
    df['cum_btsDMARDmin'] = df.groupby('pat_ID')['cum_btsDMARD'].cummin()

    # Step 2: Criteria
    df['D2T_crit1'] = (df['cum_btsDMARD'] > 1)

    # Optional improvement: rolling average DAS28
    df['rolling_avg_DAS28'] = df.groupby('pat_ID')['DAS28'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    df['D2T_crit2'] = (df['DAS28'] > 3.2) | (df['rolling_avg_DAS28'] > 3.2)

    df['D2T_crit3'] = (df['Pat_global'] > 50) | (df['Ph_global'] > 50)

    # Final D2T flag
    df['D2T_RA'] = df['D2T_crit1'] & df['D2T_crit2'] & df['D2T_crit3']

    # Step 3: Per patient summary
    summary = df.groupby('pat_ID').agg(
        Year_diagnosis=('Year_diagnosis', 'first'),
        D2T_RA_Ever=('D2T_RA', 'max'),
        cum_btsDMARDmin=('cum_btsDMARDmin', 'max'),
        minFU=('Visit_months_from_diagnosis', 'min'),
        TTE=('Visit_months_from_diagnosis', lambda x: x[df.loc[x.index, 'D2T_RA']].min() if any(df.loc[x.index, 'D2T_RA']) else np.nan),
        maxFU=('Visit_months_from_diagnosis', 'max')
    ).reset_index()

    # Clean up
    summary['D2T_RA_Ever'] = summary['D2T_RA_Ever'].fillna(0)
    summary['cum_btsDMARDmin'] = summary['cum_btsDMARDmin'].fillna(0)
    summary['TTE'] = summary['TTE'].fillna(summary['maxFU'])

    # Step 4: Censoring type
    summary['cens'] = np.select(
        condlist=[
            (summary['D2T_RA_Ever'] == 1) & (summary['cum_btsDMARDmin'] > 2),
            (summary['D2T_RA_Ever'] == 0)
        ],
        choicelist=['interval', 'right'],
        default='no'
    )

    # Step 5: Time, time2, status
    summary['time'] = np.where(summary['cens'] == 'interval', 0, summary['TTE'])
    summary['time2'] = np.where(summary['cens'] == 'interval', summary['minFU'], summary['TTE'])
    summary['status'] = np.select(
        condlist=[
            (summary['cens'] == 'interval'),
            (summary['cens'] == 'no')
        ],
        choicelist=[3, 1],
        default=0
    )

    return summary
