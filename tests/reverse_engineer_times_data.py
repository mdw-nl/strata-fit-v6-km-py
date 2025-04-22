import pandas as pd
import os
from pathlib import Path

def reverse_engineer_strata_fit(suv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse-engineer a raw STRATA-FIT-style longitudinal dataset
    from interval survival data: suv_df has columns ['interval_start','interval_end','event_type'].
    Fills only the fields required by the PatientData schema (month_diagnosis dropped).
    """
    rows = []
    # Dummy static values for optional covariates
    dummy_values = {
        "Age_diagnosis": 50,
        "Sex": 1,
        "RF_positivity": 0,
        "anti_CCP": 0,
        "CRP": 1.0,
        "ESR": 10,
        "SJC28": 2,
        "TJC28": 3,
        "csDMARD1": 0,
        "csDMARD2": 0,
        "csDMARD3": 0,
        "conc_MTX_dose": 15.0,
        "N_prev_csDMARD": 0,
        "bDMARD": 0,
        "N_prev_bDMARD": 0,
        "tsDMARD": 0,
        "N_prev_tsDMARD": 0,
        "GC": 0,
        "GC_type": 1,
        "GC_dose": 5.0,
        "eq5d": 0.8,
        "HAQ": 1.0,
        "Year_diagnosis": 2015,
        "Symptom_duration": 12.0
    }
    
    for idx, row in suv_df.iterrows():
        pid = f"SE{idx+1}"  # match pattern: two letters plus number
        start = row['interval_start']
        end   = row['interval_end']
        etype = row['event_type']
        
        if etype == "interval":
            # Pre-event visit
            rec = {
                "pat_ID": pid,
                "Visit_months_from_diagnosis": round(start, 2),
                "D2T_RA": False
            }
            rec.update(dummy_values)
            rows.append(rec)
            # Post-event visit
            rec2 = {
                "pat_ID": pid,
                "Visit_months_from_diagnosis": round(end, 2),
                "D2T_RA": True
            }
            rec2.update({**dummy_values, **{"bDMARD":1, "tsDMARD":1, "csDMARD1":1, "DAS28":4.0, "Pat_global":60.0, "Ph_global":60.0}})
            rows.append(rec2)
        else:
            is_event = (etype == "exact")
            rec = {
                "pat_ID": pid,
                "Visit_months_from_diagnosis": round(end, 2),
                "D2T_RA": is_event
            }
            # adjust dummy for event vs censor
            if is_event:
                rec.update({**dummy_values, **{"bDMARD":1, "tsDMARD":1, "csDMARD1":1, "DAS28":4.0, "Pat_global":60.0, "Ph_global":60.0}})
            else:
                rec.update(dummy_values)
            rows.append(rec)

    return pd.DataFrame(rows)

# Example usage:
input_times = Path("tests/data/data_times/Synthetic_interval_survival_data.csv")
output_data = input_times.parent / "alpha.csv"
suv = pd.read_csv(input_times)
raw = reverse_engineer_strata_fit(suv)
raw.to_csv(output_data)
print(raw.head(10))
