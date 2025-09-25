import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_mvhr_ols(data, start=None, end=None, hac_lags=5):
    """
    data: DataFrame avec r_spot, r_fut et soit:
          - une colonne 'Date' (datetime), soit
          - un index de type DatetimeIndex.
    """
    d = data.copy()

    # --- Harmoniser l'accès aux dates ---
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"])
        if start is not None:
            d = d[d["Date"] >= pd.to_datetime(start)]
        if end is not None:
            d = d[d["Date"] <= pd.to_datetime(end)]
    else:
        if not isinstance(d.index, pd.DatetimeIndex):
            raise KeyError("Aucune colonne 'Date' et l'index n'est pas DatetimeIndex.")
        # Filtrage par l'index datetime
        if start is not None:
            d = d.loc[pd.to_datetime(start):]
        if end is not None:
            d = d.loc[:pd.to_datetime(end)]

    d = d.dropna(subset=["r_spot", "r_fut"])

    # --- OLS: y = r_spot, X = [const, r_fut] ---
    X = sm.add_constant(d["r_fut"].values)
    y = d["r_spot"].values

    res = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    return {
        "alpha":      float(res.params[0]),
        "beta_hstar": float(res.params[1]),
        "t_alpha":    float(res.tvalues[0]),
        "t_beta":     float(res.tvalues[1]),
        "R2":         float(res.rsquared),
        "n":          int(res.nobs),
    }


# --- Récap OLS avec fenêtres intégrées (pré/crise/post) ---
def ols_summary_table_fixed(
    data,
    hac_lags=5,
    digits=6,
    pre=("2021-11-01", "2022-02-23"),
    crisis=("2022-02-24", "2022-03-15"),
    post=("2022-03-16", "2022-08-31"),
    include_full=True,
):
    windows = {
        "Pre-crisis":  pre,
        "Crisis":      crisis,
        "Post-crisis": post,
    }

    rows = []
    if include_full:
        res = estimate_mvhr_ols(data, hac_lags=hac_lags)
        rows.append({
            "regime": "Full sample", "start": None, "end": None, "n": res["n"],
            "alpha": res["alpha"], "t_alpha": res["t_alpha"],
            "beta":  res["beta_hstar"], "t_beta": res["t_beta"], "R2": res["R2"]
        })

    for label, (start, end) in windows.items():
        res = estimate_mvhr_ols(data, start, end, hac_lags=hac_lags)
        rows.append({
            "regime": label, "start": start, "end": end, "n": res["n"],
            "alpha": res["alpha"], "t_alpha": res["t_alpha"],
            "beta":  res["beta_hstar"], "t_beta": res["t_beta"], "R2": res["R2"]
        })

    out = pd.DataFrame(rows, columns=["regime","start","end","n","alpha","t_alpha","beta","t_beta","R2"])
    for c in ["alpha","t_alpha","beta","t_beta","R2"]:
        out[c] = out[c].round(digits)
    return out


# dh doit contenir Date (colonne ou index), r_spot, r_fut
print(ols_summary_table_fixed(dh))