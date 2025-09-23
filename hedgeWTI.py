
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model

# --------- 1) Charger et aligner ----------
path = "./RWTCd.xls"  # <-- change le nom de fichier si besoin
spot = pd.read_excel(path, sheet_name="Spot")
fut  = pd.read_excel(path, sheet_name="CLc2")

# Normaliser les noms
spot = spot.rename(columns={col: col.strip() for col in spot.columns})
fut  = fut.rename(columns={col: col.strip() for col in fut.columns})

# Assurer le type date et garder les colonnes utiles
spot["Date"] = pd.to_datetime(spot["Date"])
fut["Date"]  = pd.to_datetime(fut["Date"])
spot = spot[["Date", "r_spot"]].dropna()
fut  = fut[["Date", "r_fut"]].dropna()

# Merge sur dates communes
df = pd.merge(spot, fut, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
dh=df.copy()  
# Option : enlever les jours de retour 0 absolu (si tu en as créés par erreur)
# df = df[(df["r_spot"] != 0) | (df["r_fut"] != 0)]

# --------- 2) Fonction d'estimation OLS ----------
def estimate_mvhr_ols(data, start=None, end=None, hac_lags=5):
    """
    data: DataFrame avec colonnes r_spot, r_fut, Date
    start/end: limites temporelles (str 'YYYY-MM-DD' ou Timestamp) pour un régime
    hac_lags: lags Newey-West pour erreurs robustes (autocorrélation/hétéro)
    Retourne: dict avec beta (h*), alpha, t-stats, R2, n
    """
    d = data.copy()
    if start is not None:
        d = d[d["Date"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["Date"] <= pd.to_datetime(end)]
    d = d.dropna(subset=["r_spot", "r_fut"])

    # X = r_fut, y = r_spot
    X = sm.add_constant(d["r_fut"].values)   # ajoute l'intercept alpha
    y = d["r_spot"].values

    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})  # Newey-West

    alpha = res.params[0]
    beta  = res.params[1]                 # <-- h* (hedge ratio)
    t_a   = res.tvalues[0]
    t_b   = res.tvalues[1]
    r2    = res.rsquared
    n     = int(res.nobs)

    return {
        "alpha": alpha,
        "beta_hstar": beta,
        "t_alpha": t_a,
        "t_beta": t_b,
        "R2": r2,
        "n": n
    }

# --------- 3) Exemple d’appels ----------
# a) Un seul hedge ratio sur toute la période disponible
res_all = estimate_mvhr_ols(df)
print("Global  :", res_all)


res_pre  = estimate_mvhr_ols(df, start="2020-01-01", end="2020-03-05")
res_cris = estimate_mvhr_ols(df, start="2020-03-06", end="2020-04-20")
res_post = estimate_mvhr_ols(df, start="2020-04-21", end="2020-06-30")
print("Pré-crise :", res_pre)
print("Crise     :", res_cris)
print("Post-crise:", res_post)

# --------- 4) (Option) Rolling OLS pour voir la dérive de beta (si tu veux) ----------
def rolling_beta(data, window=60, hac_lags=0):
    """
    Calcule un beta OLS roulant (sans HAC par défaut pour la vitesse).
    Retourne un DataFrame avec Date (fin de fenêtre) et beta.
    """
    out = []
    d = data.dropna(subset=["r_spot", "r_fut"]).reset_index(drop=True)
    for i in range(window, len(d)):
        sub = d.iloc[i-window:i]
        X = sm.add_constant(sub["r_fut"].values)
        y = sub["r_spot"].values
        res = sm.OLS(y, X).fit() if hac_lags==0 else sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        out.append({"Date": d.loc[i, "Date"], "beta_hstar": res.params[1]})
    return pd.DataFrame(out)

# Exemple rolling (facultatif)
beta_roll = rolling_beta(df, window=60)
print(beta_roll.tail())




def subset(df, start=None, end=None):
    d = df.copy()
    if start is not None:
        d = d[d["Date"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["Date"] <= pd.to_datetime(end)]
    return d.dropna(subset=["r_spot", "r_fut"]).reset_index(drop=True)

def hedged_returns(d, h):
    """ r_H = r_spot - h * r_fut """
    out = d.copy()
    out["r_H"] = out["r_spot"] - h * out["r_fut"]
    return out

def hedge_effectiveness(d):
    """ HE = 1 - Var(r_H)/Var(r_spot) (variance empirique, ddof=1) """
    var_spot = np.var(d["r_spot"], ddof=1)
    var_hedg = np.var(d["r_H"],   ddof=1)
    return 1.0 - (var_hedg / var_spot)

# --- Définis clairement les fenêtres des trois régimes ---
pre_start, pre_end   = "2020-01-01", "2020-03-05"
cris_start, cris_end = "2020-03-06", "2020-04-20"
post_start, post_end = "2020-04-21", "2020-06-30"


h_pre, h_cris, h_post = res_pre["beta_hstar"], res_cris["beta_hstar"], res_post["beta_hstar"]

# --- 2) Construire r_H et calculer HE dans chaque régime (avec h* du même régime) ---
def eval_regime_HE(df, start, end, h, label):
    d = subset(df, start, end)
    dH = hedged_returns(d, h)
    HE = hedge_effectiveness(dH)
    return {"regime": label, "start": start, "end": end, "n": len(dH), "h_used": h, "HE": HE}

res_HE_pre  = eval_regime_HE(df, pre_start,  pre_end,  h_pre,  "Pré-crise (OLS du régime)")
res_HE_cris = eval_regime_HE(df, cris_start, cris_end, h_cris, "Crise (OLS du régime)")
res_HE_post = eval_regime_HE(df, post_start, post_end, h_post, "Post-crise (OLS du régime)")

# --- 3) Tableau récap propre (uniquement les 3 HE demandés) ---
table_HE = pd.DataFrame([res_HE_pre, res_HE_cris, res_HE_post])[
    ["regime", "start", "end", "n", "h_used", "HE"]
].reset_index(drop=True)

print("\n=== Hedge Effectiveness (HE) — OLS par régime ===")
print(table_HE.to_string(index=False))


# --- 1-to-1 Hedge Ratio Evaluation ---
# This assumes a hedge ratio of 1 (h = 1) and evaluates its effectiveness by regime.

def eval_1to1_hedge(df, start, end, label):
    """
    Evaluate the effectiveness of a 1-to-1 hedge ratio (h = 1) for a specific regime.
    Args:
        df: DataFrame with columns 'Date', 'r_spot', 'r_fut'.
        start: Start date of the regime (string or Timestamp).
        end: End date of the regime (string or Timestamp).
        label: Label for the regime (string).
    Returns:
        dict: Results including regime label, start/end dates, number of observations, hedge ratio used, and HE.
    """
    # Subset the data for the given regime
    d = subset(df, start, end)
    
    # Apply the 1-to-1 hedge ratio (h = 1)
    dH = hedged_returns(d, h=1)
    
    # Calculate hedge effectiveness (HE)
    HE = hedge_effectiveness(dH)
    
    # Return results as a dictionary
    return {"regime": label, "start": start, "end": end, "n": len(dH), "h_used": 1, "HE": HE}

# --- Evaluate 1-to-1 Hedge Effectiveness for Each Regime ---
res_1to1_pre  = eval_1to1_hedge(df, pre_start,  pre_end,  "Pré-crise (1-to-1)")
res_1to1_cris = eval_1to1_hedge(df, cris_start, cris_end, "Crise (1-to-1)")
res_1to1_post = eval_1to1_hedge(df, post_start, post_end, "Post-crise (1-to-1)")

# --- Combine Results into a Summary Table ---
table_1to1_HE = pd.DataFrame([res_1to1_pre, res_1to1_cris, res_1to1_post])[
    ["regime", "start", "end", "n", "h_used", "HE"]
].reset_index(drop=True)

# Print the results for the 1-to-1 hedge ratio
print("\n=== Hedge Effectiveness (HE) — 1-to-1 Hedge Ratio ===")
print(table_1to1_HE.to_string(index=False))

from arch import arch_model
import pandas as pd

def estimate_garch_hedge_ratio_direct(df):
    """
    Estimate the hedge ratio directly using a GARCH(1,1) model with ARX mean specification.
    
    Args:
        df: DataFrame with columns 'r_spot' and 'r_fut'.
        
    Returns:
        float: Hedge ratio estimated within the GARCH(1,1) model.
    """
    # Remove any missing values and reset index
    df = df.dropna(subset=["r_spot", "r_fut"]).reset_index(drop=True)
    
    # Scale data by 1000 for improved numerical stability during estimation
    scale_factor = 1000
    r_spot_scaled = df["r_spot"] * scale_factor
    r_fut_scaled = df["r_fut"] * scale_factor
    
    # Prepare exogenous variable (futures returns) as 2D array for arch_model
    exog_data = r_fut_scaled.values.reshape(-1, 1)
    
    # Specify GARCH(1,1) model with ARX mean equation: r_spot = alpha + beta*r_fut + error
    # ARX allows inclusion of exogenous variables in the mean equation
    model = arch_model(r_spot_scaled, x=exog_data, vol="Garch", p=1, q=1, mean="ARX", lags=0)
    
    # Estimate model parameters using maximum likelihood
    garch_res = model.fit(disp="off")
    
    
    # Extract hedge ratio (beta coefficient of futures returns)
    # Parameter order: [0] = constant, [1] = beta (hedge ratio), [2+] = GARCH parameters
    hedge_ratio = garch_res.params.iloc[1]
    
    return hedge_ratio

def eval_garch_hedge(df, start, end, label):
    """
    Evaluate hedge effectiveness using GARCH(1,1) estimated hedge ratio for a specific time period.
    
    Args:
        df: DataFrame with columns 'Date', 'r_spot', 'r_fut'.
        start: Start date of the regime (string or Timestamp).
        end: End date of the regime (string or Timestamp).
        label: Label for the regime (string).
        
    Returns:
        dict: Results including regime label, dates, sample size, hedge ratio, and effectiveness.
    """
    # Extract data subset for the specified time period
    d = subset(df, start, end)
    
    # Estimate optimal hedge ratio using GARCH(1,1) model
    h_garch = estimate_garch_hedge_ratio_direct(d)
    
    # Construct hedged portfolio returns using estimated hedge ratio
    dH = hedged_returns(d, h_garch)
    
    # Calculate hedge effectiveness metric
    HE = hedge_effectiveness(dH)
    
    # Return comprehensive results as dictionary
    return {"regime": label, "start": start, "end": end, "n": len(dH), "h_used": h_garch, "HE": HE}

# === MAIN ANALYSIS: Evaluate GARCH hedge effectiveness across different market regimes ===

# Estimate hedge effectiveness for pre-crisis period
res_garch_pre = eval_garch_hedge(df, pre_start, pre_end, "Pré-crise (GARCH)")

# Estimate hedge effectiveness for crisis period
res_garch_cris = eval_garch_hedge(df, cris_start, cris_end, "Crise (GARCH)")

# Estimate hedge effectiveness for post-crisis period
res_garch_post = eval_garch_hedge(df, post_start, post_end, "Post-crise (GARCH)")

# Consolidate results into summary table
table_garch_HE = pd.DataFrame([res_garch_pre, res_garch_cris, res_garch_post])[
    ["regime", "start", "end", "n", "h_used", "HE"]
].reset_index(drop=True)

# Display final results
print("\n=== Hedge Effectiveness (HE) — GARCH(1,1) with ARX Mean Specification ===")
print(table_garch_HE.to_string(index=False))