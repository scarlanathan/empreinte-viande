"""Prévision de la consommation de viande par habitant (kg/hab) par zone et par viande.

Seules les années observées (<= 2017) servent à l'apprentissage ; les projections
OCDE sont gardées à part, comme point de comparaison.

Modèles comparés par backtest (origines glissantes) :
- naive    : dernière valeur connue, constante
- linear   : tendance linéaire sur les 10 dernières années
- holt     : lissage exponentiel avec tendance amortie (Holt « damped »)
- gbm      : gradient boosting global, entraîné sur toutes les séries à la fois
- ensemble : moyenne de linear, holt et gbm

Toutes les séries sont modélisées en log(1 + kg), ce qui rend les tendances
multiplicatives et garantit des prévisions positives.

Lancer :  python forecast.py   (régénère data/processed/forecast*.csv)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import impact as im

HORIZON_END = 2030
BACKTEST_ORIGINS = [2005, 2007, 2009, 2011]  # on prédit les 6 années suivantes
BACKTEST_H = 6
N_LAGS = 5
MODELS = ["naive", "linear", "holt", "gbm", "ensemble"]
OUT_FC = im.ROOT / "data" / "processed" / "forecast.csv"
OUT_BT = im.ROOT / "data" / "processed" / "forecast_backtest.csv"


def observed_series() -> pd.DataFrame:
    """Matrice (location, meat) x année en log1p(kg/hab), années observées seulement."""
    cons = im.load_consumption()
    obs = cons[~cons.projection]
    wide = obs.pivot_table(index=["location", "meat"], columns="year", values="kg_cap")
    return np.log1p(wide)


# --- Modèles série par série -------------------------------------------------
def _naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def _linear(y: np.ndarray, h: int, window: int = 10) -> np.ndarray:
    tail = y[-window:]
    slope, intercept = np.polyfit(np.arange(len(tail)), tail, 1)
    return intercept + slope * np.arange(len(tail), len(tail) + h)


def _holt(y: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit()
    return fit.forecast(h)


# --- Modèle global (gradient boosting) -------------------------------------
def _features(lags: np.ndarray, meat: str) -> list[float]:
    """lags = N_LAGS dernières valeurs (log1p), de la plus ancienne à la plus récente."""
    diffs = np.diff(lags)
    return [lags[-1], *diffs, diffs.mean(), *(float(meat == m) for m in im.MEATS)]


def _train_gbm(panel: pd.DataFrame) -> HistGradientBoostingRegressor:
    X, y = [], []
    for (_, meat), row in panel.iterrows():
        s = row.dropna().to_numpy()
        for t in range(N_LAGS, len(s)):
            X.append(_features(s[t - N_LAGS:t], meat))
            y.append(s[t] - s[t - 1])  # on prédit la variation annuelle
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=30, l2_regularization=1.0,
                                          random_state=0)
    return model.fit(np.array(X), np.array(y))


def _gbm_forecast(model, y: np.ndarray, meat: str, h: int) -> np.ndarray:
    hist = list(y[-N_LAGS:])
    out = []
    for _ in range(h):  # prévision récursive : chaque prédiction devient un lag
        nxt = hist[-1] + model.predict([_features(np.array(hist[-N_LAGS:]), meat)])[0]
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def forecast_all(panel: pd.DataFrame, h: int) -> dict[str, pd.DataFrame]:
    """Prévoit h années après la dernière colonne du panel, pour chaque modèle."""
    gbm = _train_gbm(panel)
    last = panel.columns.max()
    years = range(last + 1, last + h + 1)
    res = {m: {} for m in MODELS}
    for key, row in panel.iterrows():
        y = row.dropna().to_numpy()
        res["naive"][key] = _naive(y, h)
        res["linear"][key] = _linear(y, h)
        res["holt"][key] = _holt(y, h)
        res["gbm"][key] = _gbm_forecast(gbm, y, key[1], h)
        res["ensemble"][key] = np.mean([res[m][key] for m in ("linear", "holt", "gbm")], axis=0)
    out = {}
    for m, v in res.items():
        df = pd.DataFrame(list(v.values()), columns=years,
                          index=pd.MultiIndex.from_tuples(list(v), names=["location", "meat"]))
        out[m] = df.clip(lower=0)
    return out


# --- Backtest ----------------------------------------------------------------
def backtest(panel: pd.DataFrame) -> pd.DataFrame:
    """Erreurs de prévision en kg/hab, pour chaque origine, modèle, série et horizon."""
    rows = []
    for origin in BACKTEST_ORIGINS:
        train = panel.loc[:, :origin]
        train = train[train.notna().sum(axis=1) >= 12]  # assez d'historique
        preds = forecast_all(train, BACKTEST_H)
        for model, fc in preds.items():
            for year in fc.columns:
                actual = np.expm1(panel.loc[fc.index, year])
                pred = np.expm1(fc[year])
                for (loc, meat), a, p in zip(fc.index, actual, pred):
                    rows.append({"origin": origin, "model": model, "location": loc, "meat": meat,
                                 "h": year - origin, "actual": a, "pred": p})
    bt = pd.DataFrame(rows).dropna()
    bt["error"] = bt.pred - bt.actual
    return bt


def score(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """MAE (kg/hab) et WAPE = somme |erreur| / somme réel (pondère les gros consommateurs)."""
    g = bt.assign(abs_err=bt.error.abs()).groupby(by)
    return pd.DataFrame({"mae": g.abs_err.mean(), "wape": g.abs_err.sum() / g.actual.sum()})


def interval_quantiles(bt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Quantiles 10 % / 90 % de l'erreur relative (en log1p) par horizon.

    Au-delà de l'horizon testé, l'incertitude est extrapolée en racine de h
    (comportement d'une marche aléatoire).
    """
    d = bt[bt.model == model]
    log_err = np.log1p(d.pred) - np.log1p(d.actual)
    q = log_err.groupby(d.h).quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]
    hmax = q.index.max()
    for h in range(hmax + 1, HORIZON_END - im.LAST_OBSERVED_YEAR + 1):
        q.loc[h] = q.loc[hmax] * np.sqrt(h / hmax)
    return q


def run() -> None:
    panel = observed_series()
    print(f"{len(panel)} séries, années {panel.columns.min()}-{panel.columns.max()}")

    bt = backtest(panel)
    bt.to_csv(OUT_BT, index=False)
    overall = score(bt, ["model"]).sort_values("wape")
    print("\n=== Backtest (toutes origines, horizons 1-6) ===")
    print(overall.round(3).to_string())
    print("\n=== WAPE par horizon ===")
    print(score(bt, ["model", "h"]).wape.unstack("h").round(3).to_string())
    best = overall.index[0]
    print(f"\nModèle retenu : {best}")

    h = HORIZON_END - im.LAST_OBSERVED_YEAR
    preds = forecast_all(panel, h)
    q = interval_quantiles(bt, best)
    frames = []
    for model, fc in preds.items():
        long = fc.stack().rename("log_pred").reset_index()
        long.columns = ["location", "meat", "year", "log_pred"]
        long["model"] = model
        frames.append(long)
    fc = pd.concat(frames)
    hz = fc.year - im.LAST_OBSERVED_YEAR
    # Les erreurs mesurées sont pred - réel : l'intervalle sur le réel est pred - erreur
    fc["kg_cap"] = np.expm1(fc.log_pred)
    fc["lo"] = np.expm1(fc.log_pred - hz.map(q.q90)).clip(lower=0)
    fc["hi"] = np.expm1(fc.log_pred - hz.map(q.q10))
    fc["best"] = fc.model == best
    fc.drop(columns="log_pred").to_csv(OUT_FC, index=False)
    print(f"Prévisions {im.LAST_OBSERVED_YEAR + 1}-{HORIZON_END} -> {OUT_FC.name}")


if __name__ == "__main__":
    run()
