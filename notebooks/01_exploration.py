"""Exploration du dataset OCDE-FAO de consommation de viande (OpenML 43758)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "meat_consumption.arff"
CSV = ROOT / "data" / "processed" / "meat_consumption.csv"
FIG = ROOT / "figures"
LAST_OBSERVED_YEAR = 2017  # au-delà : projections OCDE (dataset rafraîchi en 2018)
BASE_YEAR = 2000  # première année où toutes les zones ont les 4 viandes
# Soudan : KG_CAP ~1000x trop faible (incohérent avec THND_TONNE) -> exclu des analyses par habitant
BAD_KG_CAP = {"SDN"}

sns.set_theme(style="whitegrid")

# --- Chargement -------------------------------------------------------------
lines = RAW.read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().upper() == "@DATA") + 1
rows = [l.split(",") for l in lines[start:] if l.strip()]
df = pd.DataFrame(rows, columns=["location", "subject", "measure", "year", "value"])
df["year"] = df["year"].astype(int)
df["value"] = df["value"].astype(float)
df.to_csv(CSV, index=False)

print("=== Aperçu ===")
print(df.shape)
print(df.head(), "\n")
print("Valeurs manquantes :\n", df.isna().sum(), "\n")
print("Doublons :", df.duplicated(["location", "subject", "measure", "year"]).sum())
print("Années :", df.year.min(), "->", df.year.max())
print("Viandes :", sorted(df.subject.unique()))
print("Mesures :", sorted(df.measure.unique()))
print(f"{df.location.nunique()} zones :", sorted(df.location.unique()), "\n")
print("Valeurs négatives :", (df.value < 0).sum())
print(df.groupby(["measure", "subject"]).value.describe().round(1), "\n")

kg = df[(df.measure == "KG_CAP") & ~df.location.isin(BAD_KG_CAP)]
kg_wide = kg.pivot_table(index=["location", "year"], columns="subject", values="value")
# min_count=4 : pas de total si une viande manque (sinon total sous-estimé)
kg_wide["TOTAL"] = kg_wide.sum(axis=1, min_count=4)

# --- 1. Monde vs OCDE vs France : total par habitant -------------------------
zones = {"WLD": "Monde", "OECD": "OCDE", "EU28": "UE 28 (dont France)", "USA": "États-Unis", "CHN": "Chine", "IND": "Inde"}
fig, ax = plt.subplots(figsize=(10, 5))
for code, name in zones.items():
    if code in kg_wide.index.get_level_values(0):
        s = kg_wide.loc[code, "TOTAL"]
        ax.plot(s.index, s.values, label=name, lw=2)
ax.axvline(LAST_OBSERVED_YEAR + 0.5, color="grey", ls="--")
ax.text(LAST_OBSERVED_YEAR + 1, ax.get_ylim()[1] * 0.95, "projections", color="grey")
ax.set(title="Consommation totale de viande (kg/hab/an, poids au détail)", xlabel="", ylabel="kg / habitant")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "01_total_par_zone.png", dpi=120)

# --- 2. UE 28 (la France n'est pas isolée dans ce dataset) : par type de viande
fra = kg[kg.location == "EU28"]
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=fra, x="year", y="value", hue="subject", lw=2, ax=ax)
ax.axvline(LAST_OBSERVED_YEAR + 0.5, color="grey", ls="--")
ax.set(title="UE 28 : consommation par type de viande (kg/hab/an)", xlabel="", ylabel="kg / habitant")
fig.tight_layout(); fig.savefig(FIG / "02_ue28_par_viande.png", dpi=120)

# --- 3. Monde : évolution de la part de chaque viande ------------------------
wld = kg_wide.loc["WLD"].drop(columns="TOTAL").dropna()
share = wld.div(wld.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(share.index, share.T.values, labels=share.columns, alpha=0.85)
ax.axvline(LAST_OBSERVED_YEAR + 0.5, color="black", ls="--")
ax.set(title="Monde : part de chaque viande dans la consommation (%)", xlabel="", ylabel="%", ylim=(0, 100))
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(FIG / "03_monde_parts.png", dpi=120)

# --- 4. Classement des pays (dernière année observée) ------------------------
aggregates = {"WLD", "OECD", "EU28", "BRICS", "SSA", "NAFTA"}
last = kg_wide.xs(LAST_OBSERVED_YEAR, level="year")
last = last[~last.index.isin(aggregates)].sort_values("TOTAL", ascending=False)
print(f"=== Top / flop {LAST_OBSERVED_YEAR} (kg/hab) ===")
print(last["TOTAL"].round(1).head(10).to_string(), "\n...")
print(last["TOTAL"].round(1).tail(5).to_string(), "\n")
fig, ax = plt.subplots(figsize=(10, 9))
last.drop(columns="TOTAL").plot.barh(stacked=True, ax=ax, width=0.8)
ax.invert_yaxis()
ax.set(title=f"Consommation par habitant en {LAST_OBSERVED_YEAR} (kg/an)", xlabel="kg / habitant", ylabel="")
fig.tight_layout(); fig.savefig(FIG / "04_classement_pays.png", dpi=120)

# --- 5. Croissance BASE_YEAR -> 2017 ---------------------------------------------
tot = kg_wide["TOTAL"].unstack("location")
growth = ((tot.loc[LAST_OBSERVED_YEAR] / tot.loc[BASE_YEAR] - 1) * 100).drop(list(aggregates), errors="ignore")
print(f"=== Évolution du total {BASE_YEAR} -> {LAST_OBSERVED_YEAR} (%) ===")
print(growth.sort_values(ascending=False).round(0).to_string(), "\n")

# --- 6. Corrélation entre viandes (niveau pays) -----------------------------
print("=== Corrélation entre types de viande (pays, " + str(LAST_OBSERVED_YEAR) + ") ===")
print(last.drop(columns="TOTAL").corr().round(2))
print("\nFigures enregistrées dans", FIG)
