"""Calcul de l'empreinte environnementale de la consommation de viande.

Croise la consommation OCDE-FAO (kg/hab, poids au détail) avec les coefficients
d'impact par kg de Poore & Nemecek (2018), moyennes mondiales 2010.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CONSUMPTION_CSV = ROOT / "data" / "processed" / "meat_consumption.csv"

MEATS = ["BEEF", "PIG", "POULTRY", "SHEEP"]
MEAT_LABELS = {"BEEF": "Bœuf", "PIG": "Porc", "POULTRY": "Volaille", "SHEEP": "Ovin"}
MEAT_COLORS = {"BEEF": "#B5473A", "PIG": "#E8A0A8", "POULTRY": "#E9B949", "SHEEP": "#6B8F71"}
# Poids carcasse -> poids au détail (doc OCDE), pour retrouver la population
CARCASS_TO_RETAIL = {"BEEF": 0.7, "PIG": 0.78, "POULTRY": 0.88, "SHEEP": 0.88}
LAST_OBSERVED_YEAR = 2017
# Soudan : KG_CAP ~1000x trop faible, incohérent avec les tonnages
EXCLUDED = {"SDN"}
AGGREGATES = {"WLD", "OECD", "EU28", "BRICS", "SSA", "NAFTA"}

ZONE_NAMES = {
    "WLD": "Monde", "OECD": "OCDE", "EU28": "Union européenne (28)", "BRICS": "BRICS",
    "SSA": "Afrique subsaharienne", "NAFTA": "ALENA",
    "ARG": "Argentine", "AUS": "Australie", "BGD": "Bangladesh", "BRA": "Brésil",
    "CAN": "Canada", "CHE": "Suisse", "CHL": "Chili", "CHN": "Chine", "COL": "Colombie",
    "DZA": "Algérie", "EGY": "Égypte", "ETH": "Éthiopie", "GHA": "Ghana", "HTI": "Haïti",
    "IDN": "Indonésie", "IND": "Inde", "IRN": "Iran", "ISR": "Israël", "JPN": "Japon",
    "KAZ": "Kazakhstan", "KOR": "Corée du Sud", "MEX": "Mexique", "MOZ": "Mozambique",
    "MYS": "Malaisie", "NGA": "Nigeria", "NOR": "Norvège", "NZL": "Nouvelle-Zélande",
    "PAK": "Pakistan", "PER": "Pérou", "PHL": "Philippines", "PRY": "Paraguay",
    "RUS": "Russie", "SAU": "Arabie saoudite", "THA": "Thaïlande", "TUR": "Turquie",
    "TZA": "Tanzanie", "UKR": "Ukraine", "URY": "Uruguay", "USA": "États-Unis",
    "VNM": "Vietnam", "ZAF": "Afrique du Sud", "ZMB": "Zambie",
}


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    icon: str
    file: str
    coef_unit: str      # unité du coefficient, par kg de viande
    cap_unit: str       # unité affichée par habitant et par an
    cap_factor: float   # coef_unit * kg -> cap_unit
    total_unit: str     # unité affichée pour un pays entier
    total_factor: float  # coef_unit * kg -> total_unit
    color_scale: str


INDICATORS = {
    "ghg": Indicator("ghg", "Gaz à effet de serre", "🌡️", "ghg-per-kg-poore.csv",
                     "kg CO₂e", "kg CO₂e", 1, "Mt CO₂e", 1e-9, "OrRd"),
    "water": Indicator("water", "Eau prélevée", "💧", "water-withdrawals-per-kg-poore.csv",
                       "L", "m³", 1e-3, "km³", 1e-12, "Blues"),
    "land": Indicator("land", "Surface utilisée", "🌾", "land-use-per-kg-poore.csv",
                      "m²", "m²", 1, "Mha", 1e-10, "YlGn"),
    "eutro": Indicator("eutro", "Eutrophisation", "🦠", "eutrophying-emissions-per-kg-poore.csv",
                       "g PO₄e", "kg PO₄e", 1e-3, "kt PO₄e", 1e-9, "Purples"),
}

_POORE_ENTITIES = {
    "Beef (beef herd)": "BEEF_HERD", "Beef (dairy herd)": "BEEF_DAIRY",
    "Pig Meat": "PIG", "Poultry Meat": "POULTRY", "Lamb & Mutton": "SHEEP",
}


def load_coefficients() -> pd.DataFrame:
    """Coefficients d'impact par kg : lignes = viandes Poore, colonnes = indicateurs."""
    cols = {}
    for ind in INDICATORS.values():
        raw = pd.read_csv(RAW / ind.file)
        raw = raw[raw.entity.isin(_POORE_ENTITIES)]
        cols[ind.key] = raw.set_index(raw.entity.map(_POORE_ENTITIES)).iloc[:, -1]
    return pd.DataFrame(cols)


def meat_coefficients(coefs: pd.DataFrame, beef_herd_share: float) -> pd.DataFrame:
    """Coefficients par viande OCDE ; le bœuf mélange troupeau allaitant et laitier."""
    beef = beef_herd_share * coefs.loc["BEEF_HERD"] + (1 - beef_herd_share) * coefs.loc["BEEF_DAIRY"]
    return pd.DataFrame({"BEEF": beef, "PIG": coefs.loc["PIG"],
                         "POULTRY": coefs.loc["POULTRY"], "SHEEP": coefs.loc["SHEEP"]}).T


def load_consumption() -> pd.DataFrame:
    """Une ligne par (zone, année, viande) : kg/hab et population (millions)."""
    df = pd.read_csv(CONSUMPTION_CSV)
    df = df[~df.location.isin(EXCLUDED)]
    wide = df.pivot_table(index=["location", "year", "subject"], columns="measure", values="value")
    wide = wide.dropna().reset_index()
    wide["pop"] = wide.THND_TONNE * 1e6 * wide.subject.map(CARCASS_TO_RETAIL) / wide.KG_CAP
    pop = wide.groupby(["location", "year"])["pop"].median()
    kg = wide.pivot_table(index=["location", "year"], columns="subject", values="KG_CAP")
    # On ne garde que les années où les 4 viandes sont connues (sinon totaux faussés)
    kg = kg[MEATS].dropna()
    out = kg.stack().rename("kg_cap").reset_index().rename(columns={"subject": "meat"})
    out["pop"] = out.set_index(["location", "year"]).index.map(pop)
    out["zone"] = out.location.map(ZONE_NAMES)
    out["is_aggregate"] = out.location.isin(AGGREGATES)
    out["projection"] = out.year > LAST_OBSERVED_YEAR
    return out


def compute_footprint(cons: pd.DataFrame, coefs: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Ajoute l'impact par habitant (cap) et total pour un indicateur."""
    ind = INDICATORS[indicator]
    out = cons.copy()
    out["coef"] = out.meat.map(coefs[indicator])
    raw = out.kg_cap * out.coef
    out["cap"] = raw * ind.cap_factor
    out["total"] = raw * out["pop"] * ind.total_factor
    return out


def _logmean(a: float, b: float) -> float:
    return a if np.isclose(a, b) else (a - b) / (np.log(a) - np.log(b))


def decompose(fp: pd.DataFrame, location: str, y0: int, y1: int, per_capita: bool) -> dict:
    """Décomposition LMDI de la variation d'impact entre y0 et y1.

    Impact = Population × Quantité/hab × Intensité moyenne du mix
    L'effet « mix » isole la substitution entre viandes (coefficients fixes).
    """
    z = fp[fp.location == location].set_index(["year", "meat"])
    def parts(y):
        kg = z.loc[y, "kg_cap"]
        q = kg.sum()
        intensity = (kg * z.loc[y, "coef"]).sum() / q
        pop = z.loc[y, "pop"].iloc[0]
        value = z.loc[y, "cap" if per_capita else "total"].sum()
        return pop, q, intensity, value
    p0, q0, i0, v0 = parts(y0)
    p1, q1, i1, v1 = parts(y1)
    lm = _logmean(v1, v0)
    effects = {
        "Population": 0.0 if per_capita else lm * np.log(p1 / p0),
        "Quantité consommée / hab": lm * np.log(q1 / q0),
        "Mix des viandes": lm * np.log(i1 / i0),
    }
    if per_capita:
        effects.pop("Population")
    return {"start": v0, "end": v1, "effects": effects}


def counterfactual(fp: pd.DataFrame, location: str, base_year: int, per_capita: bool) -> pd.DataFrame:
    """Série réelle vs série si le mix de viandes était resté celui de base_year."""
    z = fp[(fp.location == location) & (fp.year >= base_year)]
    col = "cap" if per_capita else "total"
    real = z.groupby("year")[col].sum()
    base = z[z.year == base_year].set_index("meat")
    base_intensity = (base.kg_cap * base.coef).sum() / base.kg_cap.sum()
    g = z.groupby("year")
    q = g.kg_cap.sum()
    intensity = g.apply(lambda d: (d.kg_cap * d.coef).sum()) / q
    frozen = real * base_intensity / intensity
    return pd.DataFrame({"real": real, "frozen_mix": frozen, "avoided": frozen - real})
