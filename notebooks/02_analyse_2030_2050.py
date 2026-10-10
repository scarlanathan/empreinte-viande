"""Conséquences environnementales de la consommation mondiale de viande : 2024, 2030, 2035, 2050.

Chaque chiffre a un statut explicite :
- OBSERVÉ   : OECD-FAO Agricultural Outlook 2026-2035 (période de référence 2023-2025)
- PROJETÉ   : projections officielles OCDE-FAO 2026-2035
- HYPOTHÈSE : nos extrapolations au-delà de 2035 (2050), clairement étiquetées

Sources téléchargées dans data/raw/ :
- oecd_fao_outlook_2026_2035_meat.csv : API SDMX OCDE, DSD_AGR@DF_OUTLOOK_2026_2035,
  mesure FO_PC (consommation humaine, kg/personne, poids au détail)
- owid_population_un.csv : ONU World Population Prospects 2024 (via Our World in Data)
- *-poore.csv : Poore & Nemecek (2018), via Our World in Data
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import impact as im  # noqa: E402

RAW = im.RAW
FIG = im.ROOT / "figures"
MEAT_CODES = {"CPC_EX_BV": "BEEF", "CPC_EX_PK": "PIG", "CPC_EX_PT": "POULTRY", "CPC_EX_SH": "SHEEP"}
YEARS = [2017, 2024, 2030, 2035, 2050]

# Repères vérifiés
GLOBAL_GHG_2023 = 57.1     # Gt CO2e — UNEP Emissions Gap Report 2024
FAO_LIVESTOCK_2015 = 6.2   # Gt CO2e, ensemble de l'élevage, 2015 — FAO (2023) Pathways towards lower emissions
FRANCE_MHA = 54.394        # France métropolitaine, 543 940 km² — INSEE
EAT_LANCET_G_DAY = {"red": 14, "POULTRY": 29}  # bœuf+agneau+porc / volaille — EAT-Lancet (2019, inchangé en 2025)

# --- Données ---------------------------------------------------------------
oecd = pd.read_csv(RAW / "oecd_fao_outlook_2026_2035_meat.csv")
oecd = oecd[(oecd.REF_AREA == "W") & (oecd.MEASURE == "FO_PC") & (oecd.TIME_PERIOD >= 1991)]
kg = (oecd.assign(meat=oecd.COMMODITY.map(MEAT_CODES))
      .pivot_table(index="TIME_PERIOD", columns="meat", values="OBS_VALUE")[im.MEATS])

pop_raw = pd.read_csv(RAW / "owid_population_un.csv")
w = pop_raw[pop_raw.entity == "World"].set_index("year")
pop = w.iloc[:, -2].fillna(w.iloc[:, -1])  # estimations jusqu'en 2023, projection médiane ensuite

coefs_raw = im.load_coefficients()


def status(year: int) -> str:
    return "observé" if year <= 2024 else "projeté OCDE-FAO" if year <= 2035 else "HYPOTHÈSE"


# --- 2050 : deux hypothèses explicites --------------------------------------
# H1 : consommation par habitant figée au niveau projeté pour 2035 (seule la population change)
# H2 : la tendance linéaire OCDE-FAO 2025-2035 de chaque viande se poursuit jusqu'en 2050
kg_2050_h1 = kg.loc[2035]
slope = (kg.loc[2035] - kg.loc[2025]) / 10
kg_2050_h2 = kg.loc[2035] + slope * 15

red_mix_2024 = kg.loc[2024, ["BEEF", "PIG", "SHEEP"]] / kg.loc[2024, ["BEEF", "PIG", "SHEEP"]].sum()
eat = pd.concat([red_mix_2024 * EAT_LANCET_G_DAY["red"] * 365 / 1000,
                 pd.Series({"POULTRY": EAT_LANCET_G_DAY["POULTRY"] * 365 / 1000})])[im.MEATS]

cases = [(y, "Tendance", kg.loc[y]) for y in YEARS if y <= 2035]
cases += [(2050, "H1 : conso/hab figée au niveau 2035", kg_2050_h1),
          (2050, "H2 : tendance 2025-2035 prolongée", kg_2050_h2),
          (2035, "Régime EAT-Lancet", eat), (2050, "Régime EAT-Lancet", eat)]


def impacts(k: pd.Series, year: int, beef_herd_share: float) -> dict:
    coefs = im.meat_coefficients(coefs_raw, beef_herd_share)
    out = {}
    for key, ind in im.INDICATORS.items():
        raw = (k * coefs[key]).sum()
        out[f"{key}_cap"] = raw * ind.cap_factor
        out[f"{key}_total"] = raw * pop[year] * ind.total_factor
    return out


rows = []
for year, name, k in cases:
    # Un régime alternatif est toujours une hypothèse ; sinon le statut dépend de l'année
    statut = "HYPOTHÈSE" if "EAT" in name else status(year)
    r = {"année": year, "statut": statut,
         "scénario": name, "pop_Md": pop[year] / 1e9, "kg_hab": k.sum(), **k.round(2).to_dict()}
    r.update(impacts(k, year, 2 / 3))
    # Sensibilité : tout le bœuf issu de troupeaux laitiers (0) vs allaitants (1)
    r["ghg_Gt_min"] = impacts(k, year, 0)["ghg_total"] / 1000
    r["ghg_Gt_max"] = impacts(k, year, 1)["ghg_total"] / 1000
    rows.append(r)
res = pd.DataFrame(rows)
res["ghg_Gt"] = res.ghg_total / 1000
res["pct_GES_mondiaux_2023"] = res.ghg_Gt / GLOBAL_GHG_2023 * 100
res["surface_x_France"] = res.land_total / FRANCE_MHA

pd.set_option("display.width", 250)
print("=== Monde ===")
print(res[["année", "statut", "scénario", "pop_Md", "kg_hab", "BEEF", "PIG", "POULTRY", "SHEEP",
           "ghg_cap", "ghg_Gt", "ghg_Gt_min", "ghg_Gt_max", "pct_GES_mondiaux_2023",
           "land_total", "surface_x_France", "water_total", "eutro_total"]].round(2).to_string(index=False))

# --- Décomposition : population vs consommation par habitant ---------------
base = res[(res.année == 2024) & (res.scénario == "Tendance")].iloc[0]
for _, r in res[(res.année > 2024) & ~res.scénario.str.contains("EAT")].iterrows():
    pop_effect = base.ghg_Gt * (r.pop_Md / base.pop_Md - 1)
    print(f"{r.année} {r.scénario[:30]:30s} : {r.ghg_Gt - base.ghg_Gt:+.2f} Gt vs 2024 "
          f"(population {pop_effect:+.2f}, conso/mix par hab {r.ghg_Gt - base.ghg_Gt - pop_effect:+.2f})")

# --- Figure ------------------------------------------------------------------
colors = {"Tendance": "#B5473A", "H1 : conso/hab figée au niveau 2035": "#D98C6F",
          "H2 : tendance 2025-2035 prolongée": "#8E2F25", "Régime EAT-Lancet": "#6B8F71"}
fig, ax = plt.subplots(figsize=(11, 5.5))
x, ticks = 0, []
for year in YEARS:
    sub = res[res.année == year]
    start = x
    for _, r in sub.iterrows():
        hatch = "//" if r.statut == "HYPOTHÈSE" else None
        ax.bar(x, r.ghg_Gt, color=colors[r.scénario], hatch=hatch, edgecolor="white",
               label=r.scénario if r.scénario not in ax.get_legend_handles_labels()[1] else None)
        ax.vlines(x, r.ghg_Gt_min, r.ghg_Gt_max, color="black", lw=1)
        ax.text(x + 0.08, r.ghg_Gt + 0.12, f"{r.ghg_Gt:.1f}".replace(".", ","), fontsize=9)
        x += 1
    ticks.append(((start + x - 1) / 2, f"{year}\n({status(year)})"))
    x += 0.7
ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
ax.set_ylabel("Gt CO₂e / an")
ax.set_title("Émissions liées à la consommation mondiale de viande (bœuf, porc, volaille, ovin)")
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.text(0.99, 0.01, "Barres hachurées = hypothèses · trait noir = sensibilité au type de troupeau bovin",
         ha="right", fontsize=8, color="grey")
fig.savefig(FIG / "05_scenarios_2030_2050.png", dpi=130)
res.to_csv(im.ROOT / "data" / "processed" / "analyse_2030_2050.csv", index=False)
print("\nFigure ->", FIG / "05_scenarios_2030_2050.png")
