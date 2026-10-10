"""Empreinte environnementale de la consommation de viande — app Streamlit.

Lancer :  python -m streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import forecast as fc_module
import impact as im

st.set_page_config(page_title="Empreinte de la viande", page_icon="🥩", layout="wide")

# Équivalents concrets (ordres de grandeur), exprimés dans l'unité du coefficient
EQUIVALENTS = {
    "ghg": (0.2, "km en voiture thermique", "≈ 0,2 kg CO₂e/km, voiture moyenne"),
    "water": (60, "douches de 60 L", "1 douche ≈ 60 L"),
    "land": (260, "courts de tennis", "1 court ≈ 260 m²"),
}
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def fmt(x: float, decimals: int = 0) -> str:
    """Format français : espace fine pour les milliers, virgule décimale."""
    s = f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return s


def pct(x: float) -> str:
    """Pourcentage signé à une décimale, au format français."""
    return f"{x:+.1f}".replace(".", ",") + " %"


def fmt_auto(x: float) -> str:
    return fmt(x, 0 if abs(x) >= 100 else 1 if abs(x) >= 10 else 2)


def shade_projections(fig: go.Figure, x_max: int) -> None:
    if x_max > im.LAST_OBSERVED_YEAR:
        fig.add_vrect(x0=im.LAST_OBSERVED_YEAR + 0.5, x1=x_max + 0.5, fillcolor="grey",
                      opacity=0.08, line_width=0, annotation_text="Projections OCDE",
                      annotation_position="top left", annotation_font_color="grey")


# --- Données ---------------------------------------------------------------
@st.cache_data
def load():
    return im.load_consumption(), im.load_coefficients()


@st.cache_data
def footprint(indicator: str, beef_herd_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cons, raw_coefs = load()
    coefs = im.meat_coefficients(raw_coefs, beef_herd_share)
    return im.compute_footprint(cons, coefs, indicator), coefs


# --- Barre latérale ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    ind_key = st.radio(
        "Indicateur environnemental",
        list(im.INDICATORS),
        format_func=lambda k: f"{im.INDICATORS[k].icon} {im.INDICATORS[k].label}",
    )
    beef_share = st.slider(
        "Part du bœuf issue de troupeaux à viande", 0, 100, 67, step=5, format="%d %%",
        help="Le bœuf issu de vaches laitières réformées a un impact ~3× plus faible "
             "en CO₂e (l'impact est partagé avec le lait). Les données OCDE ne "
             "distinguent pas les deux : ce curseur fixe l'hypothèse.",
    ) / 100
    show_proj = st.toggle("Inclure les projections OCDE (2018-2026)", value=True)
    st.divider()
    st.caption(
        "**Sources** : consommation OCDE-FAO (OpenML 43758) · coefficients d'impact "
        "Poore & Nemecek (2018), *Science*, via Our World in Data."
    )

ind = im.INDICATORS[ind_key]
fp, coefs = footprint(ind_key, beef_share)
if not show_proj:
    fp = fp[~fp.projection]
year_max = int(fp.year.max())
countries = fp[~fp.is_aggregate]
zone_options = (
    [z for z in ["Monde", "Union européenne (28)", "OCDE", "BRICS", "Afrique subsaharienne"]
     if z in set(fp.zone)]
    + sorted(countries.zone.unique())
)
code_of = {v: k for k, v in im.ZONE_NAMES.items()}

st.title("🥩 Empreinte environnementale de la viande")
st.markdown(
    f"Combien pèse notre consommation de viande en **{ind.icon} {ind.label.lower()}** ? "
    "Consommation réelle par pays × impact moyen de chaque viande."
)

tab_map, tab_subst, tab_pred, tab_method = st.tabs(
    ["🌍 Empreinte par pays", "🔀 Effet de substitution", "🔮 Prédiction 2030", "📖 Méthode & sources"]
)

# =============================================================================
# 1. EMPREINTE PAR PAYS
# =============================================================================
with tab_map:
    c1, c2, c3 = st.columns([2, 3, 2])
    zone = c1.selectbox("Zone", zone_options, index=zone_options.index("Union européenne (28)"))
    year = c2.slider("Année", 2000, year_max, min(im.LAST_OBSERVED_YEAR, year_max))
    mode = c3.segmented_control("Affichage", ["Par habitant", "Total"], default="Par habitant")
    per_cap = mode != "Total"
    col, unit = ("cap", ind.cap_unit) if per_cap else ("total", ind.total_unit)
    if year > im.LAST_OBSERVED_YEAR:
        st.caption("🔮 Année projetée par l'OCDE, pas une observation.")

    zone_code = code_of[zone]
    zy = fp[(fp.location == zone_code) & (fp.year == year)].set_index("meat")
    world = fp[(fp.location == "WLD") & (fp.year == year)]
    past = fp[(fp.location == zone_code) & (fp.year == year - 10)]

    if zy.empty:
        st.warning(f"Pas de données complètes pour {zone} en {year}.")
    else:
        cap = zy.cap.sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            f"Par habitant en {year}", f"{fmt_auto(cap)} {ind.cap_unit}",
            delta=f"{(cap / world.cap.sum() - 1) * 100:+.0f} % vs moyenne mondiale" if not world.empty else None,
            delta_color="inverse", border=True,
        )
        k2.metric(
            "Total de la zone", f"{fmt_auto(zy.total.sum())} {ind.total_unit}",
            delta=f"{(zy.total.sum() / past.total.sum() - 1) * 100:+.0f} % en 10 ans" if not past.empty else None,
            delta_color="inverse", border=True,
        )
        beef_kg = zy.kg_cap["BEEF"] / zy.kg_cap.sum()
        beef_imp = zy.cap["BEEF"] / cap
        k3.metric(
            "Poids du bœuf", f"{beef_imp:.0%} de l'impact",
            delta=f"pour {beef_kg:.0%} des kg consommés", delta_color="off", border=True,
        )
        if ind_key in EQUIVALENTS:
            per_unit, label, note = EQUIVALENTS[ind_key]
            raw_cap = (zy.kg_cap * zy.coef).sum()  # dans l'unité du coefficient
            k4.metric("Soit, par habitant et par an", f"≈ {fmt_auto(raw_cap / per_unit)}",
                      delta=label, delta_color="off", border=True, help=note)
        else:
            k4.metric("Consommation", f"{fmt(zy.kg_cap.sum(), 1)} kg/hab",
                      delta="de viande par an", delta_color="off", border=True)

    st.subheader(f"Carte et classement — {year}")
    m1, m2 = st.columns([3, 2])
    cy = countries[countries.year == year]
    by_country = cy.groupby(["location", "zone"])[col].sum().reset_index()
    fig = px.choropleth(
        by_country, locations="location", color=col, hover_name="zone",
        color_continuous_scale=ind.color_scale, projection="natural earth",
        labels={col: unit}, hover_data={"location": False, col: ":,.1f"},
    )
    fig.update_geos(showcountries=True, countrycolor="#ccc", showframe=False,
                    landcolor="#eee", showland=True)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430,
                      coloraxis_colorbar=dict(title=unit, thickness=12))
    m1.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    m1.caption("Pays en gris : non couverts par les données OCDE (seuls ~40 pays sont détaillés).")

    order = by_country.sort_values(col).zone.tolist()
    fig = px.bar(
        cy, x=col, y="zone", color="meat", orientation="h",
        color_discrete_map=im.MEAT_COLORS, category_orders={"zone": order[::-1], "meat": im.MEATS},
        labels={col: unit, "zone": "", "meat": "Viande"},
        hover_data={"kg_cap": ":.1f"},
    )
    fig.for_each_trace(lambda t: t.update(name=im.MEAT_LABELS[t.name]))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0), legend=dict(
        orientation="h", y=1.04, x=0, title=None), yaxis=dict(tickfont_size=10))
    with m2.container(height=450, border=False):
        fig.update_layout(height=max(430, 16 * len(order)))
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    st.subheader("Évolution dans le temps")
    e1, e2 = st.columns([3, 2])
    picked = e1.multiselect(
        "Zones à comparer", zone_options,
        default=["Monde", "Union européenne (28)", "États-Unis", "Chine", "Inde"],
        max_selections=8,
    )
    ts = fp[fp.zone.isin(picked)].groupby(["zone", "year"])[col].sum().reset_index()
    fig = px.line(ts, x="year", y=col, color="zone", markers=False,
                  labels={col: unit, "year": "", "zone": ""})
    fig.update_traces(line_width=2.5, hovertemplate="%{y:,.1f}")
    fig.update_layout(height=380, hovermode="x unified", margin=dict(t=10, b=0),
                      legend=dict(orientation="h", y=-0.15))
    shade_projections(fig, year_max)
    e1.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    zt = fp[fp.location == zone_code]
    fig = px.area(zt, x="year", y=col, color="meat", color_discrete_map=im.MEAT_COLORS,
                  category_orders={"meat": im.MEATS}, labels={col: unit, "year": ""})
    fig.for_each_trace(lambda t: t.update(name=im.MEAT_LABELS[t.name], hovertemplate="%{y:,.1f}"))
    fig.update_layout(height=380, hovermode="x unified", margin=dict(t=40, b=0),
                      title=dict(text=f"{zone} : détail par viande", font_size=14),
                      legend=dict(orientation="h", y=-0.15, title=None))
    shade_projections(fig, year_max)
    e2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# =============================================================================
# 2. EFFET DE SUBSTITUTION
# =============================================================================
with tab_subst:
    st.markdown(
        "#### Le passage du bœuf à la volaille a-t-il freiné l'impact ?\n"
        "On compare l'impact **réel** à celui qu'on aurait observé si chaque pays avait "
        "gardé **la même répartition entre viandes** qu'en année de départ, à quantités "
        "consommées identiques."
    )
    s1, s2, s3 = st.columns([2, 3, 2])
    s_zone = s1.selectbox("Zone ", zone_options, index=0, key="s_zone")
    s_code = code_of[s_zone]
    s_years = sorted(fp[fp.location == s_code].year.unique())
    y0, y1 = s2.select_slider(
        "Période", options=s_years,
        value=(max(2000, s_years[0]), min(im.LAST_OBSERVED_YEAR, s_years[-1])),
    )
    s_mode = s3.segmented_control("Mesure", ["Total", "Par habitant"], default="Total", key="s_mode")
    s_cap = s_mode == "Par habitant"
    s_unit = ind.cap_unit if s_cap else ind.total_unit

    if y0 >= y1:
        st.info("Choisissez une période d'au moins deux ans.")
    else:
        dec = im.decompose(fp, s_code, y0, y1, s_cap)
        cf = im.counterfactual(fp, s_code, y0, s_cap).loc[:y1]
        mix = dec["effects"]["Mix des viandes"]
        frozen_end = cf.frozen_mix.iloc[-1]

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Variation réelle {y0} → {y1}",
                  f"{(dec['end'] / dec['start'] - 1) * 100:+.0f} %",
                  delta=f"{fmt_auto(dec['start'])} → {fmt_auto(dec['end'])} {s_unit}",
                  delta_color="off", border=True)
        k2.metric(f"Sans changement de mix",
                  f"{(frozen_end / dec['start'] - 1) * 100:+.0f} %",
                  delta=f"{fmt_auto(frozen_end)} {s_unit} en {y1}", delta_color="off", border=True)
        k3.metric("Effet du mix des viandes", f"{fmt_auto(mix)} {s_unit}",
                  delta=f"{pct(mix / dec['start'] * 100)} de l'impact de {y0}",
                  delta_color="inverse", border=True,
                  help="Négatif = la substitution entre viandes a réduit l'impact.")

        g1, g2 = st.columns([3, 2])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cf.index, y=cf.frozen_mix, name=f"Si le mix de {y0} était resté",
                                 line=dict(color="#888", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=cf.index, y=cf.real, name="Réel",
                                 line=dict(color="#B5473A", width=3),
                                 fill="tonexty",
                                 fillcolor="rgba(107,143,113,0.25)" if mix <= 0 else "rgba(181,71,58,0.2)"))
        fig.update_traces(hovertemplate="%{y:,.1f}")
        fig.update_layout(height=400, hovermode="x unified", yaxis_title=s_unit,
                          margin=dict(t=40, b=0), legend=dict(orientation="h", y=-0.12),
                          title=dict(text="Impact réel vs mix figé (zone colorée = écart)", font_size=14))
        shade_projections(fig, y1)
        g1.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        eff = dec["effects"]
        fig = go.Figure(go.Waterfall(
            x=[str(y0)] + list(eff) + [str(y1)],
            measure=["absolute"] + ["relative"] * len(eff) + ["total"],
            y=[dec["start"]] + list(eff.values()) + [dec["end"]],
            text=[fmt_auto(dec["start"])] + [f"{v:+,.0f}".replace(",", " ") if abs(v) >= 100 else f"{v:+.1f}".replace(".", ",") for v in eff.values()] + [fmt_auto(dec["end"])],
            textposition="outside",
            increasing=dict(marker_color="#B5473A"), decreasing=dict(marker_color="#6B8F71"),
            totals=dict(marker_color="#555"), connector=dict(line_color="#bbb"),
        ))
        fig.update_layout(height=400, yaxis_title=s_unit, margin=dict(t=40, b=0), showlegend=False,
                          title=dict(text="Ce qui explique la variation", font_size=14))
        g2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        with st.expander("Comment lire la décomposition ?"):
            st.markdown(
                "L'impact se décompose en **population × kg consommés par habitant × "
                "impact moyen d'un kg de viande**. Ce dernier terme ne bouge que si la "
                "répartition entre viandes change (les coefficients sont fixes) : c'est "
                "l'**effet mix**. Méthode LMDI (*Logarithmic Mean Divisia Index*), "
                "qui répartit la variation sans résidu."
            )

        st.subheader("Répartition des kg vs répartition de l'impact")
        rows = []
        for y in (y0, y1):
            d = fp[(fp.location == s_code) & (fp.year == y)].set_index("meat")
            for m in im.MEATS:
                rows.append({"Année": str(y), "Mesure": "kg consommés", "meat": m,
                             "part": d.kg_cap[m] / d.kg_cap.sum()})
                rows.append({"Année": str(y), "Mesure": ind.label, "meat": m,
                             "part": d.cap[m] / d.cap.sum()})
        shares = pd.DataFrame(rows)
        fig = px.bar(shares, x="part", y="Année", color="meat", facet_row="Mesure", orientation="h",
                     color_discrete_map=im.MEAT_COLORS, category_orders={"meat": im.MEATS},
                     text=shares.part.map(lambda p: f"{p:.0%}" if p >= 0.05 else ""),
                     labels={"part": "", "Année": ""})
        fig.for_each_trace(lambda t: t.update(name=im.MEAT_LABELS[t.name], hovertemplate="%{x:.1%}"))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=300, xaxis_tickformat=".0%", margin=dict(t=10, b=0),
                          legend=dict(orientation="h", y=1.12, title=None))
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    st.subheader(f"Comparaison des pays — effet mix {y0} → {y1}")
    st.caption("Effet de la substitution entre viandes, en % de l'impact de départ. "
               "Vert = la répartition a évolué vers des viandes moins impactantes.")
    rows = []
    for code in fp.location.unique():
        yrs = set(fp[fp.location == code].year)
        if y0 in yrs and y1 in yrs and y0 < y1:
            d = im.decompose(fp, code, y0, y1, per_capita=True)
            rows.append({"zone": im.ZONE_NAMES[code],
                         "mix_pct": d["effects"]["Mix des viandes"] / d["start"] * 100,
                         "agg": code in im.AGGREGATES})
    if rows:
        comp = pd.DataFrame(rows).sort_values("mix_pct")
        fig = px.bar(comp, x="mix_pct", y="zone", orientation="h", color="mix_pct",
                     color_continuous_scale=["#6B8F71", "#f2f2f2", "#B5473A"],
                     color_continuous_midpoint=0, labels={"mix_pct": "Effet mix (%)", "zone": ""})
        fig.update_traces(hovertemplate="%{y} : %{x:+.1f} %<extra></extra>")
        fig.update_layout(height=max(400, 17 * len(comp)), coloraxis_showscale=False,
                          margin=dict(t=10, b=0), yaxis=dict(tickfont_size=10))
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# =============================================================================
# 3. PRÉDICTION
# =============================================================================
@st.cache_data
def load_forecast():
    if not fc_module.OUT_FC.exists():
        return None, None
    return pd.read_csv(fc_module.OUT_FC), pd.read_csv(fc_module.OUT_BT)


@st.cache_data
def population_path() -> pd.Series:
    """Population (millions) par zone et année jusqu'en 2030.

    Déduite des données OCDE jusqu'en 2026, puis prolongée par la tendance
    linéaire des 5 dernières années.
    """
    cons, _ = load()
    pop = cons.groupby(["location", "year"])["pop"].first() / 1e6
    rows = []
    for loc, s in pop.groupby(level=0):
        s = s.droplevel(0)
        slope = (s.iloc[-1] - s.iloc[-5]) / 4
        for y in range(s.index.max() + 1, fc_module.HORIZON_END + 1):
            rows.append(((loc, y), s.iloc[-1] + slope * (y - s.index.max())))
    ext = pd.Series(dict(rows))
    ext.index.names = pop.index.names
    return pd.concat([pop, ext]).sort_index()


MODEL_LABELS = {
    "gbm": "Gradient boosting global", "ensemble": "Ensemble (moyenne)",
    "holt": "Lissage de Holt amorti", "linear": "Tendance linéaire (10 ans)",
    "naive": "Naïf (dernière valeur)",
}

def prediction_tab() -> None:
    fc_all, bt = load_forecast()
    if fc_all is None:
        st.warning("Prévisions absentes : lancez `python forecast.py` puis rechargez la page.")
        return
    best_model = fc_all.loc[fc_all.best, "model"].iloc[0]

    st.markdown(
        "#### Où en sera la consommation de viande — et son impact — en 2030 ?\n"
        "Un modèle apprend sur les **années réellement observées (1990-2017)** puis prolonge "
        "chaque série zone × viande. La bande colorée montre l'incertitude (80 %), "
        "calibrée sur les erreurs passées du modèle."
    )
    p1, p2, p3 = st.columns([2, 2, 3])
    p_zone = p1.selectbox("Zone  ", zone_options, index=0, key="p_zone")
    p_code = code_of[p_zone]
    model_keys = [best_model] + [m for m in MODEL_LABELS if m != best_model]
    p_model = p2.selectbox(
        "Modèle", model_keys,
        format_func=lambda m: MODEL_LABELS[m] + (" ⭐ recommandé" if m == best_model else ""),
        help="Le modèle recommandé est celui qui a fait le moins d'erreurs au backtest "
             "(voir plus bas).",
    )
    subst = p3.slider(
        "🔀 Scénario : part du bœuf remplacée par de la volaille d'ici 2030", 0, 50, 0,
        step=5, format="%d %%",
        help="Transfert progressif (linéaire de 2018 à 2030), à kg de viande constants.",
    ) / 100

    hist = fp[(fp.location == p_code) & (~fp.projection)]
    oecd = load()[0]
    oecd = oecd[(oecd.location == p_code) & oecd.projection]
    pred = fc_all[(fc_all.location == p_code) & (fc_all.model == p_model)].copy()
    if pred.empty or hist.empty:
        st.info("Pas de prévision disponible pour cette zone.")
        return

    # --- Scénario : on transfère une part du bœuf vers la volaille ----------
    ramp = ((pred.year - im.LAST_OBSERVED_YEAR) / (fc_module.HORIZON_END - im.LAST_OBSERVED_YEAR)).clip(0, 1)
    pred["moved"] = 0.0
    beef = pred.meat == "BEEF"
    pred.loc[beef, "moved"] = pred.loc[beef, "kg_cap"] * subst * ramp[beef]
    moved_by_year = pred[beef].set_index("year").moved
    pred["kg_scen"] = pred.kg_cap - pred.moved
    poultry = pred.meat == "POULTRY"
    pred.loc[poultry, "kg_scen"] += pred.loc[poultry, "year"].map(moved_by_year).values

    coef = coefs[ind_key]
    pop = population_path()
    pop_of = lambda d: d.set_index(["location", "year"]).index.map(pop).to_numpy()
    p_mode = st.segmented_control("Mesure  ", ["Par habitant", "Total"], default="Par habitant",
                                  key="p_mode")
    p_cap = p_mode != "Total"
    scale = ind.cap_factor if p_cap else ind.total_factor
    p_unit = ind.cap_unit if p_cap else ind.total_unit

    def impact(d: pd.DataFrame, kg_col: str) -> pd.Series:
        v = d[kg_col] * d.meat.map(coef) * scale
        if not p_cap:
            v = v * pop_of(d) * 1e6
        return v.groupby(d.year).sum()

    hist_imp = hist.groupby("year")["cap" if p_cap else "total"].sum()
    base_imp, scen_imp = impact(pred, "kg_cap"), impact(pred, "kg_scen")
    lo_imp, hi_imp = impact(pred, "lo"), impact(pred, "hi")
    oecd_imp = impact(oecd, "kg_cap") if not oecd.empty else None

    kg17 = hist[hist.year == im.LAST_OBSERVED_YEAR].kg_cap.sum()
    kg30 = pred[pred.year == fc_module.HORIZON_END].kg_cap.sum()
    i17, i30, s30 = hist_imp.loc[im.LAST_OBSERVED_YEAR], base_imp.iloc[-1], scen_imp.iloc[-1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Consommation prévue en 2030", f"{fmt(kg30, 1)} kg/hab",
              delta=f"{pct((kg30 / kg17 - 1) * 100)} vs 2017", delta_color="off", border=True)
    k2.metric(f"{ind.label} en 2030", f"{fmt_auto(i30)} {p_unit}",
              delta=f"{pct((i30 / i17 - 1) * 100)} vs 2017", delta_color="inverse", border=True)
    k3.metric("Avec le scénario de substitution", f"{fmt_auto(s30)} {p_unit}",
              delta=f"{pct((s30 / i30 - 1) * 100)} vs tendance" if subst else "réglez le curseur ↑",
              delta_color="inverse" if subst else "off", border=True)

    def band_fig(x_hist, y_hist, fc_df, y_col, lo_col, hi_col, unit, oecd_s=None, scen=None, height=380):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fc_df.index) + list(fc_df.index[::-1]),
                                 y=list(fc_df[hi_col]) + list(fc_df[lo_col][::-1]),
                                 fill="toself", fillcolor="rgba(181,71,58,0.15)", line_width=0,
                                 name="Intervalle 80 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_hist, y=y_hist, name="Observé", line=dict(color="#333", width=2.5)))
        # On relie la prévision au dernier point observé pour éviter un trou visuel
        fig.add_trace(go.Scatter(x=[x_hist[-1], *fc_df.index], y=[y_hist[-1], *fc_df[y_col]],
                                 name="Prévision", line=dict(color="#B5473A", width=3)))
        if oecd_s is not None and len(oecd_s):
            fig.add_trace(go.Scatter(x=oecd_s.index, y=oecd_s.values, name="Projection OCDE (2018)",
                                     line=dict(color="#888", dash="dot", width=2)))
        if scen is not None:
            fig.add_trace(go.Scatter(x=[x_hist[-1], *scen.index], y=[y_hist[-1], *scen.values],
                                     name="Scénario substitution", line=dict(color="#6B8F71", width=3, dash="dash")))
        fig.add_vrect(x0=im.LAST_OBSERVED_YEAR + 0.5, x1=fc_module.HORIZON_END + 0.5,
                      fillcolor="grey", opacity=0.06, line_width=0)
        fig.update_traces(hovertemplate="%{y:,.1f}")
        fig.update_layout(height=height, hovermode="x unified", yaxis_title=unit,
                          margin=dict(t=30, b=0), legend=dict(orientation="h", y=-0.15))
        return fig

    g1, g2 = st.columns(2)
    kg_hist = hist.groupby("year").kg_cap.sum()
    kg_fc = pred.groupby("year")[["kg_cap", "lo", "hi"]].sum()
    oecd_kg = oecd.groupby("year").kg_cap.sum() if not oecd.empty else None
    fig = band_fig(list(kg_hist.index), list(kg_hist.values), kg_fc, "kg_cap", "lo", "hi",
                   "kg / habitant", oecd_kg)
    fig.update_layout(title=dict(text="Consommation totale de viande", font_size=14))
    g1.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    imp_fc = pd.DataFrame({"v": base_imp, "lo": lo_imp, "hi": hi_imp})
    fig = band_fig(list(hist_imp.index), list(hist_imp.values), imp_fc, "v", "lo", "hi", p_unit,
                   oecd_imp, scen_imp if subst else None)
    fig.update_layout(title=dict(text=f"{ind.icon} {ind.label}", font_size=14))
    g2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    st.subheader("Détail par viande")
    cols = st.columns(4)
    for c, meat in zip(cols, im.MEATS):
        h = hist[hist.meat == meat].set_index("year").kg_cap
        f = pred[pred.meat == meat].set_index("year")
        o = oecd[oecd.meat == meat].set_index("year").kg_cap if not oecd.empty else None
        fig = band_fig(list(h.index), list(h.values), f, "kg_cap", "lo", "hi", "kg/hab", o, height=260)
        fig.update_layout(title=dict(text=im.MEAT_LABELS[meat], font_size=13), showlegend=False,
                          margin=dict(t=30, b=0, l=0, r=0), xaxis=dict(tickvals=[1990, 2010, 2030]))
        c.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption("Trait plein rouge = notre prévision · pointillés gris = projection OCDE publiée en 2018 "
               "(pour comparaison, ce n'est pas une donnée réelle).")

    st.subheader("🧪 Quelle confiance accorder au modèle ?")
    st.markdown(
        "On a **rejoué le passé** : le modèle est entraîné sur les données jusqu'en 2005, 2007, "
        "2009 ou 2011, puis on compare ses prévisions des 6 années suivantes à ce qui s'est "
        "réellement passé. L'erreur (WAPE) = somme des écarts absolus / somme des valeurs réelles."
    )
    sc = fc_module.score(bt, ["model"]).sort_values("wape")
    by_h = fc_module.score(bt, ["model", "h"]).reset_index()
    t1, t2 = st.columns([2, 3])
    t1.dataframe(
        pd.DataFrame({
            "Modèle": [MODEL_LABELS[m] + (" ⭐" if m == best_model else "") for m in sc.index],
            "Erreur (WAPE)": sc.wape.map(lambda v: f"{v:.1%}"),
            "Erreur moy. (kg/hab)": sc.mae.map(lambda v: fmt(v, 2)),
        }), hide_index=True, width="stretch",
    )
    fig = px.line(by_h.assign(model=by_h.model.map(MODEL_LABELS)), x="h", y="wape", color="model",
                  markers=True, labels={"h": "Années après la fin de l'apprentissage", "wape": "Erreur (WAPE)", "model": ""})
    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(t=10, b=0),
                      legend=dict(orientation="h", y=-0.3))
    t2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    with st.expander("🔍 Comment fonctionne la prédiction, en détail"):
        st.markdown(
            f"""
1. **Données d'apprentissage** : 180 séries (≈ 45 zones × 4 viandes), kg/hab de 1990 à 2017.
   Les projections OCDE ne servent **jamais** à l'apprentissage.
2. **Transformation** : chaque série est passée en `log(1 + kg)` → les tendances deviennent
   des taux de croissance, et une prévision ne peut pas être négative.
3. **Cinq modèles comparés** :
   - *Naïf* : la consommation reste au niveau de 2017. C'est la référence à battre.
   - *Tendance linéaire* : droite ajustée sur les 10 dernières années.
   - *Holt amorti* : lissage exponentiel dont la tendance s'atténue avec le temps.
   - *Gradient boosting global* : un seul modèle entraîné sur **toutes** les séries.
     À partir des 5 dernières années (niveau, variations annuelles, type de viande), il prédit
     la variation de l'année suivante, puis on réinjecte la prédiction pour aller plus loin
     (prévision récursive). Il apprend par exemple qu'une consommation déjà élevée croît
     moins vite.
   - *Ensemble* : moyenne des trois précédents.
4. **Sélection** : le modèle au WAPE le plus faible au backtest est recommandé
   (ici : *{MODEL_LABELS[best_model]}*).
5. **Incertitude** : les erreurs du backtest donnent, pour chaque horizon, les quantiles
   10 % et 90 % de l'erreur ; au-delà de 6 ans, la largeur est extrapolée en √h.
   Pour le total, on additionne les bornes des 4 viandes (hypothèse prudente).
6. **Impact** : kg prévus × coefficients Poore & Nemecek (× population pour un total ;
   population OCDE jusqu'en 2026, puis prolongée par tendance linéaire).

**Limites** : aucune variable explicative (revenu, prix, politiques) ; les chocs comme la peste
porcine africaine en Chine (2018-2019) ou le Covid sont imprévisibles par construction.
Le gain du meilleur modèle sur le modèle naïf est réel mais modeste : la consommation de viande
évolue lentement.
"""
        )


with tab_pred:
    prediction_tab()

# =============================================================================
# 4. MÉTHODE
# =============================================================================
with tab_method:
    st.markdown(
        """
#### Calcul
**Impact = kg consommés par habitant × coefficient d'impact par kg**, pour chaque viande,
puis × population pour le total d'une zone.

- **Consommation** : OCDE-FAO *Meat Consumption* (kg/hab, poids au détail), 1990-2017
  observé, 2018-2026 projeté. La population est déduite des tonnages (poids carcasse
  converti au détail avec les facteurs OCDE : 0,7 bœuf · 0,78 porc · 0,88 volaille et ovin).
- **Coefficients** : Poore & Nemecek (2018), *Reducing food's environmental impacts
  through producers and consumers*, Science 360. Moyennes mondiales 2010, cycle de vie
  complet (alimentation animale, changement d'usage des sols, élevage, transformation,
  transport, distribution).
"""
    )
    tbl = coefs.rename(index=im.MEAT_LABELS).rename(columns={
        k: f"{v.icon} {v.label} ({v.coef_unit}/kg)" for k, v in im.INDICATORS.items()})
    st.dataframe(tbl.style.format("{:,.1f}"), width="stretch")
    st.caption(f"Bœuf = {beef_share:.0%} troupeau à viande + {1 - beef_share:.0%} troupeau laitier "
               "(réglable dans la barre latérale).")
    st.markdown(
        """
#### Limites à garder en tête
- **Coefficients mondiaux fixes** : un kg de bœuf issu de zones déforestées émet bien
  plus que la moyenne, un kg de bœuf européen souvent moins. Les écarts entre pays
  reflètent donc uniquement les *quantités* et le *mix*, pas les pratiques d'élevage.
- **Pas d'évolution technique** : les gains de productivité depuis 1990 ne sont pas
  pris en compte (coefficients de 2010 appliqués à toutes les années).
- **Eau** : il s'agit des **prélèvements** (eau bleue : irrigation, abreuvement), pas
  de l'empreinte eau totale incluant l'eau de pluie (≈ 15 000 L/kg de bœuf chez
  Mekonnen & Hoekstra).
- **Couverture** : poisson, chèvre (partiellement), gibier et abats non inclus.
  ~40 pays détaillés seulement ; la France est incluse dans l'UE 28.
- **Données exclues** : le Soudan (valeurs par habitant incohérentes avec ses tonnages)
  et les années où une des 4 viandes manque.
- **Équivalents** (km en voiture, douches, courts de tennis) : ordres de grandeur.
"""
    )
