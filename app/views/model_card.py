"""Model card: what the model predicts, how it was tested, and where it falls short."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title("Model card")
metrics = sh.load_metrics()
train, valid = metrics["train_years"], metrics["validation_years"]
sh.kicker(f"Two gradient-boosted classifiers over 771 one-degree cells and every fire-season day. Trained on "
          f"{train[0]}–{train[1]}, calibrated and early-stopped on {valid[0]}–{valid[1]}, tested once on "
          f"{metrics['test_from']}–2024.")

TARGET = {"has_fire": "Any new fire", "has_large_fire": "A fire that grows past 200 ha"}
SPLIT = {"validation": f"{valid[0]}–{valid[1]}", "test": f"{metrics['test_from']}–2024"}
METHOD = {"model": "This model", "climatology": "Normal for the month", "fwi_logistic": "FWI logistic regression"}
FEATURE_NOTES = {
    "temp": "Noon temperature", "rh": "Noon relative humidity", "ws": "Noon wind speed",
    "precip": "Rain in the 24 hours to noon", "ffmc": "Fine Fuel Moisture Code", "dmc": "Duff Moisture Code",
    "dc": "Drought Code", "isi": "Initial Spread Index", "bui": "Buildup Index", "fwi": "Fire Weather Index",
    "dsr": "Daily Severity Rating", "fwi_mean_3d": "FWI, 3-day mean", "fwi_mean_7d": "FWI, 7-day mean",
    "fwi_max_7d": "FWI, 7-day maximum", "isi_mean_3d": "ISI, 3-day mean", "temp_mean_3d": "Temperature, 3-day mean",
    "rh_mean_3d": "Humidity, 3-day mean", "precip_sum_3d": "Rain, 3 days", "precip_sum_7d": "Rain, 7 days",
    "precip_sum_14d": "Rain, 14 days", "days_since_rain": "Days since 2 mm of rain",
    "dc_change_7d": "Drought Code change over 7 days", "doy_sin": "Day of year (sine)",
    "doy_cos": "Day of year (cosine)", "lat": "Cell latitude", "lon": "Cell longitude",
    "lightning_share": "Share of the cell's past fires started by lightning",
    "clim_month_rate": "Cell's fire-day rate for the month, training years",
    "clim_cell_rate": "Cell's fire-day rate for the season, training years",
    "station_km": "Distance to the nearest reporting station",
}

st.subheader("What it predicts")
st.markdown(
    """
For one cell on one day of the April–October season, the probability that the National Fire Database records
**at least one new fire** reported there, and separately that a fire reported there **grows past 200 ha**. Large
fires are 5.6% of the record and 99% of the area burned. Fires set as prescribed burns are excluded.
"""
)

st.subheader("How it scored on seasons it never saw")
rows = []
for target, info in metrics["targets"].items():
    for split, methods in info["splits"].items():
        for method, s in methods.items():
            rows.append({"Target": TARGET[target], "Seasons": SPLIT[split], "Method": METHOD[method],
                         "ROC-AUC": s["roc_auc"], "PR-AUC": s["pr_auc"], "Brier": s["brier"],
                         "Fires in top 10%": s["share_of_fires_in_top_decile"], "Lift": s["lift_top_decile"]})
table = pd.DataFrame(rows)
# Streamlit's "percent" preset prints whatever precision the float has, so 45.64%
# and 52% land in the same column; carry the value as a percentage instead.
table["Fires in top 10%"] *= 100
target_pick = st.segmented_control("Target", options=list(TARGET.values()), default=TARGET["has_fire"], required=True)
st.dataframe(table[table["Target"] == (target_pick or TARGET["has_fire"])].drop(columns="Target"), hide_index=True,
             column_config={
                 "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                 "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                 "Brier": st.column_config.NumberColumn(format="%.4f"),
                 "Fires in top 10%": st.column_config.NumberColumn(
                     format="%.1f%%",
                     help="Share of all fires that started in the 10% of cell-days ranked riskiest, pooled over the seasons."),
                 "Lift": st.column_config.NumberColumn(format="%.1f×", help="Fire-day rate in that top 10%, against the overall rate."),
             })
st.caption("Normal for the month is each cell's own fire-day rate for that month in the training years - the "
           "baseline a model has to beat to have learned anything about weather. The FWI logistic regression uses "
           "FWI, ISI, BUI and month, roughly what a danger-class table gives an agency.")

target_key = "has_fire" if (target_pick or TARGET["has_fire"]) == TARGET["has_fire"] else "has_large_fire"
info = metrics["targets"][target_key]
left, right = st.columns(2)
with left:
    reliability = pd.DataFrame(info["reliability"])
    fig = go.Figure()
    top = float(max(reliability["predicted"].max(), reliability["observed"].max()) * 1.1)
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=sh.MUTED, dash="dot", width=1),
                    name="Perfect calibration", hoverinfo="skip")
    fig.add_scatter(x=reliability["predicted"], y=reliability["observed"], mode="lines+markers", name="Test seasons",
                    line=dict(color=sh.MODEL, width=2.5),
                    hovertemplate="Predicted %{x:.2%}<br>Observed %{y:.2%}<extra></extra>")
    sh.show(fig, height=330, title="Calibration, by tenth of predicted risk",
            xaxis=dict(title="Predicted", tickformat=".0%"), yaxis=dict(title="Observed", tickformat=".0%"))
    top_bin = reliability.iloc[-1]
    drift = "over" if top_bin["observed"] < top_bin["predicted"] else "under"
    st.caption(f"The calibration was fitted on {valid[0]}–{valid[1]} and is shown here on the test seasons, so "
               f"the drift is real rather than fitted away: the riskiest tenth {drift}-predicts, "
               f"{sh.pct(top_bin['predicted'], 1)} against {sh.pct(top_bin['observed'], 1)} observed.")
with right:
    by_year = pd.DataFrame(info["by_year"])
    fig = go.Figure(go.Bar(x=by_year["year"].astype(str), y=by_year["roc_auc"], marker_color=sh.MODEL,
                           text=by_year["roc_auc"].map(lambda v: f"{v:.3f}"), textposition="outside",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    sh.show(fig, height=330, title="ROC-AUC by test season",
            yaxis=dict(range=[0.5, 1]), showlegend=False)

left, right = st.columns(2)
with left:
    gain = pd.Series(info["feature_gain"]).sort_values().tail(15)
    fig = go.Figure(go.Bar(x=gain.to_numpy(), y=[FEATURE_NOTES.get(k, k) for k in gain.index], orientation="h",
                           marker_color=sh.MODEL, hovertemplate="%{y}: %{x:.1%} of gain<extra></extra>"))
    sh.show(fig, height=460, title="What the trees split on (share of gain)",
            xaxis=dict(tickformat=".0%"), showlegend=False)
    st.caption("Gain says where the trees found the most separation, not what causes fires; correlated inputs "
               "such as the rain windows share credit.")
with right:
    provinces = pd.DataFrame(info["by_province"]).sort_values("roc_auc", ascending=False)
    st.markdown("**By province, test seasons**")
    st.dataframe(provinces[["province", "rows", "positive_rate", "roc_auc", "share_of_fires_in_top_decile"]],
                 hide_index=True, height=430, column_config={
                     "province": "Province", "rows": st.column_config.NumberColumn("Cell-days", format="localized"),
                     "positive_rate": st.column_config.NumberColumn("Fire-day rate", format="percent"),
                     "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                     "share_of_fires_in_top_decile": st.column_config.NumberColumn("Fires in the province's top 10%", format="percent"),
                 })

st.subheader("Data and method")
st.markdown(
    f"""
- **Fires.** The Canadian National Fire Database point layer, 2000–2024: report date, location, cause and final
  size of every fire agencies recorded. Prescribed burns and records without a valid date or location are
  dropped.
- **Weather.** CWFIS noon observations from about 2,850 stations with the official FWI System codes, where CWFIS
  computed them from the station's own record. Each cell takes an inverse-distance-weighted average of up to four
  stations within 200 km, variable by variable.
- **Cells.** 1° cells with at least ten fires in 2000–2016 - 771 of them, holding almost all of Canada's recorded
  fires outside the far north.
- **Features.** {len(metrics['features'])} per cell-day: the day's weather and codes, 3- to 14-day windows of FWI,
  humidity and rain, days since rain, the week's change in Drought Code, season, position, the cell's lightning
  share, its normal fire rate for the month, and how far the nearest station is.
- **Model.** XGBoost (histogram trees, depth 7, learning rate 0.05, early stopping on {valid[0]}–{valid[1]}),
  then an isotonic calibration fitted on the same validation seasons. The test seasons were scored once.
- **Split by season, never by row.** A random split would put a July day in training and the next July day in
  test, and the weather they share would inflate every score.
"""
)
st.subheader("Limits")
st.markdown(
    """
- **Lightning is not an input.** Lightning starts most of the area burned in Canada. The model knows which cells
  tend to get lightning fires, not where today's storms are.
- **Report date, not ignition date.** A fire that smoulders for days before it is found is labelled on the day
  it was reported.
- **Coarse cells.** A 1° cell is roughly 110 × 70 km. The output says where to look, not where a fire will be.
- **Sparse stations in the north.** Where the nearest station is hundreds of kilometres away the codes are thin
  and the model leans on the cell's history instead.
- **Recording changes.** Agencies' reporting practices differ and change; the NFDB is the best national record,
  not a perfect one.
- **Not an official product.** It is not the Canadian Forest Fire Danger Rating System and not a substitute for
  CWFIS or provincial and territorial wildfire services.
"""
)
sh.sources_footer()
