"""Replay 2020–2024: the model's daily risk maps against the fires that actually started."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title("Replay a real fire season")
sh.kicker("Every day of the 2020–2024 seasons, scored by a model trained only on 2000–2016, against the fires the "
          "National Fire Database records as reported that day. The weather is what stations measured that day, "
          "so this is the forecast at lead 0.")

summary = sh.load_published("backtest_summary")
summary["year"] = summary["date"].dt.year
summary["share"] = summary["fires_in_top_decile"] / summary["fires"].where(summary["fires"] > 0)
years = sorted(summary["year"].unique())

left, right = st.columns([1, 3], vertical_alignment="bottom")
year = left.selectbox("Season", years, index=years.index(2023) if 2023 in years else len(years) - 1)
season = summary[summary["year"] == year].reset_index(drop=True)
season_days = [pd.Timestamp(d) for d in season["date"]]
busiest = pd.Timestamp(season.loc[season["fires"].idxmax(), "date"])
day = right.select_slider("Day", options=season_days, value=busiest, format_func=sh.day_label, key=f"replay-day-{year}")
row = season.set_index("date").loc[day]

cells = sh.replay_day(day)
lightning = int(cells["lightning_fires"].sum())
cols = st.columns(4)
cols[0].metric("Fires reported", f"{int(row['fires']):,}", border=True)
cols[1].metric("Inside the riskiest 10% of cells", sh.pct(row["share"], 0), border=True,
               help=f"{int(row['fires_in_top_decile']):,} of {int(row['fires']):,} fires started in the {sh.TOP_CELLS} cells the "
                    "model ranked riskiest that morning. Picking cells at random would catch about 10%.")
cols[2].metric("Fires that grew past 200 ha", f"{int(row['large_fires']):,}", border=True,
               help=f"{int(row['large_in_top_decile'])} of them started inside the top 10% of large-fire risk.")
cols[3].metric("Lightning-caused", sh.pct(lightning / max(int(row["fires"]), 1), 0), border=True,
               help="Share of the day's fires the NFDB attributes to lightning. Lightning is not a model input.")

scale_key = st.segmented_control("Colour cells by", options=["risk", "large_risk", "fwi"], default="risk",
                                 required=True, format_func=lambda k: sh.SCALES[k].title)
scale = sh.SCALES[scale_key or "risk"]
view = cells.assign(
    risk_text=cells["risk"].map(lambda v: sh.pct(v, 1)),
    large_text=cells["large_risk"].map(lambda v: sh.pct(v, 2)),
    fwi_text=cells["fwi"].map(lambda v: sh.number(v)),
)
sh.draw_map(view, scale,
            "<b>{province}</b> · cell {cell_id}<br/>New fire: <b>{risk_text}</b> · over 200 ha: {large_text}<br/>"
            "FWI {fwi_text}<br/>Fires reported: <b>{fires}</b> ({lightning_fires} lightning, {human_fires} human)",
            rings=cells[cells["fires"] > 0])
notes, caption = sh.band_notes(scale)
sh.legend(scale, notes)
st.caption(caption + " White rings: cells where fires were reported, sized by count.")

st.subheader(f"The {year} season, day by day")
fig = go.Figure()
fig.add_bar(x=season["date"], y=season["fires_in_top_decile"], name="Started in the riskiest 10% of cells",
            marker_color=sh.MODEL, hovertemplate="%{x|%b %d}: %{y} fires<extra></extra>")
fig.add_bar(x=season["date"], y=season["fires"] - season["fires_in_top_decile"], name="Started elsewhere",
            marker_color=sh.OUTSIDE, hovertemplate="%{x|%b %d}: %{y} fires<extra></extra>")
sh.day_marker(fig, day)
sh.show(fig, height=320, barmode="stack", bargap=0.1, yaxis=dict(title="Fires reported"))

busy = season[season["fires"] >= 50]
if len(busy):
    worst = busy.loc[busy["share"].idxmin()]
    worst_cells = sh.replay_day(pd.Timestamp(worst["date"]))
    worst_lightning = worst_cells["lightning_fires"].sum() / max(worst_cells["fires"].sum(), 1)
    # Why it struggled depends on who started the fires, so the sentence reads the
    # day rather than assuming: a lightning bust and a human-caused day are
    # different blind spots, and asserting the wrong one is worse than saying neither.
    if worst_lightning >= 0.5:
        because = ("Lightning is not an input. A storm crossing cells the weather rated moderate starts fires the "
                   "model has no way to see coming.")
    else:
        because = ("Most of them were human-caused, which is the blind spot the weather cannot cover: people light "
                   "fires on a long weekend or beside a road on a day the fuels are only middling.")
    st.markdown(
        f"**Where it struggled.** Of the {len(busy)} days in {year} with 50 or more fires, the weakest was "
        f"{sh.day_label(worst['date'])}: {int(worst['fires'])} fires, {sh.pct(worst['share'], 0)} of them inside the top "
        f"tenth, and {sh.pct(worst_lightning, 0)} of them lightning-caused. {because}"
    )

st.subheader("Every test season")
by_year = summary.groupby("year").agg(fires=("fires", "sum"), captured=("fires_in_top_decile", "sum"),
                                      large=("large_fires", "sum"), large_captured=("large_in_top_decile", "sum"))
by_year["share"] = by_year["captured"] / by_year["fires"]
by_year["large_share"] = by_year["large_captured"] / by_year["large"]
metrics = sh.load_metrics()["targets"]["has_fire"]
auc = {entry["year"]: entry["roc_auc"] for entry in metrics["by_year"]}
by_year["auc"] = [auc.get(y) for y in by_year.index]
st.dataframe(by_year.reset_index()[["year", "fires", "share", "large", "large_share", "auc"]], hide_index=True,
             column_config={
                 "year": st.column_config.NumberColumn("Season", format="%d"),
                 "fires": st.column_config.NumberColumn("Fires reported", format="localized"),
                 "share": st.column_config.NumberColumn("In the day's riskiest 10%", format="percent"),
                 "large": st.column_config.NumberColumn("Fires over 200 ha", format="localized"),
                 "large_share": st.column_config.NumberColumn("In the day's riskiest 10% (large-fire model)", format="percent"),
                 "auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
             })
st.caption(f"\"The day's riskiest 10%\" ranks the {sh.CELLS_IN_GRID} cells afresh every morning, the way crews would be positioned. "
           "The model card's pooled figures rank all cell-days of the season together, which also rewards knowing "
           "July is busier than April, so they run higher.")
sh.sources_footer()
