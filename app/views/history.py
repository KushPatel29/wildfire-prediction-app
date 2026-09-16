"""Fires since 2000: what the model learns from."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title("Canada's fires since 2000")
sh.kicker("Every fire in the Canadian National Fire Database point layer from 2000 to 2024, excluding prescribed "
          "burns. Area is the sum of recorded final fire sizes.")

by_year = sh.load_published("fires_by_year")
by_month = sh.load_published("fires_by_month")
provinces = sorted(by_year["province"].unique(), key=lambda p: -by_year.loc[by_year["province"] == p, "fires"].sum())
chosen = st.multiselect("Provinces and territories", provinces, placeholder="All of Canada")
if chosen:
    by_year = by_year[by_year["province"].isin(chosen)]
    by_month = by_month[by_month["province"].isin(chosen)]

fires, area = int(by_year["fires"].sum()), float(by_year["area_ha"].sum())
lightning = by_year[by_year["cause_label"] == "Lightning"]
worst = by_year.groupby("year")["area_ha"].sum()
cols = st.columns(4)
cols[0].metric("Fires recorded", f"{fires:,}", border=True)
cols[1].metric("Area burned", f"{area / 1e6:,.1f} million ha", border=True)
cols[2].metric("Started by lightning", sh.pct(lightning["fires"].sum() / max(fires, 1), 0), border=True,
               help=f"...and {sh.pct(lightning['area_ha'].sum() / max(area, 1), 0)} of the area burned.")
cols[3].metric("Worst year for area", f"{int(worst.idxmax())}", border=True,
               help=f"{worst.max() / 1e6:,.1f} million ha, {sh.pct(worst.max() / max(area, 1), 0)} of the 25-year total.")

st.markdown(
    f"**Lightning starts {sh.pct(lightning['fires'].sum() / max(fires, 1), 0)} of fires but "
    f"{sh.pct(lightning['area_ha'].sum() / max(area, 1), 0)} of the area burned.** Lightning fires start in remote "
    "forest and are found late; human-caused fires start near roads and towns and are usually caught small."
)

causes = ["Lightning", "Human", "Unknown"]
left, right = st.columns(2)
for column, value, title, axis in ((left, "fires", "Fires by year", "Fires"),
                                   (right, "area_ha", "Area burned by year", "Million ha")):
    with column:
        pivot = by_year.pivot_table(index="year", columns="cause_label", values=value, aggfunc="sum").reindex(columns=causes).fillna(0)
        if value == "area_ha":
            pivot = pivot / 1e6
        fig = go.Figure()
        for cause in causes:
            fig.add_bar(x=pivot.index, y=pivot[cause], name=cause, marker_color=sh.CAUSE_COLOURS[cause],
                        hovertemplate="%{x}: %{y:,.2f}<extra>" + cause + "</extra>" if value == "area_ha"
                        else "%{x}: %{y:,.0f}<extra>" + cause + "</extra>")
        sh.show(fig, height=330, barmode="stack", bargap=0.15, title=title,
                yaxis=dict(title=axis))

st.subheader("When the season runs")
season = by_month.pivot_table(index="province", columns="month", values="fires", aggfunc="sum").fillna(0)
season = season.loc[season.sum(axis=1).sort_values().index]
share = season.div(season.sum(axis=1), axis=0)
months = [pd.Timestamp(2000, m, 1).strftime("%b") for m in season.columns]
fig = go.Figure(go.Heatmap(
    z=share.to_numpy(), x=months, y=[sh.PROVINCE_SHORT.get(p, p) for p in share.index],
    text=np.vectorize(lambda v: "" if v < 0.005 else f"{v:.0%}")(share.to_numpy()), texttemplate="%{text}",
    colorscale=[[0, sh.SURFACE], [0.35, "#7A4A22"], [0.7, sh.MODEL], [1, "#FDE3C4"]], showscale=False,
    customdata=season.to_numpy(), hovertemplate="%{y}, %{x}: %{z:.1%} of the year's fires (%{customdata:,.0f})<extra></extra>"))
sh.show(fig, height=80 + 30 * len(share), xaxis=dict(side="top"))
st.caption("Each row is one province or territory's fires split by month, so seasons compare regardless of size. "
           "April to October holds 98% of fires nationally, which is why the model covers those months.")
sh.sources_footer()
