"""The current season, scored against satellite detections as it happened."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shared as sh

sh.page_style()
st.title(f"The {sh.SEASON_YEAR} season, scored as it happened")

report = sh.season_report()
if report is None:
    st.info(f"The {sh.SEASON_YEAR} season check has not been published yet (`python pipelines/season_check.py`).")
    sh.sources_footer()
    st.stop()

sh.kicker(
    f"No part of the model saw {report['year']}. Every cell on every day from {sh.day_label(report['first_day'])} to "
    f"{sh.day_label(report['last_day'])} is scored from that day's station weather, then checked against CWFIS "
    f"satellite hotspots: a cell counts as <b>new fire activity</b> on a day it shows hotspots after "
    f"{report['lookback_days']} days with none.")

pooled, daily_share = report["pooled"], report["same_day_top_decile"]
names = {"model": "Any-fire model", "large_fire_model": "Large-fire model",
         "fwi_alone": "FWI alone", "climatology": "Normal for the month"}
colours = {"model": sh.MODEL, "large_fire_model": sh.LARGE_MODEL, "fwi_alone": sh.FWI_ONLY,
           "climatology": sh.CLIMATOLOGY}
ranked = sorted(names, key=lambda k: -daily_share[k])
# The same four, named as they read inside a sentence.
in_prose = {"model": "the any-fire model", "large_fire_model": "the large-fire model",
            "fwi_alone": "FWI alone", "climatology": "normal for the month"}


def points(a: str, b: str) -> str:
    return f"{(daily_share[a] - daily_share[b]) * 100:+.1f} pts on {in_prose[b]}"


cols = st.columns(4)
cols[0].metric("New fire activity, cell-days", f"{report['new_detections']:,}", border=True,
               help=f"Out of {report['cell_days']:,} cell-days ({sh.pct(report['positive_rate'], 1)}).")
cols[1].metric("Any-fire model, top 10%", sh.pct(daily_share["model"], 1),
               delta=points("model", "climatology"), border=True,
               help=f"Each morning the {sh.CELLS_IN_GRID} cells are ranked; this is the share of new activity that appeared in the "
                    f"top {sh.TOP_CELLS}. Picking cells at random would catch 10%.")
cols[2].metric("Large-fire model, top 10%", sh.pct(daily_share["large_fire_model"], 1),
               delta=points("large_fire_model", "fwi_alone"), border=True,
               help="The model trained on fires that grow past 200 ha, ranked the same way.")
cols[3].metric("Station days scored", f"{report['station_days']}", border=True,
               help=f"{report['hotspot_days']} days of hotspot files; {report['hotspot_days_missing']} missing from "
                    "the archive, all before the season opened.")

best = ranked[0]
rest = ", ".join(f"{sh.pct(daily_share[key], 1)} for {in_prose[key]}" for key in ranked[1:])
verdict = (f"**{names[best]} ranked this season's satellite detections best**, catching "
           f"{sh.pct(daily_share[best], 1)} of new activity in the day's riskiest tenth of cells, against "
           f"{rest}.")
best_auc = max(names, key=lambda k: pooled[k]["roc_auc"])
if best_auc != best:
    verdict += (f" Across every cell-day of the season, though, {in_prose[best_auc]} has the higher ROC-AUC, "
                f"{pooled[best_auc]['roc_auc']:.3f} against {pooled[best]['roc_auc']:.3f}: the two are close enough "
                "that the order depends on the question asked.")
if daily_share["fwi_alone"] > daily_share["model"]:
    verdict += (" FWI alone out-ranks the any-fire model here, and the answer key is part of why: a satellite sees "
                "fires large and hot enough to detect from orbit, which are the fires weather drives. The any-fire "
                "model is trained on every reported start, including the small human-caused fires near roads and "
                "towns that its fire-history features are there to find and that satellites rarely see.")
st.markdown(verdict)
left, right = st.columns(2)
with left:
    fig = go.Figure(go.Bar(
        x=[names[k] for k in names], y=[daily_share[k] for k in names], marker_color=[colours[k] for k in names],
        text=[sh.pct(daily_share[k], 0) for k in names], textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=0.10, line=dict(color=sh.MUTED, dash="dot", width=1))
    sh.show(fig, height=300, title="New activity caught in the day's riskiest 10%",
            yaxis=dict(tickformat=".0%", range=[0, max(daily_share.values()) * 1.25]), showlegend=False)
    st.caption("The dotted line is what picking cells at random would catch.")
with right:
    months = pd.DataFrame(report["by_month"])
    fig = go.Figure()
    for key in names:
        fig.add_scatter(x=pd.to_datetime(months["month"], format="%m").dt.strftime("%b"), y=months[f"{key}_roc_auc"],
                        name=names[key], mode="lines+markers", line=dict(color=colours[key], width=2.5 if key == "model" else 1.5),
                        hovertemplate="%{x}: %{y:.3f}<extra>" + names[key] + "</extra>")
    sh.show(fig, height=300, title="ROC-AUC by month", yaxis=dict(range=[0.45, 1]))

daily = sh.load_published(f"season_{sh.SEASON_YEAR}_daily")
days = [pd.Timestamp(d) for d in sorted(daily["date"].unique())]
per_day = daily.groupby("date").agg(new=("new_detection", "sum"),
                                    caught=("new_detection", lambda s: int(s[daily.loc[s.index, "in_top_decile"]].sum())))
busiest = pd.Timestamp(per_day["new"].idxmax())

st.subheader("Pick a day")
day = st.select_slider("Day", options=days, value=busiest, format_func=sh.day_label)
cells = daily[daily["date"] == day]
scale_key = st.segmented_control("Colour cells by", options=["risk", "large_risk", "fwi"], default="risk",
                                 required=True, format_func=lambda k: sh.SCALES[k].title)
scale = sh.SCALES[scale_key or "risk"]
view = cells.assign(risk_text=cells["risk"].map(lambda v: sh.pct(v, 1)), fwi_text=cells["fwi"].map(sh.number),
                    normal_text=(cells["risk"] / cells["clim_month_rate"].clip(lower=0.001)).map(lambda v: f"{v:.1f}×"),
                    status=cells["new_detection"].map({1: "new fire activity", 0: "no new activity"}))
sh.draw_map(view, scale, "<b>{province}</b> · cell {cell_id}<br/>New fire: <b>{risk_text}</b> ({normal_text} normal)"
                         "<br/>FWI {fwi_text}<br/>Hotspots: {hotspots} · {status}",
            rings=cells[cells["new_detection"] == 1], ring_column="hotspots")
notes, caption = sh.band_notes(scale)
sh.legend(scale, notes)
row = per_day.loc[day]
st.caption(f"{caption} White rings: new fire activity, sized by hotspot count. On {sh.day_label(day)}, "
           f"{int(row['caught'])} of {int(row['new'])} cells with new activity were in that morning's top {sh.TOP_CELLS}.")

fig = go.Figure()
fig.add_bar(x=per_day.index, y=per_day["caught"], name="In the day's riskiest 10%", marker_color=sh.MODEL)
fig.add_bar(x=per_day.index, y=per_day["new"] - per_day["caught"], name="Elsewhere", marker_color=sh.OUTSIDE)
sh.day_marker(fig, day)
sh.show(fig, height=300, barmode="stack", bargap=0.1, yaxis=dict(title="Cells with new activity"))

with st.expander("Why hotspots, and what that label gets wrong"):
    st.markdown(
        f"""
The National Fire Database, which the 2020–2024 replay is scored against, is compiled from agency reports a
season or more after the fact, so it cannot grade {report['year']} yet. Satellite hotspots can, independently of
the model and of anything it was trained on.

They are a different answer key. A satellite pass misses fires that start and are put out between overpasses,
and fires under cloud. A large fire that spreads into the next cell shows up there as "new" activity although
nothing new started, and a persistent industrial heat source is only counted the first day it appears. The
rates are therefore not comparable to the replay's; the ranking is. All three methods above are scored against
the same label.
"""
    )
sh.sources_footer()
