"""Streamlit dashboard for a pipeline run. Reads an analytics output directory
and shows counts, zone stats, the movement heatmap, and the active-tracks
time-series.

Run:  streamlit run web/dashboard.py
Point it at a run via env var:  VIDANALYTICS_OUT=data/run/analytics streamlit run web/dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

OUT = Path(os.environ.get("VIDANALYTICS_OUT", "data/run/analytics"))

st.set_page_config(page_title="Video Analytics", layout="wide")
st.title("🎥 Video Analytics Dashboard")
st.caption(f"Reading: `{OUT}`")

summary_path = OUT / "summary.json"
if not summary_path.exists():
    st.warning(f"No summary found at {summary_path}. Run the pipeline first (see README).")
    st.stop()

summary = json.loads(summary_path.read_text())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", summary.get("n_frames", 0))
c2.metric("Tracks", summary.get("n_tracks", 0))
c3.metric("Detections", summary.get("n_detections", 0))
c4.metric(f"Avg speed ({summary.get('speed_unit', 'px/s')})", summary.get("avg_speed", 0.0))

st.subheader("Line crossings")
lc = summary.get("line_counts", [])
if lc:
    st.dataframe({
        "line": [x["name"] for x in lc],
        "forward": [x["forward"] for x in lc],
        "backward": [x["backward"] for x in lc],
        "total": [x["total"] for x in lc],
    }, use_container_width=True)
else:
    st.info("No counting lines configured.")

zs = summary.get("zone_stats", [])
if zs:
    st.subheader("Zone dwell")
    st.dataframe({
        "zone": [x["name"] for x in zs],
        "unique_tracks": [x["unique_tracks"] for x in zs],
        "total_dwell_s": [x["total_dwell_s"] for x in zs],
        "avg_dwell_s": [x["avg_dwell_s"] for x in zs],
    }, use_container_width=True)

col_a, col_b = st.columns(2)
heat_path = OUT / "heatmap.npy"
if heat_path.exists():
    heat = np.load(heat_path)
    if heat.max() > 0:
        heat = heat / heat.max()
    col_a.subheader("Movement heatmap")
    col_a.image(heat, clamp=True, use_container_width=True)

ts_path = OUT / "counts_timeseries.parquet"
if ts_path.exists():
    import pandas as pd

    ts = pd.read_parquet(ts_path).set_index("frame")
    col_b.subheader("Active tracks over time")
    col_b.line_chart(ts)
