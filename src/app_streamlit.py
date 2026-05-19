# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SDI-NLP Candidate Review", layout="wide")
st.title("SDI-NLP: Candidate Review")

base = Path("artifacts/parsed")
wide_path = base / "filled_suggestions_wide.csv"
long_path = base / "filled_suggestions_long.csv"

if not wide_path.exists() and not long_path.exists():
    st.warning("Run the pipeline first to generate suggestions.")
    st.stop()

if long_path.exists():
    long_df = pd.read_csv(long_path)
    filenames = sorted(long_df["filename"].unique().tolist())
    fname = st.selectbox("Record filename", filenames)
    sub = long_df[long_df["filename"] == fname].copy()
    st.dataframe(sub[["field", "before_present", "predicted_value", "confidence", "action", "top3_candidates"]], use_container_width=True)
else:
    wide_df = pd.read_csv(wide_path)
    filenames = sorted(wide_df["filename"].unique().tolist())
    fname = st.selectbox("Record filename", filenames)
    r = wide_df[wide_df["filename"] == fname].iloc[0]
    st.write(r.to_dict())
