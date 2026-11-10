from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Lincolnshire Mental Health Dashboard", 
    page_icon=str(Path(__file__).resolve().parent / "assets" / "favicon.png"),
    layout="wide"
)

st.title("Lincolnshire Mental Health Dashboard")
st.markdown(
    """
This dashboard contains:

- Lincolnshire Mental Health Need Index
- Small Area Mental Health Index (SAMHI)

Use the page selector in the sidebar to navigate.
"""
)

st.sidebar.caption(
    """
    © 2026 [University of Lincoln](https://www.lincoln.ac.uk/)

    [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/)

    Lincolnshire Mental Health Need Index v1.0
    """
)
