from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from analysis.samhi import SAMHI_YEARS, get_samhi_columns
from data.loader import get_prepared_bundle_cached, resolve_base_dir
from maps.folium_map import build_choropleth_map

st.set_page_config(
    page_title="Small Area Mental Health Index (SAMHI)",
    page_icon=str(Path(__file__).resolve().parents[1] / "assets" / "favicon.png"),
    layout="wide"
)

def init_session_state() -> None:
    defaults = {
        "samhi_explorer_year": 2011,
        "samhi_explorer_mode": "Index",
        "samhi_explorer_analysis_mode": "Single Year",
        "samhi_explorer_change_from_year": 2011,
        "samhi_explorer_change_to_year": 2022,
        "samhi_explorer_playing": False,
        "samhi_explorer_interval": 0.8,
        "samhi_explorer_last_tick": 0.0,
        "samhi_explorer_restart_requested": False,
        "samhi_explorer_reset_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render() -> None:
    st.title("Small Area Mental Health Index (SAMHI)")

    st.markdown(
        """
        The <strong><a href="https://pldr.org/dataset/small-area-mental-health-index-samhi-2noyv"
        target="_blank" rel="noopener noreferrer">Small Area Mental Health Index (SAMHI)</a></strong>
        estimates the relative level of mental health need for each Lower Layer Super Output Area (LSOA)
        in England. It combines several indicators associated with mental health need into a single
        standardised index, allowing comparisons between neighbourhoods and across years.
        """,
        unsafe_allow_html=True,
    )

    with st.expander("About the SAMHI data"):
        st.markdown("""
        ### What does the SAMHI Index mean?

        - **0** = Average estimated mental health need across England.
        - **Positive values** = Higher-than-average estimated mental health need.
        - **Negative values** = Lower-than-average estimated mental health need.

        ### What do the deciles mean?

        - **1** = Lowest 10% of estimated mental health need.
        - **10** = Highest 10% of estimated mental health need.

        **Note:** SAMHI measures **relative estimated need**, not diagnosed
        mental illness.
        """)

    init_session_state()

    if st.session_state.samhi_explorer_analysis_mode == "Change Between Years":
        st.session_state.samhi_explorer_playing = False

    if st.session_state.samhi_explorer_reset_requested:
        st.session_state.samhi_explorer_playing = False
        st.session_state.samhi_explorer_year = min(SAMHI_YEARS)
        st.session_state.samhi_explorer_last_tick = 0.0
        st.session_state.samhi_explorer_reset_requested = False

    if st.session_state.samhi_explorer_restart_requested:
        st.session_state.samhi_explorer_year = min(SAMHI_YEARS)
        st.session_state.samhi_explorer_restart_requested = False

    if st.session_state.samhi_explorer_playing:
        now = time.time()
        last_tick = float(st.session_state.samhi_explorer_last_tick)
        interval = float(st.session_state.samhi_explorer_interval)
        if (now - last_tick) >= interval:
            if st.session_state.samhi_explorer_year < max(SAMHI_YEARS):
                st.session_state.samhi_explorer_year += 1
                st.session_state.samhi_explorer_last_tick = now
            else:
                st.session_state.samhi_explorer_playing = False

    base_dir = resolve_base_dir(Path(__file__))
    try:
        bundle = get_prepared_bundle_cached(str(base_dir))
    except Exception as exc:
        st.error("Failed to load SAMHI data.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("SAMHI Controls")
        st.radio(
            "Analysis Mode",
            options=["Single Year", "Change Between Years"],
            horizontal=True,
            key="samhi_explorer_analysis_mode",
        )
        analysis_mode = str(st.session_state.samhi_explorer_analysis_mode)

        st.slider(
            "Year",
            min_value=min(SAMHI_YEARS),
            max_value=max(SAMHI_YEARS),
            step=1,
            key="samhi_explorer_year",
            disabled=(analysis_mode == "Change Between Years"),
        )
        st.radio(
            "Display Mode",
            options=["Index", "Decile"],
            horizontal=True,
            key="samhi_explorer_mode",
            help="""
        Index: Standardised score (0 = England average).

        Decile: National ranking from 1 (lowest estimated need) to 10 (highest estimated need).
        """,
        )

        if analysis_mode == "Change Between Years":
            st.slider(
                "From Year",
                min_value=min(SAMHI_YEARS),
                max_value=max(SAMHI_YEARS),
                step=1,
                key="samhi_explorer_change_from_year",
            )
            st.slider(
                "To Year",
                min_value=min(SAMHI_YEARS),
                max_value=max(SAMHI_YEARS),
                step=1,
                key="samhi_explorer_change_to_year",
            )

        c1, c2, c3 = st.columns(3)
        if c1.button("Play", use_container_width=True):
            if analysis_mode == "Single Year":
                if int(st.session_state.samhi_explorer_year) >= max(SAMHI_YEARS):
                    st.session_state.samhi_explorer_restart_requested = True
                st.session_state.samhi_explorer_playing = True
                st.session_state.samhi_explorer_last_tick = time.time()
                st.rerun()
        if c2.button("Pause", use_container_width=True):
            st.session_state.samhi_explorer_playing = False
        if c3.button("Reset", use_container_width=True):
            st.session_state.samhi_explorer_reset_requested = True
            st.rerun()

        st.select_slider(
            "Playback Speed",
            options=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
            key="samhi_explorer_interval",
            format_func=lambda x: f"{x:.1f}s per year",
            disabled=(analysis_mode == "Change Between Years"),
        )

        st.sidebar.divider()

        st.sidebar.caption(
            """
            © 2026 [University of Lincoln](https://www.lincoln.ac.uk/)

            [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/)

            Lincolnshire Mental Health Need Index v1.0
            """
        )

    analysis_mode = str(st.session_state.samhi_explorer_analysis_mode)
    year = int(st.session_state.samhi_explorer_year)
    mode = str(st.session_state.samhi_explorer_mode)

    display_df = bundle["lsoa_metrics"].copy()

    if analysis_mode == "Single Year":
        samhi_index_col, samhi_dec_col = get_samhi_columns(year)
        metric_col = samhi_index_col if mode == "Index" else samhi_dec_col

        if metric_col not in bundle["lsoa_metrics"].columns:
            st.error(f"Column not found for selected year/mode: {metric_col}")
            st.stop()

        display_df["SAMHI_Value"] = pd.to_numeric(display_df[metric_col], errors="coerce")
        value_col = "SAMHI_Value"
        map_value_col = value_col
        map_label = f"SAMHI {mode} ({year})"
        map_scale = "YlGnBu_09" if mode == "Index" else "YlOrRd_09"
        value_title = f"SAMHI {mode} ({year})"
        sort_ascending = False
    else:
        from_year = int(st.session_state.samhi_explorer_change_from_year)
        to_year = int(st.session_state.samhi_explorer_change_to_year)
        from_index_col, from_dec_col = get_samhi_columns(from_year)
        to_index_col, to_dec_col = get_samhi_columns(to_year)
        from_col = from_index_col if mode == "Index" else from_dec_col
        to_col = to_index_col if mode == "Index" else to_dec_col

        missing_cols = [c for c in [from_col, to_col] if c not in bundle["lsoa_metrics"].columns]
        if missing_cols:
            st.error(f"Missing SAMHI columns for change view: {', '.join(missing_cols)}")
            st.stop()

        display_df["SAMHI_From"] = pd.to_numeric(display_df[from_col], errors="coerce")
        display_df["SAMHI_To"] = pd.to_numeric(display_df[to_col], errors="coerce")
        display_df["SAMHI_Change"] = display_df["SAMHI_To"] - display_df["SAMHI_From"]
        # For mapping only, invert sign so decreases (improvement) render in blue and increases in red.
        display_df["SAMHI_Change_Map"] = -display_df["SAMHI_Change"]
        value_col = "SAMHI_Change"
        map_value_col = "SAMHI_Change_Map"
        map_label = f"SAMHI {mode} Change ({to_year} - {from_year})"
        map_scale = "RdYlBu_11"
        value_title = f"SAMHI {mode} Change ({to_year} - {from_year})"
        sort_ascending = False

    display_df = display_df.dropna(subset=[value_col, map_value_col]).copy()
    if display_df.empty:
        st.warning("No SAMHI values available for the selected view.")
        st.stop()

    metric_layers = [
        {
            "key": map_value_col,
            "label": map_label,
            "value_col": map_value_col,
            "scale": map_scale,
            "default": True,
        }
    ]
    if analysis_mode == "Single Year":
        tooltip_fields = ["LSOA_CODE", "LSOA21NM", value_col]
        tooltip_aliases = ["LSOA:", "LSOA Name:", f"SAMHI {mode} ({year}):"]
    else:
        tooltip_fields = ["LSOA_CODE", "LSOA21NM", "SAMHI_From", "SAMHI_To", value_col]
        tooltip_aliases = [
            "LSOA:",
            "LSOA Name:",
            f"SAMHI {mode} ({from_year}):",
            f"SAMHI {mode} ({to_year}):",
            f"Change ({to_year} - {from_year}):",
        ]
        st.caption("Change map colors: blue = lower estimated need than baseline year, red = higher estimated need.")

    fmap = build_choropleth_map(
        display_df,
        metric_layers=metric_layers,
        tooltip_fields=tooltip_fields,
        tooltip_aliases=tooltip_aliases,
        show_gps=False,
    )
    st_folium(
        fmap,
        key="samhi_map",
        height=700,
        width=None,
        returned_objects=[],
        render=False,
        use_container_width=True,
    )

    series = display_df[value_col].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min", f"{series.min():.4f}")
    c2.metric("Max", f"{series.max():.4f}")
    c3.metric("Mean", f"{series.mean():.4f}")
    c4.metric("Median", f"{series.median():.4f}")

    if analysis_mode == "Single Year":
        table_df = display_df[["LSOA_CODE", "LSOA21NM", value_col]].copy()
        table_df = table_df.rename(
            columns={
                "LSOA_CODE": "LSOA Code",
                "LSOA21NM": "LSOA Name",
                value_col: value_title,
            }
        )
        table_df = table_df.sort_values(value_title, ascending=sort_ascending)
    else:
        table_df = display_df[["LSOA_CODE", "LSOA21NM", "SAMHI_From", "SAMHI_To", value_col]].copy()
        table_df = table_df.rename(
            columns={
                "LSOA_CODE": "LSOA Code",
                "LSOA21NM": "LSOA Name",
                "SAMHI_From": f"SAMHI {mode} ({from_year})",
                "SAMHI_To": f"SAMHI {mode} ({to_year})",
                value_col: value_title,
            }
        )
        table_df = table_df.sort_values(value_title, ascending=sort_ascending)

    search_text = st.text_input("Search rows", placeholder="Type LSOA code or value...")
    if search_text:
        mask = table_df.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False))
        table_view = table_df[mask.any(axis=1)].copy()
    else:
        table_view = table_df.copy()

    page_size = st.select_slider(
        "Rows per page",
        options=[25, 50, 100, 200, 500],
        value=100,
        help="Smaller pages reduce websocket load and prevent flicker.",
    )
    total_rows = len(table_view)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (int(page_num) - 1) * page_size
    end_idx = start_idx + page_size

    st.markdown(table_view.iloc[start_idx:end_idx].to_html(index=False, escape=True), unsafe_allow_html=True)

    csv_data = table_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current table as CSV",
        data=csv_data,
        file_name=(
            f"lincolnshire_samhi_{mode.lower()}_{year}.csv"
            if analysis_mode == "Single Year"
            else f"lincolnshire_samhi_{mode.lower()}_change_{to_year}_minus_{from_year}.csv"
        ),
        mime="text/csv",
    )

    if st.session_state.samhi_explorer_playing and analysis_mode == "Single Year":
        time.sleep(0.1)
        st.rerun()


render()
