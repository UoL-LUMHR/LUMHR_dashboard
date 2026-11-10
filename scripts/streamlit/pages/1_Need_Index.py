from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from analysis.need_index import (
    WEIGHT_KEYS,
    apply_need_score,
    normalize_weights,
    rebalance_weight_points,
)

from analysis.samhi import SAMHI_YEARS, get_samhi_columns
from data.loader import get_prepared_bundle_cached, resolve_base_dir
from maps.folium_map import build_choropleth_map

st.set_page_config(
    page_title="Lincolnshire Mental Health Need Index",
    page_icon=str(Path(__file__).resolve().parents[1] / "assets" / "favicon.png"),
    layout="wide"
)

def init_session_state() -> None:
    defaults = {
        "dep_weight": 25.0,
        "smi_weight": 25.0,
        "prescribing_weight": 25.0,
        "samhi_weight": 25.0,
        "samhi_year": 2022,
        "show_gps": True,
        "filter_enabled": False,
        "filter_metric": "Need_Score",
        "filter_percentile": 80,
        "_weight_rebalancing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render() -> None:
    st.title("Lincolnshire Mental Health Need Index")

    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] table {
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        iframe {
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Method Summary", expanded=False):
        st.markdown(
            """
            This Index proportionally allocates GP-level QOF registers to LSOAs using the GP patient registration data.

            Allocation equations:
            - GP total mapped patients: sum of NUMBER_OF_PATIENTS across all mapped LSOAs for each GP.
            - Weight(i,j): NUMBER_OF_PATIENTS(i,j) / GP_Total_Mapped_Patients(j)
            - Allocated register(i,j): GP_Register(j) * Weight(i,j)
            - LSOA register(i): sum of allocated registers across all GPs.
            - LSOA prevalence(i): LSOA_Register(i) / LSOA_Total_List(i)

            Need Index:
            - Depression prevalence, SMI prevalence, antidepressant items per patient, and SAMHI are min-max normalized to [0,1].
            - Need Index = (Depression_Normalized * w_dep) + (SMI_Normalized * w_smi)
              + (Prescribing_Normalized * w_rx) + (SAMHI_Normalized * w_samhi)

            Data sources:
            - [GP Patient Registration](https://digital.nhs.uk/data-and-information/publications/statistical/patients-registered-at-a-gp-practice/july-2026)
            - [QOF Depression and SMI prevalence](https://digital.nhs.uk/data-and-information/publications/statistical/quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data/2024-25)
            - [Antidepressant items per patient](https://openprescribing.net/analyse/#org=practice&orgIds=71E&numIds=4.3&denom=nothing&selectedTab=map)
            - [SAMHI](https://pldr.org/dataset/small-area-mental-health-index-samhi-2noyv)
            """
        )

    base_dir = resolve_base_dir(Path(__file__))
    try:
        bundle = get_prepared_bundle_cached(str(base_dir))
    except Exception as exc:
        st.error("Failed to load or prepare data. Check file paths, column names, and data quality.")
        st.exception(exc)
        st.stop()

    init_session_state()

    filter_options = [
        "Need_Score",
        "Depression_Prevalence",
        "SMI_Prevalence",
        "Antidepressant_Items_Per_Patient",
        "SAMHI_Selected",
    ]

    with st.sidebar:
        st.header("Need Index Controls")
        st.caption("Weight points always sum to 100.")

        st.slider(
            "Depression Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="dep_weight",
            on_change=rebalance_weight_points,
            args=("dep_weight", WEIGHT_KEYS),
            help="Weight points (0-100). Final contribution is normalized with the other three controls.",
        )
        st.slider(
            "SMI Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="smi_weight",
            on_change=rebalance_weight_points,
            args=("smi_weight", WEIGHT_KEYS),
            help="Weight points (0-100). Final contribution is normalized with the other three controls.",
        )
        st.slider(
            "Prescribing Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="prescribing_weight",
            on_change=rebalance_weight_points,
            args=("prescribing_weight", WEIGHT_KEYS),
            help="Weight points (0-100). Final contribution is normalized with the other three controls.",
        )
        st.slider(
            "SAMHI Weight Points",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="samhi_weight",
            on_change=rebalance_weight_points,
            args=("samhi_weight", WEIGHT_KEYS),
            help="Weight points (0-100). Final contribution is normalized with the other three controls.",
        )
        st.slider(
            "SAMHI Year",
            min_value=min(SAMHI_YEARS),
            max_value=max(SAMHI_YEARS),
            step=1,
            key="samhi_year",
        )

        st.header("Map Layers")
        st.toggle("Show GP Locations Overlay", key="show_gps")
        st.caption("Choropleth metric is selected in the map key using radio buttons.")

        st.header("Map Filter")
        st.toggle("Filter to Areas Above Percentile", key="filter_enabled")
        st.selectbox(
            "Filter Metric",
            options=filter_options,
            key="filter_metric",
            format_func=lambda x: {
                "Need_Score": "Need Index",
                "Depression_Prevalence": "Depression Prevalence",
                "SMI_Prevalence": "SMI Prevalence",
                "Antidepressant_Items_Per_Patient": "Antidepressant Items Per Patient",
                "SAMHI_Selected": "SAMHI Index",
            }[x],
        )
        st.slider(
            "Percentile Threshold",
            min_value=50,
            max_value=99,
            step=1,
            key="filter_percentile",
        )

    dep_weight = float(st.session_state.dep_weight)
    smi_weight = float(st.session_state.smi_weight)
    prescribing_weight = float(st.session_state.prescribing_weight)
    samhi_weight = float(st.session_state.samhi_weight)
    samhi_year = int(st.session_state.samhi_year)
    show_gps = bool(st.session_state.show_gps)
    filter_enabled = bool(st.session_state.filter_enabled)
    filter_metric = str(st.session_state.filter_metric)
    filter_percentile = int(st.session_state.filter_percentile)

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    scored_lsoa = apply_need_score(
        bundle["lsoa_metrics"],
        dep_weight,
        smi_weight,
        prescribing_weight,
        samhi_weight,
        samhi_index_col,
    )
    scored_lsoa["Depression_Prevalence_Pct"] = (scored_lsoa["Depression_Prevalence"] * 100).round(2)
    scored_lsoa["SMI_Prevalence_Pct"] = (scored_lsoa["SMI_Prevalence"] * 100).round(2)

    # Merge the live score columns onto the cached, map-ready geometry base.
    map_lsoa = bundle["lsoa_map_base"].merge(
        scored_lsoa[
            [
                "LSOA_CODE",
                "Need_Score",
                "Depression_Prevalence",
                "Depression_Prevalence_Pct",
                "SMI_Prevalence",
                "SMI_Prevalence_Pct",
                "Antidepressant_Items_Per_Patient",
                "SAMHI_Selected",
            ]
        ],
        on="LSOA_CODE",
        how="left",
    )

    normalized_weights = normalize_weights(
        {
            "dep_weight": dep_weight,
            "smi_weight": smi_weight,
            "prescribing_weight": prescribing_weight,
            "samhi_weight": samhi_weight,
        }
    )
    raw_total = dep_weight + smi_weight + prescribing_weight + samhi_weight
    st.sidebar.caption(f"Weight points total: {raw_total:.1f}")
    st.sidebar.caption(
        "Effective Need Index weights (always normalized to 100%): "
        f"Depression {normalized_weights['dep_weight'] * 100:.1f}% | "
        f"SMI {normalized_weights['smi_weight'] * 100:.1f}% | "
        f"Prescribing {normalized_weights['prescribing_weight'] * 100:.1f}% | "
        f"SAMHI {normalized_weights['samhi_weight'] * 100:.1f}%"
    )

    st.sidebar.divider()
    st.sidebar.caption(
        """
        © 2026 [University of Lincoln](https://www.lincoln.ac.uk/)

        [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/)

        Lincolnshire Mental Health Need Index v1.0
        """
    )

    display_lsoa = scored_lsoa.copy()
    threshold_value = None
    if filter_enabled:
        metric_series = display_lsoa[filter_metric].dropna()
        if not metric_series.empty:
            threshold_value = float(np.nanpercentile(metric_series, filter_percentile))
            display_lsoa = display_lsoa[display_lsoa[filter_metric] >= threshold_value].copy()

    # Keep the map data aligned with the active filter without recomputing geometry.
    map_display_lsoa = map_lsoa
    if filter_enabled and threshold_value is not None:
        map_display_lsoa = map_display_lsoa[map_display_lsoa[filter_metric] >= threshold_value].copy()

    active_view = st.radio("View", options=["Interactive Map", "Data Explorer"], horizontal=True, index=0)

    if active_view == "Interactive Map":
        if display_lsoa.empty:
            st.warning("No LSOAs match the selected filter. Relax the percentile threshold or disable filtering.")
        else:
            metric_layers = [
                {"key": "Need_Score", "label": "Need Index", "value_col": "Need_Score", "scale": "YlOrRd_09", "default": True},
                {
                    "key": "Depression_Prevalence",
                    "label": "Depression Prevalence",
                    "value_col": "Depression_Prevalence",
                    "scale": "YlOrRd_09",
                },
                {"key": "SMI_Prevalence", "label": "SMI Prevalence", "value_col": "SMI_Prevalence", "scale": "YlOrRd_09"},
                {
                    "key": "Antidepressant_Items_Per_Patient",
                    "label": "Antidepressant Items Per Patient",
                    "value_col": "Antidepressant_Items_Per_Patient",
                    "scale": "YlOrRd_09",
                },
                {
                    "key": "SAMHI_Selected",
                    "label": f"SAMHI Index ({samhi_year})",
                    "value_col": "SAMHI_Selected",
                    "scale": "YlGnBu_09",
                },
            ]
            tooltip_fields = [
                "LSOA_CODE",
                "Need_Score",
                "Depression_Prevalence_Pct",
                "SMI_Prevalence_Pct",
                "Antidepressant_Items_Per_Patient",
                "SAMHI_Selected",
            ]
            tooltip_aliases = [
                "LSOA:",
                "Need Index:",
                "Depression Prevalence (%):",
                "SMI Prevalence (%):",
                "Antidepressant Items Per Patient:",
                f"SAMHI Index ({samhi_year}):",
            ]
            weight_lines = [
                f"Depression: {normalized_weights['dep_weight']:.2f}",
                f"SMI: {normalized_weights['smi_weight']:.2f}",
                f"Prescribing: {normalized_weights['prescribing_weight']:.2f}",
                f"SAMHI: {normalized_weights['samhi_weight']:.2f}",
            ]

            map_key = (
                dep_weight,
                smi_weight,
                prescribing_weight,
                samhi_weight,
                samhi_year,
                show_gps,
                filter_enabled,
                filter_metric,
                filter_percentile,
            )

            if st.session_state.get("need_map_key") != map_key:
                # The Folium object is rebuilt only when the actual map inputs change.
                # Zoom/pan reruns reuse the same cached map object to reduce flicker.
                st.session_state.need_map_obj = build_choropleth_map(
                    map_display_lsoa,
                    metric_layers=metric_layers,
                    tooltip_fields=tooltip_fields,
                    tooltip_aliases=tooltip_aliases,
                    show_gps=show_gps,
                    gp_marker_df=bundle["gp_marker_df"],
                    weight_legend_lines=weight_lines,
                )
                st.session_state.need_map_key = map_key

            fmap = st.session_state.need_map_obj
            st_folium(
                fmap,
                key="need_map",
                height=700,
                returned_objects=[],
                render=False,
                use_container_width=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LSOAs Displayed", f"{display_lsoa['LSOA_CODE'].nunique():,}")
        c2.metric("Mapped GP Practices", f"{bundle['mapping_df']['PRACTICE_CODE'].nunique():,}")
        c3.metric(
            "GPs with map coordinates",
            f"{bundle['gp_marker_df'].dropna(subset=['Lat', 'Lon'])['PRACTICE_CODE'].nunique():,}",
        )
        c4.metric("Applied Threshold", f"{threshold_value:.4f}" if threshold_value is not None else "None")

        practices_in_mapping = set(bundle["mapping_df"]["PRACTICE_CODE"].unique())
        practices_in_qof = set(bundle["gp_master"]["PRACTICE_CODE"].dropna().unique())
        practices_in_gp_locations = set(bundle["gp_loc_df"]["PRACTICE_CODE"].dropna().unique())
        qof_without_mapping = len(practices_in_qof - practices_in_mapping)
        gp_locations_without_qof = len(practices_in_gp_locations - practices_in_qof)
        mismatch_summary = bundle["mismatch_summary"]

        if mismatch_summary["mapped_practices_without_any_qof"] > 0:
            st.warning(
                "There are mapped GP practices with no matching QOF records. "
                f"Count: {mismatch_summary['mapped_practices_without_any_qof']}. "
                "Those practices contribute to LSOA list size but not allocated register counts."
            )

        if mismatch_summary["mapped_practices_without_dep_qof"] > 0 or mismatch_summary["mapped_practices_without_smi_qof"] > 0:
            st.info(
                "QOF linkage details: "
                f"missing depression register for {mismatch_summary['mapped_practices_without_dep_qof']} mapped practices, "
                f"missing SMI register for {mismatch_summary['mapped_practices_without_smi_qof']} mapped practices."
            )

        if mismatch_summary["mapped_practices_without_prescribing"] > 0:
            st.info(
                "Prescribing linkage details: "
                f"missing antidepressant prescribing records for {mismatch_summary['mapped_practices_without_prescribing']} mapped practices."
            )

        if qof_without_mapping > 0:
            st.info(f"There are {qof_without_mapping} QOF practices with no mapping rows in the GP registration data.")

        if gp_locations_without_qof > 0:
            st.info(f"There are {gp_locations_without_qof} GP location records without matching QOF records.")

        if len(bundle["out_of_area_lsoa_codes"]) > 0:
            st.info(
                "The GP registration data includes out-of-Lincolnshire LSOAs that are not in the boundary file. "
                f"Distinct out-of-area LSOAs: {len(bundle['out_of_area_lsoa_codes']):,}. "
                f"Patient records in those LSOAs: {bundle['out_of_area_patients']:,.0f}. "
                "They are retained in GP allocation weights but excluded from Lincolnshire map polygons."
            )

    if active_view == "Data Explorer":
        explorer_cols = [
            "LSOA_CODE",
            "LSOA21NM",
            "LSOA_Total_List",
            "Allocated_Depression",
            "Allocated_SMI",
            "Allocated_Antidepressant_Items",
            "Depression_Prevalence",
            "SMI_Prevalence",
            "Antidepressant_Items_Per_Patient",
            "SAMHI_Selected",
            "Need_Score",
        ]

        explorer = scored_lsoa[explorer_cols].copy()
        explorer = explorer.rename(
            columns={
                "LSOA_CODE": "LSOA Code",
                "LSOA21NM": "LSOA Name",
                "LSOA_Total_List": "List Size",
                "Allocated_Depression": "Mapped Depression Count",
                "Allocated_SMI": "Mapped SMI Count",
                "Allocated_Antidepressant_Items": "Mapped Antidepressant Items",
                "Depression_Prevalence": "Depression Prevalence",
                "SMI_Prevalence": "SMI Prevalence",
                "Antidepressant_Items_Per_Patient": "Antidepressant Items Per Patient",
                "SAMHI_Selected": f"SAMHI Index ({samhi_year})",
                "Need_Score": "Need Index",
            }
        )

        search_text = st.text_input("Search rows", placeholder="Type LSOA code or value...")
        if search_text:
            mask = explorer.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False))
            explorer_view = explorer[mask.any(axis=1)].copy()
        else:
            explorer_view = explorer.copy()

        explorer_view = explorer_view.sort_values("Need Index", ascending=False)

        page_size = st.select_slider(
            "Rows per page",
            options=[25, 50, 100, 200, 500],
            value=100,
            help="Smaller pages reduce websocket load and prevent flicker.",
        )
        total_rows = len(explorer_view)
        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (int(page_num) - 1) * page_size
        end_idx = start_idx + page_size
        explorer_preview = explorer_view.iloc[start_idx:end_idx].copy()

        st.markdown(explorer_preview.to_html(index=False, escape=True), unsafe_allow_html=True)

        if total_rows > page_size:
            st.info(
                f"Showing rows {start_idx + 1:,}-{min(end_idx, total_rows):,} of {total_rows:,}. "
                "Use the CSV download for the complete filtered dataset."
            )

        csv_data = explorer_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current table as CSV",
            data=csv_data,
            file_name="lincolnshire_lsoa_mental_health_need.csv",
            mime="text/csv",
        )


render()
