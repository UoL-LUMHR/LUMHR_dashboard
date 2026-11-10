from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.need_index import WEIGHT_KEYS, apply_need_score, normalize_weights, rebalance_weight_points
from analysis.samhi import SAMHI_YEARS, get_samhi_columns
from data.comparison_loader import get_comparison_bundle_cached, resolve_base_dir

st.set_page_config(
    page_title="Small Area Mental Health Index (SAMHI)",
    page_icon=str(Path(__file__).resolve().parents[1] / "assets" / "favicon.png"),
    layout="wide"
)

def init_session_state() -> None:
    defaults = {
        "dep_weight": 25.0,
        "smi_weight": 25.0,
        "prescribing_weight": 25.0,
        "samhi_weight": 25.0,
        "_weight_rebalancing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render() -> None:
    st.title("Comparison")

    base_dir = resolve_base_dir(Path(__file__))
    try:
        bundle = get_comparison_bundle_cached(str(base_dir))
    except Exception as exc:
        st.error("Failed to load comparison data.")
        st.exception(exc)
        st.stop()

    init_session_state()

    with st.sidebar:
        st.header("Comparison Controls")
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
        year = st.slider("SAMHI Year", min_value=min(SAMHI_YEARS), max_value=max(SAMHI_YEARS), value=2022, step=1)
        mode = st.radio("Mode", options=["Need vs SAMHI", "Difference"], index=0)        

        st.sidebar.divider()

        st.sidebar.caption(
            """
            © 2026 [University of Lincoln](https://www.lincoln.ac.uk/)

            [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/)

            Lincolnshire Mental Health Need Index v1.0
            """
        )

    dep_weight = float(st.session_state.dep_weight)
    smi_weight = float(st.session_state.smi_weight)
    prescribing_weight = float(st.session_state.prescribing_weight)
    samhi_weight = float(st.session_state.samhi_weight)

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

    samhi_index_col, _ = get_samhi_columns(year)
    scored = apply_need_score(
        bundle["lsoa_metrics"],
        dep_weight,
        smi_weight,
        prescribing_weight,
        samhi_weight,
        samhi_index_col,
    )
    scored = scored.dropna(subset=["Need_Score", "SAMHI_Selected"]).copy()
    if scored.empty:
        st.warning("No overlapping Need Index and SAMHI values for the selected year.")
        st.stop()

    corr = float(scored["Need_Score"].corr(scored["SAMHI_Selected"], method="pearson"))
    st.metric("Pearson Correlation (Need vs SAMHI)", f"{corr:.4f}")

    if mode == "Need vs SAMHI":
        st.info("Map rendering is disabled on this page for runtime stability. Use the scatter and ranking table for comparison.")

        rank_df = scored[["LSOA_CODE", "Need_Score", "SAMHI_Selected"]].copy()
        rank_df["Need_Rank"] = rank_df["Need_Score"].rank(ascending=False, method="dense")
        rank_df["SAMHI_Rank"] = rank_df["SAMHI_Selected"].rank(ascending=False, method="dense")
        rank_df["Rank_Delta"] = rank_df["Need_Rank"] - rank_df["SAMHI_Rank"]
        st.subheader("Top Rank Mismatches")
        st.dataframe(
            rank_df.sort_values("Rank_Delta", key=lambda s: s.abs(), ascending=False)
            .head(25)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        scored["Need_minus_SAMHI"] = scored["Need_Score"] - scored["SAMHI_Normalized"]
        st.info("Map rendering is disabled on this page for runtime stability. Showing largest differences as a table.")

        st.subheader("Largest Positive/Negative Differences")
        st.dataframe(
            scored[["LSOA_CODE", "Need_Score", "SAMHI_Normalized", "Need_minus_SAMHI"]]
            .sort_values("Need_minus_SAMHI", key=lambda s: s.abs(), ascending=False)
            .head(25)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Need vs SAMHI Scatter")
    scatter_df = pd.DataFrame({"Need Index": scored["Need_Score"], f"SAMHI Index ({year})": scored["SAMHI_Selected"]})
    st.scatter_chart(scatter_df, x="Need Index", y=f"SAMHI Index ({year})")


render()
