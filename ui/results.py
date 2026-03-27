# ui/results.py
"""Results display components."""

import io

import pandas as pd
import streamlit as st

from config.conversion_config import CONVERSION_CONFIG, is_in_drug_like_range


def _style_results(df: pd.DataFrame) -> pd.DataFrame.style:
    """Apply background color to numeric cells based on drug-like range."""

    # Map column label -> log_name
    col_to_logname = {}
    for log_name, cfg in CONVERSION_CONFIG.items():
        display = cfg["display_name"]
        unit = cfg["unit"]
        col_label = f"{display} ({unit})" if unit else display
        col_to_logname[col_label] = log_name

    def _cell_color(val, log_name):
        if pd.isna(val):
            return ""
        result = is_in_drug_like_range(float(val), log_name)
        if result is True:
            return "background-color: #d4edda; color: #155724"  # green
        if result is False:
            return "background-color: #f8d7da; color: #721c24"  # red
        return ""

    style = df.style.format(
        {col: "{:.3f}" for col in df.select_dtypes(include="number").columns},
        na_rep="—",
    )
    for col, log_name in col_to_logname.items():
        if col in df.columns:
            style = style.applymap(_cell_color, log_name=log_name, subset=[col])

    return style


def display_results_table(results_df: pd.DataFrame):
    """Display results with color coding, summary stats, and reference guide."""
    st.subheader("Prediction Results")

    numeric_cols = results_df.select_dtypes(include=["float64", "float32"]).columns.tolist()

    # --- Metric highlights row ---
    if numeric_cols:
        n_cols = min(len(numeric_cols), 4)
        cols = st.columns(n_cols)
        for i, col in enumerate(numeric_cols):
            values = results_df[col].dropna()
            if len(values) > 0:
                with cols[i % n_cols]:
                    mean_val = values.mean()
                    std_val = values.std() if len(values) > 1 else 0.0
                    st.metric(
                        label=col,
                        value=f"{mean_val:.3f}",
                        delta=f"±{std_val:.3f} SD" if len(values) > 1 else None,
                        help=f"Mean across {len(values)} molecule(s)",
                    )

    st.divider()

    # --- Summary statistics ---
    with st.expander("Summary Statistics", expanded=False):
        if numeric_cols:
            summary_rows = []
            for col in numeric_cols:
                values = results_df[col].dropna()
                if len(values) > 0:
                    # Find log_name for this column label to check range
                    log_name = None
                    for ln, cfg in CONVERSION_CONFIG.items():
                        display = cfg["display_name"]
                        unit = cfg["unit"]
                        label = f"{display} ({unit})" if unit else display
                        if label == col:
                            log_name = ln
                            break
                    in_range = (
                        sum(1 for v in values if is_in_drug_like_range(float(v), log_name) is True)
                        if log_name
                        else None
                    )
                    pct = f"{100 * in_range / len(values):.0f}%" if in_range is not None else "—"
                    summary_rows.append(
                        {
                            "Property": col,
                            "Mean": round(values.mean(), 4),
                            "Median": round(values.median(), 4),
                            "Std": round(values.std(), 4) if len(values) > 1 else 0.0,
                            "Min": round(values.min(), 4),
                            "Max": round(values.max(), 4),
                            "In Drug-Like Range": pct,
                        }
                    )
            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows).set_index("Property"),
                    use_container_width=True,
                )

    # --- Full results table with color coding ---
    st.markdown("**Full Results**  — green = drug-like range, red = outside range")
    st.dataframe(_style_results(results_df), use_container_width=True, height=400)

    # --- Reference ranges guide ---
    with st.expander("Reference Ranges & Interpretation Guide", expanded=False):
        rows = []
        for log_name, cfg in CONVERSION_CONFIG.items():
            rng = cfg.get("reference_range", {})
            low = rng.get("low")
            high = rng.get("high")
            if low is not None and high is not None:
                range_str = f"{low} – {high}"
            elif low is not None:
                range_str = f"> {low}"
            elif high is not None:
                range_str = f"< {high}"
            else:
                range_str = "—"

            unit = cfg["unit"]
            rows.append(
                {
                    "Property": cfg["display_name"],
                    "Unit": unit if unit else "—",
                    "Drug-Like Range": range_str,
                    "Preferred": rng.get("preferred_direction", "—").capitalize(),
                    "Note": rng.get("note", "—"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("Property"),
            use_container_width=True,
            hide_index=False,
        )


def display_property_info():
    """Display full property descriptions and reference ranges."""
    rows = []
    for log_name, config in CONVERSION_CONFIG.items():
        rng = config.get("reference_range", {})
        low = rng.get("low")
        high = rng.get("high")
        range_str = (
            f"{low} – {high}"
            if (low is not None and high is not None)
            else (f"> {low}" if low is not None else (f"< {high}" if high is not None else "—"))
        )
        rows.append(
            {
                "Property": config["display_name"],
                "Unit": config["unit"] if config["unit"] else "—",
                "Scale": "Log" if config["log_scale"] else "Linear",
                "Drug-Like Range": range_str,
                "Description": config["description"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).set_index("Property"),
        use_container_width=True,
        hide_index=False,
    )


def create_download_button(results_df: pd.DataFrame, filename: str = "predictions.csv"):
    """Create CSV and Excel download buttons side by side."""
    col1, col2 = st.columns(2)

    with col1:
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button(
            label="Download Excel",
            data=xlsx_buffer.getvalue(),
            file_name=filename.replace(".csv", ".xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
