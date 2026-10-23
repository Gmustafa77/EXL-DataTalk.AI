"""Data Explorer page: column picker, quick filters, 200-row preview and CSV export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.ui import page_title, section

PREVIEW_ROWS = 200


def render_data_explorer(current_df: pd.DataFrame) -> None:
    page_title("🗄️", "Healthcare Appeals Case Explorer",
               f"{len(current_df):,} rows · {len(current_df.columns)} columns · filter, preview and export CSV")
    if current_df.empty:
        st.info("No rows are available in the current dashboard population.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(current_df):,}", border=True)
    c2.metric("Columns", f"{len(current_df.columns)}", border=True)
    c3.metric("Units", f"{current_df['Current Unit Name'].nunique()}", border=True)
    c4.metric("Sponsors", f"{current_df['Plan Sponsor Name'].nunique()}", border=True)
    dates = current_df["Received Date"].dropna()
    c5.metric("Date Range", f"{dates.min():%b %Y} → {dates.max():%b %Y}" if not dates.empty else "—", border=True)

    with st.expander("📥 Export configuration", expanded=False):
        st.caption("Select the columns to include in the download.")
        all_cols = current_df.columns.tolist()
        selected = st.multiselect("Columns", options=all_cols, default=all_cols, label_visibility="collapsed")
        if selected:
            st.download_button("📥 Download CSV", data=current_df[selected].to_csv(index=False).encode("utf-8"),
                               file_name="EXL_Healthcare_Appeals_Analytics.csv", mime="text/csv",
                               width="stretch", type="primary")
        else:
            st.warning("Select at least one column.")

    section("🔍 Quick row filters")
    f1, f2, f3, f4 = st.columns(4)
    filters = {
        "Current Unit Name": f1.multiselect("Unit", sorted(current_df["Current Unit Name"].dropna().unique()), key="de_dom"),
        "Routed To Site Name": f2.multiselect("Site", sorted(current_df["Routed To Site Name"].dropna().unique()), key="de_rgn"),
        "Work Item Status Short Name": f3.multiselect(
            "Status", sorted(current_df["Work Item Status Short Name"].dropna().unique()), key="de_ms"),
        "ComplianceStatus": f4.multiselect("Compliance", sorted(current_df["ComplianceStatus"].dropna().unique()),
                                           key="de_app"),
    }
    view = current_df
    for column, values in filters.items():
        if values:
            view = view[view[column].isin(values)]

    st.caption(f"Showing {min(len(view), PREVIEW_ROWS):,} of {len(view):,} matching rows · use the export above "
               "for the full dataset.")
    st.dataframe(
        view.head(PREVIEW_ROWS), width="stretch", height=640, hide_index=True,
        column_config={
            "CaseAgeDays": st.column_config.NumberColumn("Case Age", format="%d d"),
            "ComplianceRiskScore": st.column_config.ProgressColumn("Risk Score", format="%.3f", min_value=0, max_value=1),
            "IsEscalated": st.column_config.CheckboxColumn("Escalated"),
            "IsOverdue": st.column_config.CheckboxColumn("Overdue"),
            "IsClosed": st.column_config.CheckboxColumn("Closed"),
        },
    )
