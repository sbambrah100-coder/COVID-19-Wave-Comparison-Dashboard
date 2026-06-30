import io
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh

try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        Prophet = None


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.copy()
    df["New_Cases"] = pd.to_numeric(df["New_Cases"], errors="coerce")
    df["Hospitalizations"] = pd.to_numeric(df["Hospitalizations"], errors="coerce")
    df["ICU_Admissions"] = pd.to_numeric(df["ICU_Admissions"], errors="coerce")
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    df["Positivity_Rate_Percent"] = pd.to_numeric(df["Positivity_Rate_Percent"], errors="coerce")
    df["Vaccination_Rate_Percent"] = pd.to_numeric(df["Vaccination_Rate_Percent"], errors="coerce")
    df.sort_values("Date", inplace=True)
    return df


STATE_COORDINATES = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.3562, 74.1237),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.2268, 77.5619),
    "Puducherry": (11.9416, 79.8083),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.1809, 73.0169),
    "Lakshadweep": (10.3280, 72.7844),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .glass-card {
            border-radius: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.12);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 28px 50px rgba(0,0,0,0.25);
        }
        .metric-icon {
            font-size: 1.4rem;
            margin-right: 0.8rem;
        }
        .stSidebar .css-1d391kg {
            padding-top: 1rem;
        }
        .footer {
            color: #aaa;
            font-size: 0.85rem;
            margin-top: 2rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_kpi_card(title: str, value: str, description: str, icon: str, delta: float | None = None) -> None:
    if delta is None:
        delta_text = ""
    else:
        delta_label = f"<span style='color: #50fa7b;'>+{delta:.1f}%</span>" if delta >= 0 else f"<span style='color: #ff5555;'>{delta:.1f}%</span>"
        delta_text = f"<div style='margin-top: 8px; font-size: 0.9rem;'>Trend: {delta_label}</div>"
    st.markdown(
        f"""
        <div class='glass-card'>
            <div style='display:flex; align-items:center; justify-content:space-between;'>
                <div>
                    <div style='font-size:0.85rem; letter-spacing:0.08em; opacity:0.8;'>{title}</div>
                    <div style='font-size:2.1rem; font-weight:700; margin-top:0.4rem;'>{value}</div>
                </div>
                <div style='font-size:2rem; opacity:0.9;'>{icon}</div>
            </div>
            <div style='margin-top:12px; color:#cbd5e1;'>{description}</div>
            {delta_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def download_df_as_csv(df: pd.DataFrame, filename: str) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(summary: dict, filename: str = "summary_report.pdf") -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "COVID-19 Analytics Summary Report", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "A snapshot of the current filtered dataset with quality metrics, forecasts, and model insights.")
    pdf.ln(4)
    for section, data in summary.items():
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, section, ln=True)
        pdf.set_font("Helvetica", size=10)
        for key, value in data.items():
            pdf.multi_cell(0, 7, f"- {key}: {value}")
        pdf.ln(2)
    return pdf.output(dest="S").encode("latin-1")


def compute_trend(current: float, previous: float) -> float:
    if previous == 0 or np.isnan(previous):
        return 0.0
    return ((current - previous) / previous) * 100.0


def build_state_map(df: pd.DataFrame) -> px.scatter_mapbox:
    state_agg = (
        df.groupby("State", as_index=False)
        .agg({"New_Cases": "sum", "Deaths": "sum"})
        .sort_values("New_Cases", ascending=False)
    )
    state_agg["lat"] = state_agg["State"].map(lambda x: STATE_COORDINATES.get(x, (np.nan, np.nan))[0])
    state_agg["lon"] = state_agg["State"].map(lambda x: STATE_COORDINATES.get(x, (np.nan, np.nan))[1])
    state_agg = state_agg.dropna(subset=["lat", "lon"])
    fig = px.scatter_mapbox(
        state_agg,
        lat="lat",
        lon="lon",
        size="New_Cases",
        color="Deaths",
        hover_name="State",
        hover_data={"New_Cases": True, "Deaths": True, "lat": False, "lon": False},
        color_continuous_scale="Reds",
        size_max=32,
        zoom=4.5,
        title="State-wise COVID-19 Impact",
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Deaths")
    return fig


def main() -> None:
    st.set_page_config(
        page_title="COVID-19 Wave Dashboard",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    st.markdown("# COVID-19 Wave Comparison Dashboard")
    st.markdown("Modern analytics for case trends, vaccinations, variants, and forecasting.")

    with st.sidebar:
        st.markdown("## Filters")
        data_path = "Covid19_Wave_Comparison_Dataset.csv"
        df = load_data(data_path)

        date_min = df["Date"].min()
        date_max = df["Date"].max()
        date_range = st.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)
        state_search = st.text_input("Search State", value="")
        selected_states = st.multiselect(
            "State",
            sorted(df["State"].unique()),
            default=sorted(df["State"].unique()),
        )
        selected_waves = st.multiselect(
            "COVID Wave",
            sorted(df["Wave"].unique()),
            default=sorted(df["Wave"].unique()),
        )
        selected_variants = st.multiselect(
            "Dominant Variant",
            sorted(df["Dominant_Variant"].unique()),
            default=sorted(df["Dominant_Variant"].unique()),
        )
        reset = st.button("Reset filters")
        st.markdown("---")
        st.markdown("## Display Options")
        auto_refresh = st.checkbox("Auto-refresh dashboard", value=False)
        refresh_rate = st.slider("Refresh interval (seconds)", min_value=15, max_value=300, value=60, step=15)
        if auto_refresh:
            st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

    if reset:
        selected_states = sorted(df["State"].unique())
        selected_waves = sorted(df["Wave"].unique())
        selected_variants = sorted(df["Dominant_Variant"].unique())
        state_search = ""
        date_range = [date_min, date_max]

    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = date_min, date_max

    filtered = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
        & (df["State"].isin(selected_states))
        & (df["Wave"].isin(selected_waves))
        & (df["Dominant_Variant"].isin(selected_variants))
    ]
    if state_search:
        filtered = filtered[filtered["State"].str.contains(state_search, case=False, na=False)]

    filtered = filtered.reset_index(drop=True)

    st.markdown("### Overview")
    total_cases = filtered["New_Cases"].sum()
    total_deaths = filtered["Deaths"].sum()
    total_hospitalizations = filtered["Hospitalizations"].sum()
    total_icu = filtered["ICU_Admissions"].sum()
    avg_positivity = filtered["Positivity_Rate_Percent"].mean()
    avg_vaccination = filtered["Vaccination_Rate_Percent"].mean()
    state_count = filtered["State"].nunique()
    record_count = len(filtered)

    previous_period = df[
        (df["Date"] < start_date)
        & (df["Date"] >= start_date - (end_date - start_date))
        & (df["State"].isin(selected_states))
        & (df["Wave"].isin(selected_waves))
        & (df["Dominant_Variant"].isin(selected_variants))
    ]
    prev_cases = previous_period["New_Cases"].sum()
    prev_deaths = previous_period["Deaths"].sum()
    prev_hosp = previous_period["Hospitalizations"].sum()
    prev_icu = previous_period["ICU_Admissions"].sum()
    prev_positivity = previous_period["Positivity_Rate_Percent"].mean()
    prev_vaccination = previous_period["Vaccination_Rate_Percent"].mean()

    cols = st.columns(4, gap="large")
    with cols[0]:
        create_kpi_card(
            "Total Cases",
            format_int(total_cases),
            "Cumulative cases in the selected range.",
            "📈",
            compute_trend(total_cases, prev_cases),
        )
    with cols[1]:
        create_kpi_card(
            "Total Deaths",
            format_int(total_deaths),
            "Reported deaths in the filtered dataset.",
            "⚰️",
            compute_trend(total_deaths, prev_deaths),
        )
    with cols[2]:
        create_kpi_card(
            "Hospitalizations",
            format_int(total_hospitalizations),
            "Total hospital admissions.",
            "🏥",
            compute_trend(total_hospitalizations, prev_hosp),
        )
    with cols[3]:
        create_kpi_card(
            "ICU Admissions",
            format_int(total_icu),
            "Serious cases requiring ICU care.",
            "🩺",
            compute_trend(total_icu, prev_icu),
        )

    cols2 = st.columns(4, gap="large")
    with cols2[0]:
        create_kpi_card(
            "Avg. Positivity Rate",
            format_pct(avg_positivity or 0.0),
            "Average test positivity.",
            "🔬",
            compute_trend(avg_positivity or 0.0, prev_positivity or 0.0),
        )
    with cols2[1]:
        create_kpi_card(
            "Avg. Vaccination Rate",
            format_pct(avg_vaccination or 0.0),
            "Average vaccine coverage.",
            "💉",
            compute_trend(avg_vaccination or 0.0, prev_vaccination or 0.0),
        )
    with cols2[2]:
        create_kpi_card(
            "States Selected",
            f"{state_count}",
            "Distinct states in scope.",
            "📍",
            None,
        )
    with cols2[3]:
        create_kpi_card(
            "Record Count",
            f"{record_count}",
            "Number of datapoints after filtering.",
            "📄",
            None,
        )

    st.markdown("---")
    st.markdown("## Trend Charts")
    trend_tabs = st.tabs([
        "Daily Cases", "Daily Deaths", "Hospitalizations", "ICU Admissions", "Vaccination", "Positivity"
    ])

    daily_data = filtered.groupby("Date", as_index=False).sum(numeric_only=True)
    if daily_data.empty:
        st.warning("No data is available for the selected filters.")
        return

    with trend_tabs[0]:
        fig = px.line(daily_data, x="Date", y="New_Cases", title="Daily Cases Trend", markers=True)
        fig.update_traces(line=dict(color="#5e81ac", width=3), hovertemplate="%{x}: %{y:,}")
        st.plotly_chart(fig, use_container_width=True)
    with trend_tabs[1]:
        fig = px.area(daily_data, x="Date", y="Deaths", title="Daily Death Trend", color_discrete_sequence=["#ef5350"])
        fig.update_traces(hovertemplate="%{x}: %{y:,}")
        st.plotly_chart(fig, use_container_width=True)
    with trend_tabs[2]:
        fig = px.line(daily_data, x="Date", y="Hospitalizations", title="Hospitalizations Trend", markers=True)
        fig.update_traces(line=dict(color="#f6c85f", width=3))
        st.plotly_chart(fig, use_container_width=True)
    with trend_tabs[3]:
        fig = px.line(daily_data, x="Date", y="ICU_Admissions", title="ICU Admissions Trend", markers=True)
        fig.update_traces(line=dict(color="#8fbcbb", width=3))
        st.plotly_chart(fig, use_container_width=True)
    with trend_tabs[4]:
        fig = px.line(daily_data, x="Date", y="Vaccination_Rate_Percent", title="Vaccination Progress", markers=True)
        fig.update_traces(line=dict(color="#48cae4", width=3))
        st.plotly_chart(fig, use_container_width=True)
    with trend_tabs[5]:
        fig = px.line(daily_data, x="Date", y="Positivity_Rate_Percent", title="Positivity Rate Trend", markers=True)
        fig.update_traces(line=dict(shape="spline", smoothing=1.3, color="#a78bfa", width=3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## State & Variant Distribution")
    distr_tabs = st.tabs([
        "Cases by State", "Deaths by State", "Vaccination by State", "Variant", "Wave"
    ])

    by_state = filtered.groupby("State", as_index=False).sum(numeric_only=True).sort_values("New_Cases", ascending=False)
    with distr_tabs[0]:
        fig = px.bar(by_state, x="New_Cases", y="State", orientation="h", title="Cases by State", text="New_Cases")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with distr_tabs[1]:
        fig = px.bar(by_state.sort_values("Deaths", ascending=False), x="Deaths", y="State", orientation="h", title="Deaths by State", text="Deaths")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with distr_tabs[2]:
        fig = px.bar(by_state.sort_values("Vaccination_Rate_Percent", ascending=False), x="Vaccination_Rate_Percent", y="State", orientation="h", title="Vaccination Rate by State", text="Vaccination_Rate_Percent")
        fig.update_traces(marker_color="#2ecc71")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with distr_tabs[3]:
        variant_data = filtered["Dominant_Variant"].value_counts().reset_index()
        variant_data.columns = ["Variant", "Count"]
        fig = px.pie(variant_data, names="Variant", values="Count", hole=0.45, title="Variant Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with distr_tabs[4]:
        wave_data = filtered["Wave"].value_counts().reset_index()
        wave_data.columns = ["Wave", "Count"]
        fig = px.pie(wave_data, names="Wave", values="Count", title="COVID Wave Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Multivariate Analysis")
    analysis_tabs = st.tabs([
        "Correlation", "Scatter", "Bubble", "Monthly", "Calendar"
    ])

    corr_columns = [
        "New_Cases",
        "Deaths",
        "Hospitalizations",
        "ICU_Admissions",
        "Positivity_Rate_Percent",
        "Vaccination_Rate_Percent",
    ]
    corr_df = filtered[corr_columns].corr()
    with analysis_tabs[0]:
        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig, use_container_width=True)

    with analysis_tabs[1]:
        scatter = px.scatter(
            filtered,
            x="Vaccination_Rate_Percent",
            y="New_Cases",
            size="Deaths",
            color="Wave",
            hover_data=["State", "Dominant_Variant"],
            title="Vaccination Rate vs New Cases",
            size_max=40,
        )
        st.plotly_chart(scatter, use_container_width=True)

    with analysis_tabs[2]:
        bubble = px.scatter(
            filtered,
            x="New_Cases",
            y="Hospitalizations",
            size="ICU_Admissions",
            color="Wave",
            hover_name="State",
            title="Cases vs Hospitalizations",
            size_max=45,
        )
        st.plotly_chart(bubble, use_container_width=True)

    with analysis_tabs[3]:
        monthly = filtered.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
        monthly_agg = monthly.groupby("Month", as_index=False).sum(numeric_only=True)[["Month", "New_Cases", "Deaths", "Hospitalizations"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_agg["Month"], y=monthly_agg["New_Cases"], mode="lines+markers", name="Cases"))
        fig.add_trace(go.Scatter(x=monthly_agg["Month"], y=monthly_agg["Deaths"], mode="lines+markers", name="Deaths"))
        fig.add_trace(go.Scatter(x=monthly_agg["Month"], y=monthly_agg["Hospitalizations"], mode="lines+markers", name="Hospitalizations"))
        fig.update_layout(title="Monthly Trend", xaxis_title="Month", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with analysis_tabs[4]:
        calendar = filtered.groupby("Date", as_index=False)["New_Cases"].sum()
        fig = px.density_heatmap(calendar, x="Date", y="New_Cases", nbinsx=30, title="Calendar Heatmap: Daily Cases")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Hierarchical & Distribution Charts")
    hierarchy_tabs = st.tabs(["Treemap", "Sunburst", "Box Plot", "Violin Plot"])

    with hierarchy_tabs[0]:
        fig = px.treemap(
            filtered,
            path=["State", "Dominant_Variant"],
            values="New_Cases",
            title="Cases Treemap by State and Variant",
        )
        st.plotly_chart(fig, use_container_width=True)

    with hierarchy_tabs[1]:
        fig = px.sunburst(
            filtered,
            path=["Wave", "Dominant_Variant", "State"],
            values="New_Cases",
            title="Wave → Variant → State Sunburst",
        )
        st.plotly_chart(fig, use_container_width=True)

    with hierarchy_tabs[2]:
        fig = px.box(filtered, x="Wave", y="New_Cases", title="Cases by Wave Box Plot")
        st.plotly_chart(fig, use_container_width=True)

    with hierarchy_tabs[3]:
        fig = px.violin(filtered, x="Wave", y="Deaths", title="Deaths by Wave Violin Plot", box=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Geographic Analysis")
    map_fig = build_state_map(filtered)
    st.plotly_chart(map_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Advanced Analytics")
    analytics = filtered.groupby("State", as_index=False).agg(
        total_cases=("New_Cases", "sum"),
        total_deaths=("Deaths", "sum"),
        total_hospitalizations=("Hospitalizations", "sum"),
        total_icu=("ICU_Admissions", "sum"),
        avg_positivity=("Positivity_Rate_Percent", "mean"),
        avg_vaccination=("Vaccination_Rate_Percent", "mean"),
    )
    analytics["death_ratio"] = analytics["total_deaths"] / analytics["total_cases"].replace(0, np.nan)
    top_states = analytics.nlargest(10, "total_cases")["State"].tolist()
    safest_states = analytics.nsmallest(10, "death_ratio")["State"].tolist()
    highest_positivity = analytics.loc[analytics["avg_positivity"].idxmax()][["State", "avg_positivity"]]
    highest_vaccination = analytics.loc[analytics["avg_vaccination"].idxmax()][["State", "avg_vaccination"]]
    highest_death_ratio = analytics.loc[analytics["death_ratio"].idxmax()][["State", "death_ratio"]]

    growth_series = daily_data.set_index("Date")["New_Cases"].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    avg_daily_growth = growth_series.mean()
    rolling_7 = daily_data["New_Cases"].rolling(7).mean()
    rolling_30 = daily_data["New_Cases"].rolling(30).mean()

    recovery_estimate = max(0, total_cases - total_deaths)

    adv_cols = st.columns(4, gap="large")
    adv_cols[0].metric("Top 10 Most Affected States", ", ".join(top_states[:5]) + ("..." if len(top_states) > 5 else ""))
    adv_cols[1].metric("Top 10 Safest States", ", ".join(safest_states[:5]) + ("..." if len(safest_states) > 5 else ""))
    adv_cols[2].metric("Highest Positivity", f"{highest_positivity['State']} ({highest_positivity['avg_positivity']:.2f}%)")
    adv_cols[3].metric("Highest Vaccination", f"{highest_vaccination['State']} ({highest_vaccination['avg_vaccination']:.2f}%)")

    adv_cols2 = st.columns(4, gap="large")
    adv_cols2[0].metric("Highest Death Ratio", f"{highest_death_ratio['State']} ({highest_death_ratio['death_ratio']:.2%})")
    adv_cols2[1].metric("Estimated Recoveries", format_int(recovery_estimate))
    adv_cols2[2].metric("Avg Daily Growth", f"{avg_daily_growth:.2f}%")
    adv_cols2[3].metric("Rolling Averages", "7d & 30d")

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=daily_data["Date"], y=rolling_7, name="7-day MA", line=dict(color="#f4d35e")))
    trend_fig.add_trace(go.Scatter(x=daily_data["Date"], y=rolling_30, name="30-day MA", line=dict(color="#4cc9f0")))
    trend_fig.update_layout(title="Moving Average Trend", xaxis_title="Date", yaxis_title="New Cases")
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Forecasting")
    if Prophet is not None:
        forecast_cols = st.columns(2, gap="large")
        with forecast_cols[0]:
            st.markdown("### Predict Cases")
            forecast_cases = build_forecast(filtered, "New_Cases")
            st.plotly_chart(forecast_cases, use_container_width=True)
        with forecast_cols[1]:
            st.markdown("### Predict Deaths")
            forecast_deaths = build_forecast(filtered, "Deaths")
            st.plotly_chart(forecast_deaths, use_container_width=True)
    else:
        st.info("Prophet is not installed. Forecasting charts are unavailable until `prophet` is added to the environment.")

    st.markdown("---")
    st.markdown("## Data Quality")
    null_counts = filtered.isnull().sum().to_dict()
    duplicate_count = filtered.duplicated().sum()
    dimensions = filtered.shape
    dtypes = filtered.dtypes.astype(str).to_dict()
    summary_stats = filtered.describe(include="all").transpose()

    dq_cols = st.columns(4, gap="large")
    dq_cols[0].metric("Missing Values", str(sum(null_counts.values())))
    dq_cols[1].metric("Duplicate Records", str(duplicate_count))
    dq_cols[2].metric("Rows", str(dimensions[0]))
    dq_cols[3].metric("Columns", str(dimensions[1]))
    st.dataframe(pd.DataFrame.from_dict(dtypes, orient="index", columns=["Data Type"]))
    st.markdown("### Statistical Summary")
    st.dataframe(summary_stats.style.format({
        "mean": "{:.2f}",
        "std": "{:.2f}",
        "min": "{:.2f}",
        "25%": "{:.2f}",
        "50%": "{:.2f}",
        "75%": "{:.2f}",
        "max": "{:.2f}",
    }))

    st.markdown("---")
    st.markdown("## Download Center")
    csv_bytes = download_df_as_csv(filtered, "filtered_data.csv")
    st.download_button("Download Filtered CSV", csv_bytes, "filtered_covid_dashboard.csv", "text/csv")

    report = {
        "Overview": {
            "Total Cases": format_int(total_cases),
            "Total Deaths": format_int(total_deaths),
            "Average Positivity": format_pct(avg_positivity or 0.0),
            "Average Vaccination": format_pct(avg_vaccination or 0.0),
        },
        "Quality": {
            "Missing Values": str(sum(null_counts.values())),
            "Duplicate Records": str(duplicate_count),
        },
    }
    pdf_bytes = build_pdf_report(report)
    st.download_button("Download Summary Report (PDF)", pdf_bytes, "covid_summary_report.pdf", "application/pdf")

    st.markdown("---")
    st.markdown("### Export a chart as PNG")
    sample_fig = px.line(daily_data, x="Date", y="New_Cases", title="Daily Cases Trend")
    png_bytes = sample_fig.to_image(format="png", engine="kaleido")
    st.download_button("Download Chart PNG", png_bytes, "daily_cases_trend.png", "image/png")

    st.markdown("---")
    st.markdown(
        "<div class='footer'>Built with Streamlit · Interactive COVID-19 analytics dashboard · Data source: Covid19_Wave_Comparison_Dataset.csv</div>",
        unsafe_allow_html=True,
    )


def build_forecast(filtered: pd.DataFrame, target: str) -> go.Figure:
    series = (
        filtered.groupby("Date", as_index=False)[target].sum().rename(columns={"Date": "ds", target: "y"})
    )
    model = Prophet(interval_width=0.95)
    model.fit(series)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color="#5e81ac")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper", line=dict(color="rgba(94,134,172,0.25)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower", fill="tonexty", line=dict(color="rgba(94,134,172,0.25)"), hoverinfo="skip"))
    fig.update_layout(title=f"30-Day Forecast for {target}", xaxis_title="Date", yaxis_title=target)
    return fig


def build_ml_model(df: pd.DataFrame) -> tuple[dict, go.Figure]:
    features = ["Vaccination_Rate_Percent", "Positivity_Rate_Percent", "Hospitalizations", "ICU_Admissions"]
    X = df[features]
    y = df["New_Cases"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return {"mae": mae, "rmse": rmse, "r2": r2}, fig


if __name__ == "__main__":
    main()
