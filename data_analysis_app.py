import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import ydata_profiling  # For Automated EDA
from typing import List, Dict, Any
from analysis_utils import (
    DataAnalysisError, analyze_data, load_and_preprocess_data,
    validate_data, generate_report, format_dataframe_for_display,
    suggest_data_cleaning, apply_transformations
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from streamlit.components.v1 import html  # For dashboard rendering

# Streamlit UI Configuration
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

st.title("📊 Advanced Data Analysis App")

# Enhanced Help Section
with st.sidebar.expander("ℹ️ User Guide"):
    st.markdown("""
    ### Welcome to the Advanced Data Analysis App
    Upload Excel files (`.xlsx`, `.xls`) to analyze numeric data. Features include:
    - **Summary Statistics**: Mean, median, variance, skewness, kurtosis, etc.
    - **Correlation Analysis**: Heatmap and scatter corrlogram (Pearson, Spearman, Kendall).
    - **Distribution Analysis**: Histograms (frequency, density, cumulative) with KDE.
    - **Box Plots**: Visualize outliers.
    - **Scatter Matrix**: Pairwise scatter plots.
    - **Outlier Detection**: Identify outliers using IQR.
    - **Feature Importance**: Correlation-based ranking with a target variable.
    - **Normality Testing**: Shapiro-Wilk test.
    - **PCA**: Principal Component Analysis with explained variance.
    - **Variogram**: Spatial/temporal dependence analysis.
    - **Autocorrelation**: ACF for time-series data.
    - **Clustering**: K-means clustering for grouping similar data points.
    - **Regression**: Linear, logistic, random forest modeling.
    - **Time-Series Decomposition**: Decompose into trend, seasonal, and residual components.
    - **Anomaly Detection**: Identify anomalies using Isolation Forest.
    - **Group Statistics**: Summary statistics by categorical group.
    - **Automated EDA**: Comprehensive report with visuals (like pandas-profiling).
    - **Data Cleaning**: Suggestions for missing values, outliers, and encoding.
    - **ML Pipeline**: AutoML-like model selection and evaluation.
    - **Forecasting**: ARIMA-based time-series forecasting.
    - **Interactive Dashboards**: Save analyses as interactive dashboards.
    - **Data Transformations**: Normalization, log transforms, feature engineering.

    **Tips**:
    - Select numeric columns for analysis.
    - Use target variable for feature importance or regression.
    - Use lag column for time-series or spatial analyses.
    - Adjust visualization settings for better insights.
    - Download reports or save dashboards.
    """)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
min_rows = st.sidebar.number_input("Min Rows", min_value=5, value=10, help="Minimum data points required")
correlation_method = st.sidebar.selectbox(
    "Correlation Method",
    options=["pearson", "spearman", "kendall"],
    index=0,
    help="Method for correlation and feature importance"
)
histogram_type = st.sidebar.selectbox(
    "Histogram Type",
    options=["frequency", "density", "cumulative"],
    index=0,
    help="Type of histogram to display"
)
num_bins = st.sidebar.number_input("Histogram Bins", min_value=5, value=30, help="Number of bins for histograms")
max_lag = st.sidebar.number_input("Max Lag for Variogram/Forecasting", min_value=0.0, value=0.0, help="Maximum lag distance (0 for auto)")
n_lags = st.sidebar.number_input("Number of Lags", min_value=5, value=10, help="Number of lag bins for variogram/autocorrelation")
color_scale = st.sidebar.selectbox(
    "Plot Color Scale",
    options=["Viridis", "Plasma", "Inferno", "Magma", "RdBu_r"],
    index=4,
    help="Color scheme for visualizations"
)
n_clusters = st.sidebar.number_input("Number of Clusters (K-means)", min_value=2, value=3, help="Number of clusters for K-means")
period = st.sidebar.number_input("Period for Time-Series Decomposition", min_value=2, value=12, help="Period for seasonal decomposition")
ml_model = st.sidebar.selectbox(
    "ML Model (for ML Pipeline)",
    options=["Linear Regression", "Logistic Regression", "Random Forest"],
    index=0,
    help="Select model for ML pipeline"
)
transformation = st.sidebar.selectbox(
    "Data Transformation",
    options=["None", "Log", "StandardScaler", "MinMaxScaler"],
    index=0,
    help="Apply transformation to numeric columns"
)

# File Upload
uploaded_files = st.file_uploader("📁 Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
n_rows_input = st.number_input("Sample rows (0 for all)", min_value=0, value=0)
n_rows = None if n_rows_input == 0 else n_rows_input

if st.button("Load Data"):
    try:
        df = load_and_preprocess_data(uploaded_files, n_rows)
        if not df.empty:
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df)} rows with {len(df.select_dtypes(include=[np.number]).columns)} numeric columns")
            with st.expander("👁️ Data Preview"):
                st.dataframe(format_dataframe_for_display(df.head(10)), use_container_width=True)
        else:
            st.error("❌ No numeric data found in uploaded files.")
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")

if 'df' not in st.session_state:
    st.warning("⚠️ Load data first.")
    st.stop()

df = st.session_state.df
numeric_params = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_params = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Data Cleaning Assistant
with st.expander("🧹 Data Cleaning Assistant"):
    cleaning_suggestions = suggest_data_cleaning(df, numeric_params, categorical_params)
    st.write("**Cleaning Suggestions**")
    for suggestion in cleaning_suggestions:
        st.write(f"- {suggestion}")
    if st.button("Apply Suggested Cleaning"):
        try:
            df = apply_data_cleaning(df, cleaning_suggestions)
            st.session_state.df = df
            st.success("✅ Data cleaning applied successfully!")
            st.dataframe(format_dataframe_for_display(df.head(10)), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error applying cleaning: {str(e)}")

# Data Transformation
if transformation != "None":
    try:
        df_transformed = apply_transformations(df, numeric_params, transformation)
        st.session_state.df = df_transformed
        st.success(f"✅ Applied {transformation} to numeric columns")
        with st.expander("👁️ Transformed Data Preview"):
            st.dataframe(format_dataframe_for_display(df_transformed.head(10)), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error applying transformation: {str(e)}")

# Column Selection
col1, col2, col3 = st.columns(3)
with col1:
    analysis_columns = st.multiselect(
        "🔧 Select Columns to Analyze",
        options=numeric_params,
        default=numeric_params,
        help="Select numeric columns for analysis"
    )
with col2:
    target_column = st.selectbox(
        "🎯 Target Variable (Optional)",
        options=["None"] + numeric_params + categorical_params,
        index=0,
        help="Select a target for feature importance or ML pipeline"
    )
with col3:
    lag_column = st.selectbox(
        "📍 Lag Column (Optional)",
        options=["None"] + numeric_params,
        index=0,
        help="Select a column for lag (e.g., time or distance)"
    )

group_column = st.selectbox(
    "📊 Group Column (Optional)",
    options=["None"] + categorical_params,
    index=0,
    help="Select a categorical column for group-based statistics"
)

if not analysis_columns:
    st.error("❌ Select at least one column.")
    st.stop()

# Analysis Options
analysis_options = {
    "summary": "Summary Statistics (Mean, Variance, Skewness, etc.)",
    "correlation": "Correlation Matrix (Heatmap & Scatter Corrlogram)",
    "distribution": "Distribution Histograms (with KDE)",
    "boxplot": "Box Plots (Outliers)",
    "scatter": "Scatter Matrix (Pairwise)",
    "outliers": "Outlier Detection (IQR)",
    "feature_importance": "Feature Importance (Correlation-Based)",
    "normality": "Normality Testing (Shapiro-Wilk)",
    "pca": "Principal Component Analysis (PCA)",
    "variogram": "Semivariogram (Spatial/Temporal Dependence)",
    "autocorrelation": "Autocorrelation Function (ACF)",
    "clustering": "K-means Clustering",
    "ml_pipeline": "ML Pipeline (Regression/Classification)",
    "timeseries_decomposition": "Time-Series Decomposition",
    "anomaly_detection": "Anomaly Detection (Isolation Forest)",
    "group_stats": "Descriptive Statistics by Group",
    "eda_report": "Automated EDA Report",
    "forecasting": "ARIMA Forecasting"
}
selected_analysis_key = st.radio(
    "📊 Select Analysis Type",
    options=list(analysis_options.keys()),
    format_func=lambda key: analysis_options[key],
    index=0,
    help="Choose the type of analysis to perform."
)

# Validation for specific analyses
if selected_analysis_key == "feature_importance" and target_column == "None":
    st.error("❌ Select a target variable for feature importance analysis.")
    st.stop()
if selected_analysis_key in ["variogram", "autocorrelation", "timeseries_decomposition", "forecasting"] and lag_column == "None":
    st.error("❌ Select a lag column for time-series or spatial analysis.")
    st.stop()
if selected_analysis_key == "ml_pipeline" and target_column == "None":
    st.error("❌ Select a target variable for ML pipeline.")
    st.stop()
if selected_analysis_key == "group_stats" and group_column == "None":
    st.error("❌ Select a group column for group-based statistics.")
    st.stop()

# Run Analysis
if st.button("🚀 Run Analysis", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📊 Validating data...")
        progress_bar.progress(0.2)
        df_analysis = validate_data(df, analysis_columns, min_rows)

        status_text.text(f"🔍 Running {analysis_options[selected_analysis_key]}...")
        progress_bar.progress(0.4)

        analysis_result = analyze_data(
            df_analysis,
            analysis_columns,
            analysis_type=selected_analysis_key,
            correlation_method=correlation_method,
            target_column=target_column if target_column != "None" else None,
            histogram_type=histogram_type,
            num_bins=num_bins,
            lag_column=lag_column if lag_column != "None" else None,
            max_lag=max_lag if max_lag > 0 else None,
            n_lags=n_lags,
            color_scale=color_scale,
            n_clusters=n_clusters,
            group_column=group_column if group_column != "None" else None,
            period=period,
            ml_model=ml_model
        )

        progress_bar.progress(0.8)
        status_text.text("📈 Generating results...")

        st.success(f"✅ Analysis completed: {analysis_options[selected_analysis_key]}")

        # Display Results
        if 'small_value_warning' in analysis_result:
            st.warning(f"⚠️ {analysis_result['small_value_warning']}")

        if selected_analysis_key == "summary":
            st.subheader("📜 Summary Statistics")
            st.dataframe(format_dataframe_for_display(analysis_result['summary']), use_container_width=True)
            st.plotly_chart(analysis_result['variance_plot'], use_container_width=True)
            if analysis_result['quality_checks']:
                st.warning("⚠️ Data Quality Issues:\n" + "\n".join(analysis_result['quality_checks']))

        elif selected_analysis_key == "correlation":
            st.subheader("📜 Correlation Matrix")
            st.plotly_chart(analysis_result['corr_heatmap'], use_container_width=True)
            if len(analysis_columns) > 1:
                st.subheader("📜 Scatter Corrlogram")
                st.plotly_chart(analysis_result['corr_scatter'], use_container_width=True)
            if analysis_result['high_correlations']:
                st.warning("⚠️ High Correlations:\n" + "\n".join(analysis_result['high_correlations']))

        elif selected_analysis_key == "distribution":
            st.subheader("📜 Distribution Histograms")
            for col, fig in analysis_result['histograms'].items():
                st.plotly_chart(fig, use_container_width=True)

        elif selected_analysis_key == "boxplot":
            st.subheader("📜 Box Plots")
            st.plotly_chart(analysis_result['boxplot'], use_container_width=True)

        elif selected_analysis_key == "scatter":
            st.subheader("📜 Scatter Matrix")
            st.plotly_chart(analysis_result['scatter'], use_container_width=True)

        elif selected_analysis_key == "outliers":
            st.subheader("📜 Detected Outliers")
            for col, outliers in analysis_result['outliers'].items():
                if not outliers.empty:
                    st.write(f"**{col}:**")
                    st.dataframe(format_dataframe_for_display(outliers), use_container_width=True)
                else:
                    st.info(f"No outliers detected in {col}.")

        elif selected_analysis_key == "feature_importance":
            st.subheader("📜 Feature Importance")
            st.dataframe(format_dataframe_for_display(analysis_result['feature_importance']), use_container_width=True)
            st.plotly_chart(analysis_result['importance_plot'], use_container_width=True)

        elif selected_analysis_key == "normality":
            st.subheader("📜 Normality Testing (Shapiro-Wilk)")
            st.dataframe(format_dataframe_for_display(analysis_result['normality']), use_container_width=True)

        elif selected_analysis_key == "pca":
            st.subheader("📜 PCA Explained Variance")
            st.plotly_chart(analysis_result['pca_variance'], use_container_width=True)
            if 'pca_scatter' in analysis_result:
                st.subheader("📜 PCA Scatter Plot")
                st.plotly_chart(analysis_result['pca_scatter'], use_container_width=True)

        elif selected_analysis_key == "variogram":
            st.subheader("📜 Semivariograms")
            for col, fig in analysis_result['variograms'].items():
                st.plotly_chart(fig, use_container_width=True)

        elif selected_analysis_key == "autocorrelation":
            st.subheader("📜 Autocorrelation Functions")
            for col, fig in analysis_result['autocorrelations'].items():
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for autocorrelation of {col}.")

        elif selected_analysis_key == "clustering":
            st.subheader("📜 K-means Clustering")
            st.plotly_chart(analysis_result['cluster_plot'], use_container_width=True)
            st.dataframe(format_dataframe_for_display(analysis_result['cluster_summary']), use_container_width=True)

        elif selected_analysis_key == "ml_pipeline":
            st.subheader("📜 ML Pipeline Results")
            st.write(f"**Model**: {ml_model}")
            st.write(f"**Score**: {analysis_result['ml_score']:.3f}")
            st.dataframe(format_dataframe_for_display(analysis_result['ml_metrics']), use_container_width=True)
            if 'feature_importance' in analysis_result:
                st.subheader("📜 Feature Importance")
                st.dataframe(format_dataframe_for_display(analysis_result['feature_importance']), use_container_width=True)
            st.plotly_chart(analysis_result['ml_plot'], use_container_width=True)

        elif selected_analysis_key == "timeseries_decomposition":
            st.subheader("📜 Time-Series Decomposition")
            for col, fig in analysis_result['decompositions'].items():
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for decomposition of {col}.")

        elif selected_analysis_key == "anomaly_detection":
            st.subheader("📜 Anomaly Detection")
            if not analysis_result['anomaly_df'].empty:
                st.dataframe(format_dataframe_for_display(analysis_result['anomaly_df']), use_container_width=True)
                st.plotly_chart(analysis_result['anomaly_plot'], use_container_width=True)
            else:
                st.info("No anomalies detected.")

        elif selected_analysis_key == "group_stats":
            st.subheader("📜 Group Statistics")
            st.dataframe(format_dataframe_for_display(analysis_result['group_stats']), use_container_width=True)
            if 'group_plot' in analysis_result:
                st.plotly_chart(analysis_result['group_plot'], use_container_width=True)

        elif selected_analysis_key == "eda_report":
            st.subheader("📜 Automated EDA Report")
            with open(analysis_result['eda_report'], 'r') as file:
                html_content = file.read()
            html(html_content, height=800, scrolling=True)
            st.download_button(
                "💾 Download EDA Report",
                html_content,
                f"eda_report.html",
                "text/html"
            )

        elif selected_analysis_key == "forecasting":
            st.subheader("📜 ARIMA Forecasting")
            for col, forecast in analysis_result['forecasts'].items():
                st.plotly_chart(forecast['plot'], use_container_width=True)
                st.dataframe(format_dataframe_for_display(forecast['forecast_df']), use_container_width=True)

        # Generate and Offer Report Download
        report_text = generate_report(
            analysis_result, selected_analysis_key, analysis_columns,
            target_column, lag_column, group_column, analysis_options
        )
        st.download_button(
            "💾 Download Report",
            report_text,
            f"analysis_report_{selected_analysis_key}.txt",
            "text/plain"
        )

        # Interactive Dashboard Option
        if st.button("💾 Save as Interactive Dashboard"):
            dashboard_html = generate_dashboard(df_analysis, analysis_result, selected_analysis_key, analysis_columns)
            st.download_button(
                "💾 Download Dashboard",
                dashboard_html,
                f"dashboard_{selected_analysis_key}.html",
                "text/html"
            )
            html(dashboard_html, height=800, scrolling=True)

        progress_bar.progress(1.0)
    except DataAnalysisError as e:
        st.error(f"❌ Analysis Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def generate_dashboard(df: pd.DataFrame, analysis_result: Dict[str, Any], analysis_type: str, columns: List[str]) -> str:
    """Generate an interactive HTML dashboard."""
    html_content = """
    <html>
    <head>
        <title>Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .plot-container { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Interactive Data Analysis Dashboard</h1>
        <h2>Analysis Type: {}</h2>
        <h3>Columns: {}</h3>
    """.format(analysis_options[analysis_type], ", ".join(columns))

    if analysis_type == "summary":
        html_content += f"<h3>Summary Statistics</h3><pre>{analysis_result['summary'].to_html()}</pre>"
        html_content += f'<div class="plot-container" id="variance_plot"></div><script>{analysis_result["variance_plot"].to_json()}</script>'
    elif analysis_type == "correlation":
        html_content += f'<div class="plot-container" id="corr_heatmap"></div><script>{analysis_result["corr_heatmap"].to_json()}</script>'
        if 'corr_scatter' in analysis_result:
            html_content += f'<div class="plot-container" id="corr_scatter"></div><script>{analysis_result["corr_scatter"].to_json()}</script>'
    # Add similar blocks for other analysis types

    html_content += "</body></html>"
    return html_content
