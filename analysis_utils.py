import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from scipy import stats
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.seasonal as smt
import statsmodels.tsa.stattools as ts
import io
import ydata_profiling
from datetime import datetime

# Custom Exception
class DataAnalysisError(Exception):
    pass

# Data Validation
def validate_data(df: pd.DataFrame, selected_columns: List[str], min_rows: int) -> pd.DataFrame:
    """Validate input data for analysis."""
    if df.empty or not selected_columns:
        raise DataAnalysisError("Empty dataframe or no columns selected.")
    
    df_selected = df[selected_columns].copy()
    mask = ~df_selected.isna().any(axis=1)
    df_selected = df_selected[mask]
    
    if len(df_selected) < min_rows:
        raise DataAnalysisError(f"Insufficient valid data: {len(df_selected)} rows (need ≥{min_rows})")
    
    if not all(df_selected.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise DataAnalysisError("Selected columns must contain numeric data only.")
    
    return df_selected

# Data Loading
def load_and_preprocess_data(uploaded_files, n_rows=None) -> pd.DataFrame:
    """Load numeric Excel data or generate sample, preserving small values."""
    if not uploaded_files:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Feature1': rng.normal(1.2, 0.05, 100),
            'Feature2': rng.normal(500, 50, 100),
            'Feature3': rng.normal(30, 2, 100) * 1e-16,
            'Time': np.arange(100),
            'Target': 2 * rng.normal(1.2, 0.05, 100) + np.sin(rng.normal(30, 2, 100)),
            'Category': rng.choice(['A', 'B', 'C'], 100)
        })
        return df

    dfs = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        df_temp = pd.read_excel(io.BytesIO(uploaded_file.read()), engine='openpyxl', dtype_backend='numpy_nullable')
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
        if numeric_cols:
            df_temp = df_temp[numeric_cols + categorical_cols].copy()
            df_temp[numeric_cols] = df_temp[numeric_cols].fillna(df_temp[numeric_cols].median())
            if n_rows:
                df_temp = df_temp.sample(n=min(n_rows, len(df_temp)), random_state=42)
            dfs.append(df_temp)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        for col in numeric_cols:
            df[col] = df[col].astype(np.float64)
        return df
    raise DataAnalysisError("No numeric data found in uploaded files.")

# Data Cleaning Suggestions
def suggest_data_cleaning(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Generate data cleaning suggestions."""
    suggestions = []
    
    # Missing Values
    missing = df.isna().sum()
    for col in df.columns:
        if missing[col] > 0:
            percent_missing = (missing[col] / len(df)) * 100
            if col in numeric_cols:
                suggestions.append(f"Missing values in {col} ({percent_missing:.1f}%): Impute with median.")
            elif col in categorical_cols:
                suggestions.append(f"Missing values in {col} ({percent_missing:.1f}%): Impute with mode or drop.")

    # Outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
        if not outliers.empty:
            suggestions.append(f"Outliers in {col}: Consider capping or removing.")

    # Categorical Encoding
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        if unique_vals > 10:
            suggestions.append(f"High cardinality in {col} ({unique_vals} unique values): Consider grouping or encoding.")
        else:
            suggestions.append(f"Categorical column {col}: Encode with LabelEncoder.")

    return suggestions

# Apply Data Cleaning
def apply_data_cleaning(df: pd.DataFrame, suggestions: List[str]) -> pd.DataFrame:
    """Apply suggested data cleaning steps."""
    df_clean = df.copy()
    for suggestion in suggestions:
        if "Missing values" in suggestion:
            col = suggestion.split(" in ")[1].split(" (")[0]
            if "Impute with median" in suggestion:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif "Impute with mode" in suggestion:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif "Outliers in" in suggestion:
            col = suggestion.split(" in ")[1].split(":")[0]
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif "Encode with LabelEncoder" in suggestion:
            col = suggestion.split("column ")[1].split(":")[0]
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    return df_clean

# Data Transformations
def apply_transformations(df: pd.DataFrame, columns: List[str], transformation: str) -> pd.DataFrame:
    """Apply transformations to numeric columns."""
    df_transformed = df.copy()
    if transformation == "Log":
        for col in columns:
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
    elif transformation == "StandardScaler":
        scaler = StandardScaler()
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
    elif transformation == "MinMaxScaler":
        scaler = MinMaxScaler()
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
    return df_transformed

# Format DataFrame for Display
def format_dataframe_for_display(df: pd.DataFrame, precision: int = 6) -> pd.DataFrame:
    """Format DataFrame to show very small numbers in scientific notation."""
    return df.apply(
        lambda x: x.map(lambda v: f"{v:.{precision}e}" if isinstance(v, (float, np.floating)) else v)
        if x.dtype in [np.float64, np.float32] else x
    )

# Main Analysis Function
@st.cache_data
def analyze_data(
    df: pd.DataFrame,
    selected_columns: List[str],
    analysis_type: str = "summary",
    correlation_method: str = "pearson",
    target_column: str = None,
    histogram_type: str = "frequency",
    num_bins: int = 30,
    lag_column: str = None,
    max_lag: float = None,
    n_lags: int = 10,
    color_scale: str = "RdBu_r",
    n_clusters: int = 3,
    group_column: str = None,
    period: int = None,
    ml_model: str = None
) -> Dict[str, Any]:
    """Perform specified data analysis, handling small values."""
    results = {}

    small_value_threshold = 1e-15
    small_value_cols = [col for col in selected_columns if df[col].abs().max() < small_value_threshold and df[col].abs().max() > 0]
    if small_value_cols:
        results['small_value_warning'] = f"Columns with very small values (< {small_value_threshold}): {', '.join(small_value_cols)}"

    if analysis_type == "summary":
        pd.options.display.float_format = '{:.16e}'.format
        summary = df[selected_columns].describe().T
        summary['range'] = summary['max'] - summary['min']
        summary['variance'] = df[selected_columns].var()
        summary['median'] = df[selected_columns].median()
        summary['mode'] = df[selected_columns].mode().iloc[0] if not df[selected_columns].mode().empty else np.nan
        summary['missing_%'] = df[selected_columns].isnull().mean() * 100
        summary['skewness'] = df[selected_columns].skew()
        summary['kurtosis'] = df[selected_columns].kurtosis()
        results['summary'] = summary

        results['variance_plot'] = px.bar(
            x=selected_columns,
            y=[df[col].var() for col in selected_columns],
            title="Variance of Selected Columns",
            labels={'x': 'Column', 'y': 'Variance'},
            color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
        )

        quality_checks = []
        for col in selected_columns:
            if summary.loc[col, 'missing_%'] > 30:
                quality_checks.append(f"{col}: High missing values ({summary.loc[col, 'missing_%']:.1f}%)")
            if summary.loc[col, 'variance'] < 1e-30:
                quality_checks.append(f"{col}: Extremely low variance ({summary.loc[col, 'variance']:.2e})")
            if summary.loc[col, 'skewness'] > 1 or summary.loc[col, 'skewness'] < -1:
                quality_checks.append(f"{col}: High skewness ({summary.loc[col, 'skewness']:.2f})")
        results['quality_checks'] = quality_checks

    elif analysis_type == "correlation":
        corr_matrix = df[selected_columns].corr(method=correlation_method)
        high_corrs = []
        if len(selected_columns) > 1:
            for i in range(len(selected_columns)):
                for j in range(i + 1, len(selected_columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        high_corrs.append(f"{selected_columns[i]} and {selected_columns[j]}: {corr:.3f}")
        results['correlation'] = corr_matrix
        results['high_correlations'] = high_corrs

        results['corr_heatmap'] = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=color_scale,
            title="Correlation Heatmap"
        )

        if len(selected_columns) > 1:
            corr_pairs = []
            for i in range(len(selected_columns)):
                for j in range(i + 1, len(selected_columns)):
                    corr = corr_matrix.iloc[i, j]
                    corr_pairs.append({
                        'x': selected_columns[i],
                        'y': selected_columns[j],
                        'Correlation': abs(corr),
                        'Signed_Correlation': corr
                    })
            corr_df = pd.DataFrame(corr_pairs)
            results['corr_scatter'] = px.scatter(
                corr_df,
                x='x',
                y='y',
                size='Correlation',
                color='Signed_Correlation',
                color_continuous_scale=color_scale,
                title="Scatter Corrlogram (Size = |Correlation|, Color = Correlation)"
            )

    elif analysis_type == "distribution":
        histograms = {}
        for col in selected_columns:
            if histogram_type == "frequency":
                fig = px.histogram(df, x=col, nbins=num_bins, title=f"Distribution of {col}", color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]])
            elif histogram_type == "density":
                fig = px.histogram(df, x=col, nbins=num_bins, histnorm='probability density', title=f"Distribution of {col}", color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]])
            elif histogram_type == "cumulative":
                fig = px.histogram(df, x=col, nbins=num_bins, cumulative=True, title=f"Cumulative Distribution of {col}", color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]])

            if histogram_type in ["frequency", "density"]:
                kde_x = np.linspace(df[col].min(), df[col].max(), 100)
                kde = stats.gaussian_kde(df[col].dropna())
                kde_y = kde(kde_x)
                if histogram_type == "frequency":
                    kde_y = kde_y * len(df[col]) * (df[col].max() - df[col].min()) / num_bins
                fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE', line=dict(color='red')))
            
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            fig.add_annotation(
                text=f"Skewness: {skewness:.2f}<br>Kurtosis: {kurtosis:.2f}",
                xref="paper", yref="paper", x=0.95, y=0.95, showarrow=False
            )
            histograms[col] = fig
        results['histograms'] = histograms

    elif analysis_type == "boxplot":
        results['boxplot'] = px.box(df[selected_columns], title="Box Plots for Selected Columns", color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]])

    elif analysis_type == "scatter":
        if len(selected_columns) > 1:
            results['scatter'] = px.scatter_matrix(df[selected_columns], title="Scatter Matrix", color_continuous_scale=color_scale)
        else:
            raise DataAnalysisError("Scatter analysis requires at least two columns.")

    elif analysis_type == "outliers":
        outliers = {}
        for col in selected_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            outliers[col] = df[outlier_mask][[col]].reset_index()
        results['outliers'] = outliers

    elif analysis_type == "feature_importance" and target_column:
        importance = []
        target_data = df[target_column].copy()
        for col in selected_columns:
            if col != target_column:
                corr = df[[col, target_column]].corr(method=correlation_method).iloc[0, 1]
                importance.append({"Feature": col, "Correlation": abs(corr)})
        importance_df = pd.DataFrame(importance).sort_values(by="Correlation", ascending=False)
        results['feature_importance'] = importance_df
        results['importance_plot'] = px.bar(
            importance_df,
            x="Feature",
            y="Correlation",
            title=f"Feature Importance (Correlation with {target_column})",
            color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
        )

    elif analysis_type == "normality":
        normality_results = []
        for col in selected_columns:
            data = df[col].dropna()
            if len(data) >= 3:
                stat, p_value = stats.shapiro(data)
                normality_results.append({
                    "Column": col,
                    "Statistic": stat,
                    "P-Value": p_value,
                    "Interpretation": "Non-normal" if p_value < 0.05 else "Likely normal"
                })
            else:
                normality_results.append({
                    "Column": col,
                    "Statistic": np.nan,
                    "P-Value": np.nan,
                    "Interpretation": "Insufficient data"
                })
        results['normality'] = pd.DataFrame(normality_results)

    elif analysis_type == "pca":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_columns].dropna())
        pca = IncrementalPCA(n_components=min(len(selected_columns), len(X_scaled)))
        pca.fit(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        components = pca.transform(X_scaled)
        
        results['pca_variance'] = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))],
            y=explained_variance_ratio,
            title="PCA Explained Variance Ratio",
            color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
        )
        
        if len(explained_variance_ratio) >= 2:
            pca_df = pd.DataFrame({
                "PC1": components[:, 0],
                "PC2": components[:, 1]
            })
            results['pca_scatter'] = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                title="PCA: First Two Principal Components",
                color_continuous_scale=color_scale
            )

    elif analysis_type == "variogram" and lag_column:
        variograms = {}
        for col in selected_columns:
            if col != lag_column:
                data = df[[col, lag_column]].dropna()
                values = data[col].values
                lags = data[lag_column].values
                
                max_lag = max_lag or (lags.max() - lags.min()) / 2
                lag_bins = np.linspace(0, max_lag, n_lags + 1)
                
                lag_diffs = []
                semi_vars = []
                for i in range(len(lags)):
                    for j in range(i + 1, len(lags)):
                        lag_diff = abs(lags[i] - lags[j])
                        if lag_diff <= max_lag:
                            semi_var = (values[i] - values[j]) ** 2 / 2
                            lag_diffs.append(lag_diff)
                            semi_vars.append(semi_var)
                
                bin_means = []
                bin_lags = []
                for i in range(len(lag_bins) - 1):
                    mask = (np.array(lag_diffs) >= lag_bins[i]) & (np.array(lag_diffs) < lag_bins[i + 1])
                    if np.sum(mask) > 0:
                        bin_means.append(np.mean(np.array(semi_vars)[mask]))
                        bin_lags.append((lag_bins[i] + lag_bins[i + 1]) / 2)
                
                fig = px.scatter(
                    x=bin_lags,
                    y=bin_means,
                    title=f"Semivariogram of {col} vs {lag_column}",
                    labels={'x': 'Lag', 'y': 'Semivariance'},
                    color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
                )
                variograms[col] = fig
        results['variograms'] = variograms

    elif analysis_type == "autocorrelation" and lag_column:
        autocorrs = {}
        for col in selected_columns:
            if col != lag_column:
                data = df[[col, lag_column]].dropna().sort_values(by=lag_column)[col].values
                if len(data) > n_lags:
                    acf = ts.acf(data, nlags=n_lags, fft=False)
                    fig = px.bar(
                        x=range(n_lags + 1),
                        y=acf,
                        title=f"Autocorrelation of {col}",
                        labels={'x': 'Lag', 'y': 'ACF'},
                        color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
                    )
                    autocorrs[col] = fig
                else:
                    autocorrs[col] = None
        results['autocorrelations'] = autocorrs

    elif analysis_type == "clustering":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_columns].dropna())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        cluster_df = df[selected_columns].dropna().copy()
        cluster_df['Cluster'] = clusters
        
        cluster_summary = cluster_df.groupby('Cluster').mean()
        results['cluster_summary'] = cluster_summary
        
        if len(selected_columns) >= 2:
            results['cluster_plot'] = px.scatter(
                cluster_df,
                x=selected_columns[0],
                y=selected_columns[1],
                color='Cluster',
                title="K-means Clustering",
                color_continuous_scale=color_scale
            )
        else:
            results['cluster_plot'] = px.scatter(
                cluster_df,
                x=selected_columns[0],
                y=cluster_df.index,
                color='Cluster',
                title="K-means Clustering (Single Feature)",
                color_continuous_scale=color_scale
            )

    elif analysis_type == "ml_pipeline" and target_column:
        X = df[[col for col in selected_columns if col != target_column]].dropna()
        y = df[target_column].dropna()
        if len(X) != len(y):
            raise DataAnalysisError("Mismatch between features and target data after removing NaNs.")
        
        # Determine if classification or regression
        is_classification = df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 10
        if is_classification:
            y = LabelEncoder().fit_transform(y)
            if ml_model == "Linear Regression":
                raise DataAnalysisError("Linear Regression not suitable for classification.")
            model = LogisticRegression() if ml_model == "Logistic Regression" else RandomForestClassifier(random_state=42)
            metrics = classification_report(y, model.fit(X, y).predict(X), output_dict=True)
            results['ml_score'] = metrics['weighted avg']['f1-score']
            results['ml_metrics'] = pd.DataFrame(metrics).T
        else:
            model = LinearRegression() if ml_model == "Linear Regression" else RandomForestRegressor(random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)
            results['ml_score'] = model.score(X, y)
            results['ml_metrics'] = pd.DataFrame({
                'Metric': ['R²', 'MSE'],
                'Value': [results['ml_score'], mean_squared_error(y, y_pred)]
            })
        
        if ml_model == "Random Forest":
            results['feature_importance'] = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        
        if len(X.columns) >= 1:
            results['ml_plot'] = px.scatter(
                x=X.iloc[:, 0], y=y,
                title=f"{ml_model}: {target_column} vs {X.columns[0]}",
                labels={'x': X.columns[0], 'y': target_column},
                color_continuous_scale=color_scale
            )
            if not is_classification:
                results['ml_plot'].add_trace(
                    go.Scatter(x=X.iloc[:, 0], y=y_pred, mode='lines', name='Fit', line=dict(color='red'))
                )

    elif analysis_type == "timeseries_decomposition" and lag_column:
        decompositions = {}
        for col in selected_columns:
            if col != lag_column:
                data = df[[col, lag_column]].dropna().sort_values(by=lag_column)[col]
                if len(data) >= period * 2:
                    decomposition = smt.DecomposeResult(
                        observed=data,
                        trend=smt.STL(data, period=period).fit().trend,
                        seasonal=smt.STL(data, period=period).fit().seasonal,
                        resid=smt.STL(data, period=period).fit().resid
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df[lag_column], y=decomposition.observed, mode='lines', name='Observed'))
                    fig.add_trace(go.Scatter(x=df[lag_column], y=decomposition.trend, mode='lines', name='Trend', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df[lag_column], y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=df[lag_column], y=decomposition.resid, mode='lines', name='Residual', line=dict(color='purple')))
                    fig.update_layout(title=f"Time-Series Decomposition of {col}", xaxis_title=lag_column, yaxis_title=col)
                    decompositions[col] = fig
                else:
                    decompositions[col] = None
        results['decompositions'] = decompositions

    elif analysis_type == "anomaly_detection":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_columns].dropna())
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        anomaly_df = df[selected_columns].dropna().copy()
        anomaly_df['Anomaly'] = anomalies == -1
        results['anomaly_df'] = anomaly_df[anomaly_df['Anomaly']][selected_columns]
        
        if len(selected_columns) >= 2:
            results['anomaly_plot'] = px.scatter(
                anomaly_df,
                x=selected_columns[0],
                y=selected_columns[1],
                color='Anomaly',
                title="Anomaly Detection (Isolation Forest)",
                color_discrete_map={True: 'red', False: 'blue'}
            )
        else:
            results['anomaly_plot'] = px.scatter(
                anomaly_df,
                x=selected_columns[0],
                y=anomaly_df.index,
                color='Anomaly',
                title="Anomaly Detection (Single Feature)",
                color_discrete_map={True: 'red', False: 'blue'}
            )

    elif analysis_type == "group_stats" and group_column:
        group_stats = df.groupby(group_column)[selected_columns].agg(['mean', 'std', 'min', 'max']).reset_index()
        results['group_stats'] = group_stats
        
        if len(selected_columns) >= 1:
            results['group_plot'] = px.bar(
                group_stats,
                x=group_column,
                y=(selected_columns[0], 'mean'),
                title=f"Mean of {selected_columns[0]} by {group_column}",
                color_discrete_sequence=[px.colors.sequential.__dict__[color_scale][0]]
            )

    elif analysis_type == "eda_report":
        profile = ydata_profiling.ProfileReport(df, title="Automated EDA Report")
        eda_file = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        profile.to_file(eda_file)
        results['eda_report'] = eda_file

    elif analysis_type == "forecasting" and lag_column:
        forecasts = {}
        for col in selected_columns:
            if col != lag_column:
                data = df[[col, lag_column]].dropna().sort_values(by=lag_column)[col]
                if len(data) >= 10:
                    model = ARIMA(data, order=(1, 1, 1)).fit()
                    forecast = model.forecast(steps=10)
                    forecast_index = range(int(df[lag_column].max()) + 1, int(df[lag_column].max()) + 11)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df[lag_column], y=data, mode='lines', name='Historical'))
                    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
                    fig.update_layout(title=f"ARIMA Forecast for {col}", xaxis_title=lag_column, yaxis_title=col)
                    forecasts[col] = {
                        'plot': fig,
                        'forecast_df': pd.DataFrame({'Time': forecast_index, 'Forecast': forecast})
                    }
                else:
                    forecasts[col] = {'plot': None, 'forecast_df': pd.DataFrame()}
        results['forecasts'] = forecasts

    return results

# Report Generation
def generate_report(
    analysis_result: Dict[str, Any],
    analysis_type: str,
    analysis_columns: List[str],
    target_column: str,
    lag_column: str,
    group_column: str,
    analysis_options: Dict[str, str]
) -> str:
    """Generate a text report of the analysis results."""
    report_text = f"""Data Analysis Report
===================

**Analysis Type**: {analysis_options[analysis_type]}
**Columns**: {', '.join(analysis_columns)}
**Target (if applicable)**: {target_column if target_column != "None" else "N/A"}
**Lag Column (if applicable)**: {lag_column if lag_column != "None" else "N/A"}
**Group Column (if applicable)**: {group_column if group_column != "None" else "N/A"}

**Results**:
"""
    if 'small_value_warning' in analysis_result:
        report_text += f"\n**Warning**: {analysis_result['small_value_warning']}\n"
    
    if analysis_type == "summary":
        report_text += analysis_result['summary'].to_string()
        if analysis_result['quality_checks']:
            report_text += "\n\n**Data Quality Issues**:\n" + "\n".join(analysis_result['quality_checks'])
    elif analysis_type == "correlation":
        report_text += analysis_result['correlation'].to_string()
        if analysis_result['high_correlations']:
            report_text += "\n\n**High Correlations**:\n" + "\n".join(analysis_result['high_correlations'])
    elif analysis_type == "outliers":
        for col, outliers in analysis_result['outliers'].items():
            report_text += f"\n**{col} Outliers**:\n{outliers.to_string() if not outliers.empty else 'None'}\n"
    elif analysis_type == "normality":
        report_text += analysis_result['normality'].to_string()
    elif analysis_type == "feature_importance":
        report_text += analysis_result['feature_importance'].to_string()
    elif analysis_type == "pca":
        report_text += f"Explained Variance Ratio: {analysis_result['pca_variance'].data[0].y.tolist()}"
    elif analysis_type == "variogram":
        report_text += f"Semivariograms computed for {', '.join(analysis_result['variograms'].keys())}"
    elif analysis_type == "autocorrelation":
        report_text += f"Autocorrelations computed for {', '.join([col for col, fig in analysis_result['autocorrelations'].items() if fig])}"
    elif analysis_type == "clustering":
        report_text += analysis_result['cluster_summary'].to_string()
    elif analysis_type == "ml_pipeline":
        report_text += f"Model Score: {analysis_result['ml_score']:.3f}\nMetrics:\n{analysis_result['ml_metrics'].to_string()}"
        if 'feature_importance' in analysis_result:
            report_text += f"\nFeature Importance:\n{analysis_result['feature_importance'].to_string()}"
    elif analysis_type == "timeseries_decomposition":
        report_text += f"Decompositions computed for {', '.join([col for col, fig in analysis_result['decompositions'].items() if fig])}"
    elif analysis_type == "anomaly_detection":
        report_text += analysis_result['anomaly_df'].to_string()
    elif analysis_type == "group_stats":
        report_text += analysis_result['group_stats'].to_string()
    elif analysis_type == "eda_report":
        report_text += "Automated EDA report generated and saved as HTML."
    elif analysis_type == "forecasting":
        report_text += f"Forecasts computed for {', '.join([col for col, forecast in analysis_result['forecasts'].items() if forecast['plot']])}"

    return report_text
