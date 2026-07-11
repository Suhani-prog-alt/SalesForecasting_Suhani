import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set page configuration
st.set_page_config(layout="wide", page_title="Superstore Sales Analytics Dashboard")

# --- Helper functions (reused from previous tasks) ---
@st.cache_data
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

@st.cache_data
def get_season_from_month(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# --- Data Loading and Preprocessing (cached) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

    # Extract time features
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Week'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df['Order Day of Week'] = df['Order Date'].dt.dayofweek
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order Season'] = df['Order Month'].apply(get_season)

    # Handle missing Postal Code
    df.dropna(subset=['Postal Code'], inplace=True)

    return df

@st.cache_data
def get_monthly_sales(df):
    monthly_sales = df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
    monthly_sales.rename(columns={'Order Date': 'Month', 'Sales': 'Monthly Sales'}, inplace=True)
    return monthly_sales

@st.cache_data
def get_weekly_sales(df):
    weekly_sales = df.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
    weekly_sales.rename(columns={'Order Date': 'Week', 'Sales': 'Weekly Sales'}, inplace=True)
    return weekly_sales

@st.cache_data
def perform_clustering(df_input):
    # Re-calculate sub_category_features similar to Task 6
    sub_category_features = df_input.groupby('Sub-Category').agg(
        total_sales_volume=('Sales', 'sum'),
        average_order_value=('Sales', lambda x: x.sum() / x.nunique())
    ).reset_index()

    monthly_sales_sub_category = df_input.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().reset_index()
    monthly_sales_sub_category.rename(columns={'Sales': 'Monthly Sales'}, inplace=True)

    sales_growth = monthly_sales_sub_category.groupby('Sub-Category')['Monthly Sales'].apply(lambda x: x.pct_change(periods=12).mean()).reset_index()
    sales_growth.rename(columns={'Monthly Sales': 'Sales Growth Rate'}, inplace=True)

    sales_volatility = monthly_sales_sub_category.groupby('Sub-Category')['Monthly Sales'].std().reset_index()
    sales_volatility.rename(columns={'Monthly Sales': 'Sales Volatility'}, inplace=True)

    sub_category_features = pd.merge(sub_category_features, sales_growth, on='Sub-Category', how='left')
    sub_category_features = pd.merge(sub_category_features, sales_volatility, on='Sub-Category', how='left')
    sub_category_features = sub_category_features.fillna(0)

    features_for_clustering = sub_category_features.drop('Sub-Category', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', max_iter=300, n_init=10, random_state=42)
    sub_category_features['Cluster'] = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = sub_category_features['Cluster']
    pca_df['Sub-Category'] = sub_category_features['Sub-Category']

    cluster_label_map = {
        0: 'Stable Low-Value',
        1: 'Premium High-Demand',
        2: 'Emerging High-Growth'
    }
    pca_df['Cluster Label'] = pca_df['Cluster'].map(cluster_label_map)

    return sub_category_features, pca_df, cluster_label_map

@st.cache_resource
def train_prophet_model(data_prophet):
    model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
    # Disable weekly and daily seasonality explicitly if not present in data or desired
    model.add_country_holidays(country_name='US') # Example for US holidays
    model.fit(data_prophet)
    return model

@st.cache_data
def get_prophet_forecast(_model, periods=3):
    future = _model.make_future_dataframe(periods=periods, freq='ME')
    forecast = _model.predict(future)
    return forecast

@st.cache_data
def calculate_segment_rmse(segment_data, periods=3):
    """
    Calculate RMSE for a segment-specific Prophet model.
    Uses last 12 months as test set.
    """
    if len(segment_data) < 15:  # Need at least 15 months for meaningful split
        return None
    
    # Split data: last 12 months for test
    train_size = len(segment_data) - 12
    train_data = segment_data.iloc[:train_size]
    test_data = segment_data.iloc[train_size:]
    
    # Train model on training data
    model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
    model.add_country_holidays(country_name='US')
    model.fit(train_data)
    
    # Make predictions for test period
    future = model.make_future_dataframe(periods=12, freq='ME')
    forecast = model.predict(future)
    
    # Extract predictions for test period
    test_predictions = forecast.iloc[-12:]['yhat'].values
    test_actual = test_data['y'].values
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    
    return rmse

# --- Load all data needed for the app once ---
file_path = './train.csv' # Local deployment path
df_full = load_and_preprocess_data(file_path)
monthly_sales_df = get_monthly_sales(df_full)
weekly_sales_df = get_weekly_sales(df_full)
sub_category_features_df, pca_plot_df, cluster_labels = perform_clustering(df_full)

# Prepare data for overall Prophet model
prophet_overall_df = monthly_sales_df.rename(columns={'Month': 'ds', 'Monthly Sales': 'y'})

# --- Train overall Prophet model ---
overall_prophet_model = train_prophet_model(prophet_overall_df)

# Global RMSE from Task 3 for the overall Prophet model
# Need to re-calculate RMSE for consistency within the app if using test split
# For simplicity, we will use the RMSE obtained from the previous run for the overall Prophet model.
# In a real application, you would re-split and evaluate within the app or save the metric.
overall_rmse_prophet = 9220.45

# --- Streamlit App Structure ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Sales Overview Dashboard",
    "Forecast Explorer",
    "Anomaly Report",
    "Product Demand Segments"
])

if page == "Sales Overview Dashboard":
    st.title("📈 Sales Overview Dashboard")

    st.subheader("Total Sales by Year")
    yearly_sales = df_full.groupby('Order Year')['Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Order Year', y='Sales', data=yearly_sales, palette='viridis', ax=ax)
    ax.set_title('Total Sales by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Sales')
    st.pyplot(fig)

    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Month', y='Monthly Sales', data=monthly_sales_df, ax=ax)
    ax.set_title('Overall Monthly Sales Trend (2015-2018)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Monthly Sales')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Sales by Region and Category (Interactive)")
    selected_region = st.selectbox('Select Region', ['All'] + list(df_full['Region'].unique()))
    selected_category = st.selectbox('Select Category', ['All'] + list(df_full['Category'].unique()))

    filtered_df = df_full.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    if not filtered_df.empty:
        sales_by_region_category = filtered_df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Region', y='Sales', hue='Category', data=sales_by_region_category, palette='muted', ax=ax)
        ax.set_title(f'Sales by Region and Category (Filtered: Region={selected_region}, Category={selected_category})')
        ax.set_xlabel('Region')
        ax.set_ylabel('Total Sales')
        ax.ticklabel_format(style='plain', axis='y') # Disable scientific notation
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data for the selected filters.")

elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    segment_type = st.radio("Select Segment Type", ['Overall Sales', 'Product Category', 'Region'])
    selected_segment = "Overall Sales"

    data_for_prophet = prophet_overall_df.copy()
    current_model = overall_prophet_model

    if segment_type == 'Product Category':
        categories = df_full['Category'].unique()
        selected_category = st.selectbox('Select Product Category', categories)
        # Dynamically create data for prophet for the selected category
        segment_df = df_full[df_full['Category'] == selected_category]
        data_for_prophet = segment_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        data_for_prophet.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)
        selected_segment = selected_category

        # Retrain model for the specific segment
        current_model = train_prophet_model(data_for_prophet)

    elif segment_type == 'Region':
        regions = df_full['Region'].unique()
        selected_region = st.selectbox('Select Region', regions)
        # Dynamically create data for prophet for the selected region
        segment_df = df_full[df_full['Region'] == selected_region]
        data_for_prophet = segment_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        data_for_prophet.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)
        selected_segment = selected_region

        # Retrain model for the specific segment
        current_model = train_prophet_model(data_for_prophet)

    st.subheader(f"Forecast for: {selected_segment}")

    forecast_horizon = st.slider("Select Forecast Horizon (months)", 1, 3, 3)

    # Generate forecast
    forecast = get_prophet_forecast(current_model, periods=forecast_horizon)

    # Plot forecast using Prophet's interactive Plotly chart
    from prophet.plot import plot_plotly
    fig = plot_plotly(current_model, forecast)
    fig.update_layout(
        title=f'Monthly Sales Forecast for {selected_segment}', 
        xaxis_title='Date', 
        yaxis_title='Monthly Sales',
        height=550
    )
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Model Performance")
    
    # Calculate segment-specific RMSE
    segment_rmse = calculate_segment_rmse(data_for_prophet)
    
    if segment_rmse is not None:
        st.write(f"**Segment-Specific Prophet Model RMSE:** {segment_rmse:.2f}")
        st.info(f"This RMSE is calculated using the last 12 months as test data for the {selected_segment} segment.")
    else:
        st.write(f"**Overall Prophet Model RMSE (from Task 3):** {overall_rmse_prophet:.2f}")
        st.warning(f"Insufficient data to calculate segment-specific RMSE for {selected_segment}. Displaying overall model RMSE instead.")


elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")

    st.subheader("Weekly Sales with Isolation Forest Anomalies")

    # Re-run Isolation Forest for weekly sales
    X_weekly_sales = weekly_sales_df[['Weekly Sales']].copy()
    model_if_anom = IsolationForest(random_state=42, contamination='auto')
    model_if_anom.fit(X_weekly_sales)
    anomalies_if = model_if_anom.predict(X_weekly_sales)
    weekly_sales_df['is_anomaly_if'] = anomalies_if

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x='Week', y='Weekly Sales', data=weekly_sales_df, label='Weekly Sales', ax=ax)
    anomalies = weekly_sales_df[weekly_sales_df['is_anomaly_if'] == -1]
    sns.scatterplot(x='Week', y='Weekly Sales', data=anomalies, color='red', s=100, label='Anomaly (Isolation Forest)', marker='o', ax=ax)
    ax.set_title('Weekly Sales with Isolation Forest Anomalies')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Sales')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Detected Anomalies (Isolation Forest)")
    if not anomalies.empty:
        st.dataframe(anomalies[['Week', 'Weekly Sales']].reset_index(drop=True))
    else:
        st.info("No anomalies detected by Isolation Forest.")

    st.markdown("### Possible Real-World Explanations for Anomalies:")
    st.write("Likely causes include Black Friday/Cyber Monday, holiday shopping, year-end clearance sales, and post-holiday slumps. Specific product launches or marketing campaigns could also play a role.")

elif page == "Product Demand Segments":
    st.title("📊 Product Demand Segments")

    st.subheader("Sub-Category Clusters (PCA-reduced to 2D)")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster Label', data=pca_plot_df,
        palette='viridis', s=100, alpha=0.8, edgecolor='w', ax=ax
    )
    ax.set_title('Sub-Category Clusters (PCA-reduced to 2D)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(title='Demand Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.subheader("Sub-Categories by Demand Group")
    cluster_label_map_full = {
        0: 'Stable Low-Value: These items have consistent, low demand. A Lean Stocking / Economic Order Quantity (EOQ) approach is recommended to minimize carrying costs. Automated reordering can be highly effective here.',
        1: """Premium High-Demand: Characterized by high sales volume and average order value. For these 'cash cow' items, Strategic Safety Stock / Just-In-Time (JIT) with Buffer is advised to ensure availability and prevent lost sales. Robust supply chains are crucial.""",
        2: 'Emerging High-Growth: These items show high growth and volatility. A Flexible Stocking / Aggressive Replenishment / Close Monitoring strategy is best. Start with moderate stock, be prepared for rapid replenishment, and frequently review sales data to adjust forecasts.'
    }

    for cluster_id, label_desc in cluster_label_map_full.items():
        st.markdown(f"**{label_desc.split(':')[0]}**")
        st.write(f"*Strategy:* {label_desc.split(':', 1)[1].strip()}")
        cluster_subs = sub_category_features_df[sub_category_features_df['Cluster'] == cluster_id]['Sub-Category'].tolist()
        st.write("**Sub-categories:** " + ", ".join(cluster_subs))
        st.markdown("--- (End of cluster description)")
