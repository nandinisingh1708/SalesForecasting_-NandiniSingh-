import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# LOAD DATA

df = pd.read_csv("train.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Monthly aggregation
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.set_index('Order Date')

monthly_sales = df['Sales'].resample('ME').sum()


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Sales Overview",
    "Forecast Explorer",
    "Anomaly Report",
    "Product Segments"
])

# PAGE 1 — SALES OVERVIEW

if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    # Total sales by year
    yearly_sales = df.groupby('Year')['Sales'].sum()

    st.subheader("Total Sales by Year")
    st.bar_chart(yearly_sales)

    # Monthly trend
    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly_sales)

    # Filters
    st.subheader("Sales by Region & Category")

    region = st.selectbox("Select Region", df['Region'].unique())
    category = st.selectbox("Select Category", df['Category'].unique())

    filtered = df[(df['Region'] == region) & (df['Category'] == category)]

    st.write("Filtered Sales Trend")
    st.line_chart(filtered.groupby('Order Date')['Sales'].sum().resample('M').sum())


# PAGE 2 — FORECAST EXPLORER

elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    steps = st.slider("Forecast Horizon (Months)", 1, 3, 3)

    # SARIMA MODEL (best model assumed)
    train = monthly_sales[:-3]

    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit()

    forecast = result.get_forecast(steps=steps)
    pred = forecast.predicted_mean

    st.subheader("Forecast Output")
    st.line_chart(pd.concat([train, pred]))

    # Accuracy (simple evaluation)
    test = monthly_sales[-3:]

    mae = mean_absolute_error(test, pred[:len(test)])
    rmse = np.sqrt(mean_squared_error(test, pred[:len(test)]))

    st.write("MAE:", mae)
    st.write("RMSE:", rmse)

# PAGE 3 — ANOMALY REPORT

elif page == "Anomaly Report":
    st.title("🚨 Anomaly Detection Report")

    weekly_sales = df.groupby('Order Date')['Sales'].sum().resample('W').sum()

    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso.fit_predict(weekly_sales.values.reshape(-1,1))

    anomaly_df = pd.DataFrame({
        "Sales": weekly_sales,
        "Anomaly": anomalies
    })

    st.subheader("Anomaly Chart")

    fig, ax = plt.subplots()
    ax.plot(anomaly_df.index, anomaly_df['Sales'])
    ax.scatter(anomaly_df[anomaly_df['Anomaly'] == -1].index,
               anomaly_df[anomaly_df['Anomaly'] == -1]['Sales'],
               color='red')
    st.pyplot(fig)

    st.subheader("Anomaly Table")
    st.dataframe(anomaly_df[anomaly_df['Anomaly'] == -1])

# PAGE 4 — PRODUCT SEGMENTS

elif page == "Product Segments":
    st.title("📦 Product Demand Segmentation")

    cluster_data = df.groupby('Sub-Category')['Sales'].sum().reset_index()

    from sklearn.cluster import KMeans

    X = cluster_data[['Sales']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(X)

    st.subheader("Clustered Sub-Categories")
    st.dataframe(cluster_data)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(cluster_data['Sub-Category'], cluster_data['Sales'],
               c=cluster_data['Cluster'])
    plt.xticks(rotation=90)
    st.pyplot(fig)
