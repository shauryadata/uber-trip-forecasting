import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(layout="wide")

st.title("🚖 Uber Trip Forecasting Dashboard")

# Load pre-aggregated hourly trip data (make sure this file is in the same repo)
data = pd.read_csv("uber_hourly_trip_data.csv", parse_dates=['Datetime_Hour'])
data = data.rename(columns={"Datetime_Hour": "ds", "Trip_Count": "y"})

st.sidebar.header("Choose View")
view = st.sidebar.radio("Select an option:", ["EDA", "Prophet Forecast", "Model Evaluation"])

if view == "EDA":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(data.set_index("ds")["y"].resample("D").sum())
        st.caption("Total trips per day")

    with col2:
        data['Hour'] = data['ds'].dt.hour
        data['Weekday'] = data['ds'].dt.dayofweek
        heatmap_data = data.groupby(['Weekday', 'Hour'])['y'].mean().unstack()

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto')
        ax.set_title("Average Trips by Hour & Weekday")
        ax.set_ylabel("Weekday (0=Mon)")
        ax.set_xlabel("Hour")
        plt.colorbar(cax)
        st.pyplot(fig)

elif view == "Prophet Forecast":
    st.subheader("🔮 Forecast Future Uber Trips (Prophet)")

    periods = st.slider("Select hours to forecast:", 24, 168, 48)
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

elif view == "Model Evaluation":
    st.subheader("⚖️ Model Comparison")
    st.markdown("""
    - **XGBoost MAPE**: 11.19%  
    - **Prophet MAPE**: 45.43%  
    
    XGBoost captures high-frequency demand shifts better. Prophet provides interpretable trends.
    """)

    st.image("comparison_xgboost_only.png", caption="XGBoost: Actual vs Predicted (First 100 Hours)")

