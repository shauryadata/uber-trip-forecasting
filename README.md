# 🚖 Uber Trip Forecasting - Machine Learning + Time Series Modeling

This project analyzes and forecasts hourly Uber trip demand in New York City using a combination of machine learning and time series techniques. It features an interactive Streamlit dashboard to explore patterns, generate forecasts, and compare models.

---

## 🚀 Live App  
🔗 [Streamlit Dashboard](https://uber-trip-forecasting-ewcnitrzgihwss84tyvkhx.streamlit.app/)

---

## 🎯 Objective  
- Forecast hourly Uber trip demand  
- Compare machine learning and time series forecasting models  
- Build a real-time interactive app for data exploration and prediction

---

## 🗂️ Dataset  
- NYC Uber pickups (April–September 2014)  
- ~4.5 million trip records cleaned and grouped into hourly counts

---

## 📈 EDA Highlights  
- **Evening peaks**: Demand surges from 5 PM to 8 PM  
- **Weekday vs weekend**: Fridays and Saturdays show highest trip counts  
- **Heatmap patterns**: Strong cyclic behavior by hour and weekday

---

## 🧠 Models Used  

### 1. **XGBoost Regressor**  
- Lag features used to predict future demand  
- Suitable for capturing short-term fluctuations  

### 2. **Facebook Prophet**  
- Automatically models trend, seasonality, and holidays  
- Ideal for long-term pattern recognition  

---

## 🧪 Model Comparison

| Model     | MAE     | RMSE    | MAPE     |
|-----------|---------|---------|----------|
| XGBoost   | 138.08  | 206.95  | 11.19%   |
| Prophet   | 349.24  | 443.37  | 45.43%   |

- **XGBoost** outperforms Prophet for short-term, high-frequency predictions  
- **Prophet** is better for explaining long-term trends  

---

## 🛠️ Tools & Technologies  
- Python, Pandas, Matplotlib, Seaborn  
- XGBoost, Facebook Prophet  
- Streamlit for deployment

---

## 📁 Files Included  
- `Uber_Analysis.ipynb` – Data cleaning, feature engineering, EDA  
- `forecast_xgboost.py` – Lag-based forecasting  
- `forecast_prophet.py` – Prophet model setup  
- `app.py` – Streamlit app  
- `uber_hourly_trip_data.csv` – Cleaned hourly trip data

---

## ✅ Author  
**Shauryaditya Singh**  
Aspiring ML Engineer | Forecasting & Real-Time Dashboards

---

## 📌 Future Enhancements  
- Integrate weather or holiday data for demand spikes  
- Include anomaly detection for outlier analysis  
- Build multi-city or multi-service comparison view
