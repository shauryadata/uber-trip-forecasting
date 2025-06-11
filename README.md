# 🚖 Uber Trip Forecasting with Machine Learning & Prophet

This project analyzes and forecasts hourly Uber trip demand in NYC using machine learning (XGBoost) and time series modeling (Facebook Prophet). An interactive Streamlit dashboard lets users explore trends and compare models.

🔗 **Live Demo**: [Streamlit App](https://uber-trip-forecasting-ewcnitrzgihwss84tyvkhx.streamlit.app/)

---

## Project Overview

* **Data**: NYC Uber pickup logs from April–September 2014 (\~4.5 million records)
* **Goal**: Forecast hourly Uber trip demand using time series models
* **Models Used**:

  * XGBoost Regressor with lag features
  * Facebook Prophet (trend & seasonality)
* **Interface**: Streamlit dashboard for exploration, forecasting, and comparison

---

## 🔍 Key Insights

* **Evening peaks**: Most Uber activity occurs between 5–8 PM
* **Weekly trend**: Fridays and Saturdays have the highest demand
* **Modeling takeaway**: XGBoost outperforms Prophet on high-frequency hourly data

---

## Model Evaluation

| Model   | MAE    | RMSE   | MAPE       |
| ------- | ------ | ------ | ---------- |
| XGBoost | 138.08 | 206.95 | **11.19%** |
| Prophet | 349.24 | 443.37 | 45.43%     |

* **XGBoost** captures short-term fluctuations with lag features
* **Prophet** excels at interpreting trends + weekly/daily cycles

---

## Streamlit App Features

* EDA Viewer: Hourly trends + heatmaps
* Prophet Forecasting: User-selectable future periods
* Model Comparison: Metric table + forecast chart overlays

---

## Contact

Feel free to connect via GitHub or LinkedIn for feedback or collaboration!

---

> Made by [@shauryadata](https://github.com/shauryadata)
