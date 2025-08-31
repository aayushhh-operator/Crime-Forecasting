# Crime Forecasting and Prediction Using Spatio-Temporal Techniques  

## Overview  
This repository contains the implementation of **Crime Forecasting and Prediction Using Spatio-Temporal Techniques**.  
We apply **machine learning, deep learning, and graph neural networks (GNNs)** to forecast urban crime patterns across **time and space** using **Los Angeles crime data (2020–2025)**.  

---

## Methodology  

### Dataset  
- **Source**: [City of Los Angeles Open Data Portal](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)  
- **Records**: 1M+ (2020–2025)  
- **Features**: Date, Time, Location (lat/lon), Crime Type  
- **Engineering**: Holiday flags, weekend/weekday, temporal bins, grouped crime categories  

### Models  
- **Classification** → Random Forest (92% accuracy), KNN, Decision Tree  
- **Temporal Forecasting** → ARIMA + Prophet, Prophet + ARIMA, RNN + LSTM (best RMSE daily: 0.0585)  
- **Spatial Forecasting** → ST-GCN (best RMSE: 244.78), GAT, STA-GNN  

---

## Results  

| Task                 | Best Model  | Metric   | Score   |
|----------------------|------------|----------|---------|
| Crime Solvability    | RandomForest | Accuracy | **92%** |
| Temporal Forecasting | RNN+LSTM     | RMSE (daily) | **0.0585** |
| Spatial Forecasting  | ST-GCN       | RMSE    | **244.78** |

---
