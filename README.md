# Crime Forecasting and Prediction Using Spatio-Temporal Techniques  

## Conference Presentation  
This paper was **presented at IJCACI Conference (2025)**.  
You can view the official **Letter of Presentation** here:  

[View Certificate / Letter](https://drive.google.com/file/d/1XYXlZ1BT14ul9mM8bb_gdw2H_n0XF1ud/view)  

## Overview  
This repository contains the implementation of **Crime Forecasting and Prediction Using Spatio-Temporal Techniques**.  
We apply **machine learning, deep learning, and graph neural networks (GNNs)** to forecast urban crime patterns across **time and space** using **Los Angeles crime data (2020–2025)**.  

---

## Methodology  

### Dataset and Preprocessing  

This study uses a **comprehensive dataset of over 1 million crime reports** from **Los Angeles (2020–2025)**, sourced from the [City of Los Angeles Open Data Portal](https://catalog.data.gov/dataset/crime-data-from-2020-to-present).  
The dataset was chosen because it captures **post-COVID crime dynamics**, reflecting **social behavior shifts, law enforcement compliance, and economic conditions**.   

**Preprocessing Steps:**  
- **Datetime Conversion** – Converted `DATE OCC` and `TIME OCC` into Python datetime format, extracting **hour, day, month, year, weekday** for temporal granularity.  
- **Time Binning** – Crimes grouped into **6 daily intervals**:  
  - Late Night (12am–3am)  
  - Early Morning (3am–6am)  
  - Morning (6am–12pm)  
  - Afternoon (12pm–5pm)  
  - Evening (5pm–9pm)  
  - Night (9pm–12am)  
- **Additional Features** –  
  - Day of Week  
  - Weekend Flag  
  - Public Holiday Indicator (via `holidays` Python library)  
- **Crime Category Consolidation** – Reduced **68 unique crime descriptions** into **17 higher-level categories** for clearer pattern recognition (e.g., burglary, theft, assault grouped).  
- **Scaling/Normalization** – For deep learning models (LSTM), features were normalized with **MinMaxScaler**.  

---

### Models Implemented  

#### Classification Models (Crime Solvability)  
**Objective:** Predict whether a reported crime will be **resolved or unsolved** based on historical features.  

- **Decision Tree (DT):** Baseline model → accuracy = **86%**  
- **K-Nearest Neighbors (KNN):** Captured neighborhood similarities → accuracy = **90%**  
- **Random Forest (RF):**  
  - RF (10 trees) → **91%**  
  - RF (20 trees) → **92% (Best)**  

**Key Takeaway:** Random Forest (20 trees) was most effective due to its ability to capture **complex decision boundaries** in noisy urban crime data.  

---

#### 2 Temporal Forecasting Models  
**Objective:** Forecast **crime frequency trends** over short- and long-term horizons.  

- **ARIMA + Prophet:**  
  - ARIMA captures **short-term autoregressive patterns**  
  - Prophet incorporates **seasonality, trend changes, holidays**  
  - Hybrid improved both short and long forecasts  

- **Prophet + ARIMA:**  
  - Prophet first captures **long-term seasonalities**  
  - ARIMA refines predictions with short-term fluctuations  
  - Outperformed ARIMA-first hybrid  

- **RNN + LSTM:**  
  - Bidirectional LSTM with **look-back window of 12 weeks**  
  - Trained for **daily, weekly, monthly horizons**  
  - Used dropout + early stopping to reduce overfitting  
  - **Best RMSEs:**  
    - Daily → **0.0585**  
    - Weekly → **0.06275**  
    - Monthly → **0.1339**  

**Key Takeaway:** **RNN+LSTM** significantly outperformed classical models, especially in capturing **non-linear sequential patterns**.  

---

#### Spatial Forecasting Models (Graph Neural Networks)  
**Objective:** Model **spatial spillover effects** across 21 LA regions. Each region is represented as a **node**, with edges defined by **geographic proximity**.  

- **ST-GCN (Spatio-Temporal Graph Convolutional Network):**  
  - Combines **GCN layers (spatial)** + **LSTM (temporal)**  
  - Best performance → **RMSE: 244.78**  

- **GAT (Graph Attention Network):**  
  - Applies **attention mechanism** to weigh neighbors dynamically  
  - RMSE = **251.59** (slightly worse due to noise sensitivity)  

- **STA-GNN (Spatio-Temporal Attention GNN):**  
  - Dual attention → **spatial + temporal**  
  - RMSE = **252.88** (did not outperform ST-GCN)  

**Key Takeaway:** **ST-GCN** proved most effective, striking the best balance between **spatial awareness** and **temporal forecasting**.  

---

## Results  

| Task                 | Best Model  | Metric   | Score   |
|----------------------|------------|----------|---------|
| Crime Solvability    | RandomForest | Accuracy | **92%** |
| Temporal Forecasting | RNN+LSTM     | RMSE (daily) | **0.0585** |
| Spatial Forecasting  | ST-GCN       | RMSE    | **244.78** |

---
