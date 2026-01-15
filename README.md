# 🥇 AI-Powered Gold Price Trend Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Module E: AI Applications – Individual Open Project**

## 📌 Overview

An AI-powered system leveraging **LSTM (Long Short-Term Memory) neural networks** to analyze historical gold price data and predict future price movements. This project demonstrates the application of deep learning techniques for financial time series forecasting.

## 🎯 Problem Statement

Gold prices are influenced by complex, non-linear factors including geopolitical events, inflation rates, interest rates, and market sentiment. Traditional statistical models struggle to capture these intricate patterns. This project aims to:

1. **Analyze** 12+ years of historical gold price data (2013-2025)
2. **Extract** meaningful patterns through technical indicators
3. **Predict** next-day closing prices with high accuracy
4. **Provide** actionable insights for investors and analysts

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [Kaggle - Gold and Silver Prices (2013-2025)](https://www.kaggle.com/datasets/kapturovalexander/gold-and-silver-prices-2013-2023) |
| **Time Period** | January 2013 – Present |
| **Frequency** | Daily OHLCV data |
| **Records** | 3000+ trading days |

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow/Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly, Seaborn |
| Environment | Jupyter Notebook |

## 🧠 Model Architecture

```
Input (OHLCV + Technical Indicators)
        │
   LSTM Layer 1 (128 units) + Dropout
        │
   LSTM Layer 2 (64 units) + Dropout
        │
   Dense Layer (32 units, ReLU)
        │
   Output Layer (1 unit - Predicted Price)
```

## 📁 Project Structure

```
AI-For-Gold-Trend-Analysis/
├── 📓 Gold_Price_Analysis.ipynb   # Main notebook (Primary artifact)
├── 📄 PROJECT_DETAILS.md          # Detailed documentation
├── 📄 README.md                   # This file
├── 📁 data/                       # Dataset files
├── 📁 models/                     # Saved models
└── 📁 images/                     # Visualizations
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn plotly scikit-learn
```

### Run the Notebook
1. Clone the repository
2. Download the dataset from Kaggle
3. Open `Gold_Price_Analysis.ipynb` in Jupyter
4. Run all cells sequentially

## 📈 Expected Results

- **MAPE:** < 2%
- **R² Score:** > 0.90
- Accurate trend direction predictions

## ⚠️ Disclaimer

This project is for **educational purposes only**. Predictions should not be used as financial advice. Always consult qualified financial advisors before making investment decisions.

## 📄 Documentation

For detailed project information, see [PROJECT_DETAILS.md](PROJECT_DETAILS.md)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Module E: AI Applications | Individual Open Project | January 2026*