# 📊 Chartbot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Supported-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Chartbot** is an open-source Python project by **Profit Adda Web** for market data analysis and visualization.  
It fetches live or historical data, computes technical indicators, and generates interactive dashboards to assist in trading and market research for more info visit https://www.profitaddaweb.com

---

## ✨ Features

- 📡 **Market Data Feed** – fetches live or historical market data via `marketfeed.py`.  
- 📈 **Technical Indicators** – compute metrics such as RSI, Moving Averages, and more with `indicators.py`.  
- 📊 **Dashboard & Charts** – visualize insights through `dashboard.py` with ready-to-use templates.  
- 📂 **CSV Data Support** – run tests or backtest strategies with example datasets in the `CSV/` folder.  
- ⚙️ **Modular Architecture** – easy to extend with new data sources, indicators, or charting styles.

---

## 📂 Project Structure

```bash
Chartbot/
│── marketfeed.py      # Fetches live/historical market data
│── indicators.py      # Implements trading indicators
│── dashboard.py       # Builds dashboard visualizations
│── requirements.txt   # Python dependencies
│── CSV/               # Sample/historical data files
│── templates/         # HTML templates for dashboard
│── README.md          # Project documentation

