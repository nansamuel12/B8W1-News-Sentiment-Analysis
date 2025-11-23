# B8W1-News-Sentiment-Analysis
**Predicting Price Moves with News Sentiment (Week 1)**

A comprehensive financial analytics project that analyzes the correlation between news sentiment and stock price movements using advanced NLP techniques and technical analysis.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Methodology](#methodology)
- [Results & KPIs](#results--kpis)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This project is part of Nova Financial Solutions' initiative to enhance predictive analytics capabilities for financial forecasting. We analyze financial news headlines and their impact on stock price movements for major tech companies (AAPL, AMZN, GOOG, META, MSFT, NVDA).

### What This Project Does:
1. **Sentiment Analysis**: Quantifies the emotional tone of financial news headlines
2. **Technical Analysis**: Calculates key technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. **Correlation Analysis**: Measures the statistical relationship between news sentiment and stock returns
4. **Data Visualization**: Provides comprehensive visual insights into market trends

---

## 🎯 Business Objective

**Nova Financial Solutions** aims to:
- Enhance predictive analytics capabilities
- Boost financial forecasting accuracy
- Improve operational efficiency through advanced data analysis

**Primary Task**: Conduct rigorous analysis of financial news datasets with a two-fold focus:
1. **Sentiment Analysis**: Quantify tone and sentiment in financial news using NLP
2. **Correlation Analysis**: Associate sentiment scores with stock symbols to understand emotional context

---

## ✨ Features

### 1. **Exploratory Data Analysis (EDA)**
- Headline length analysis
- Publisher distribution analysis
- Date-based article distribution
- Keyword frequency analysis

### 2. **Technical Analysis (Task 2)**
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility and price range analysis
- **Additional Indicators**: ATR, Stochastic Oscillator, OBV
- **Financial Metrics**: Daily returns, volatility, Sharpe ratio, cumulative returns

### 3. **News-Stock Correlation Analysis (Task 3)**
- **Dual Sentiment Analysis**: TextBlob + VADER
- **Date Normalization**: Align news and stock trading days
- **Daily Aggregation**: Average sentiment scores per day
- **Pearson Correlation**: Statistical correlation between sentiment and returns
- **Lagged Correlation**: Analyze delayed effects (0-5 days)
- **Significance Testing**: P-values for statistical validation

---

## 📁 Project Structure

```
Predicting_Price_Moves/
│
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD configuration
│
├── data/
│   ├── Data/                      # Stock price CSVs
│   │   ├── AAPL.csv
│   │   ├── AMZN.csv
│   │   ├── GOOG.csv
│   │   ├── META.csv
│   │   ├── MSFT.csv
│   │   └── NVDA.csv
│   └── raw_analyst_ratings.csv    # News headlines dataset (311 MB)
│
├── notebooks/
│   ├── EDA.ipynb                           # Exploratory Data Analysis
│   ├── Task2_Technical_Analysis.ipynb      # Technical indicators & metrics
│   ├── Task3_News_Stock_Correlation.ipynb  # Full correlation analysis
│   └── Task3_Enhanced_Correlation.ipynb    # Streamlined correlation workflow
│
├── scripts/
│   └── README.md                  # Scripts documentation
│
├── src/
│   └── __init__.py               # Source code modules
│
├── tests/
│   └── __init__.py               # Unit tests
│
├── .gitignore                    # Git ignore rules
├── .vscode/
│   └── settings.json             # VS Code settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository
```bash
git clone https://github.com/nansamuel12/B8W1-News-Sentiment-Analysis.git
cd B8W1-News-Sentiment-Analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install TA-Lib (Special Installation)
**Windows:**
```bash
# Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

### Step 5: Download NLTK Data (for TextBlob)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

---

## 📖 Usage Guide

### Step-by-Step Workflow

#### **Step 1: Exploratory Data Analysis**
```bash
jupyter notebook notebooks/EDA.ipynb
```

**What it does:**
- Loads news dataset
- Analyzes headline characteristics
- Identifies top publishers
- Visualizes article distribution over time
- Extracts keyword frequencies

**Output:**
- Statistical summaries
- Distribution plots
- Time-series visualizations

---

#### **Step 2: Technical Analysis**
```bash
jupyter notebook notebooks/Task2_Technical_Analysis.ipynb
```

**What it does:**
1. **Load Stock Data**: Reads CSV files for 6 tech stocks
2. **Calculate Indicators**:
   - Moving averages (SMA 20/50/200, EMA 12/26)
   - RSI (14-period)
   - MACD (12, 26, 9)
   - Bollinger Bands (20-period, 2 std)
   - ATR, Stochastic, OBV
3. **Compute Financial Metrics**:
   - Daily returns (percentage change)
   - Volatility (rolling 20-day)
   - Sharpe ratio
   - Cumulative returns
4. **Visualize**:
   - Price charts with moving averages
   - RSI with overbought/oversold zones
   - MACD histograms
   - Bollinger Bands
   - Volume and OBV
   - Comparative analysis across stocks

**Output:**
- Technical indicator values
- Summary statistics table
- 15+ visualization charts

---

#### **Step 3: News-Stock Correlation Analysis**

**Option A: Full Analysis**
```bash
jupyter notebook notebooks/Task3_News_Stock_Correlation.ipynb
```

**Option B: Enhanced Streamlined Analysis**
```bash
jupyter notebook notebooks/Task3_Enhanced_Correlation.ipynb
```

**What it does:**

**Phase 1: Data Preparation**
1. Load news headlines (raw_analyst_ratings.csv)
2. Load stock price data for 6 tickers
3. Normalize dates (remove timezone, align to trading days)
4. Handle missing values

**Phase 2: Sentiment Analysis**
1. Apply **TextBlob** sentiment analysis (polarity: -1 to 1)
2. Apply **VADER** sentiment analysis (compound: -1 to 1)
3. Calculate average sentiment score
4. Categorize as Positive/Neutral/Negative
5. Visualize sentiment distribution

**Phase 3: Calculate Stock Movements**
1. Compute daily returns: `(Close_today - Close_yesterday) / Close_yesterday * 100`
2. Calculate log returns
3. Determine price direction (Up/Down/Flat)
4. Visualize return distributions

**Phase 4: Aggregate Sentiments**
1. Group news by trading date
2. Calculate mean, std, min, max sentiment per day
3. Count articles per day
4. Create stock-specific sentiment aggregations
5. Visualize daily sentiment trends

**Phase 5: Merge & Correlate**
1. Merge sentiment data with stock returns by date
2. Calculate **Pearson correlation coefficient**
3. Compute p-values for significance testing
4. Test lagged correlations (0-5 days)
5. Generate scatter plots with regression lines

**Output:**
- Correlation coefficients for each stock
- P-values and significance levels
- Scatter plots showing sentiment vs returns
- Lagged correlation analysis
- Summary statistics and insights

---

## 🔬 Methodology

### 1. Data Collection
- **News Data**: 311 MB CSV with headlines, dates, publishers, stock symbols
- **Stock Data**: Daily OHLCV (Open, High, Low, Close, Volume) for 6 stocks
- **Time Period**: 2009-2020+ (varies by stock)

### 2. Sentiment Analysis Approach
- **TextBlob**: Rule-based sentiment using lexicon
- **VADER**: Specialized for social media and news text
- **Ensemble**: Average of both methods for robustness

### 3. Technical Analysis Framework
- **Trend Indicators**: Moving averages (SMA, EMA)
- **Momentum Indicators**: RSI, Stochastic
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV

### 4. Statistical Analysis
- **Pearson Correlation**: Measures linear relationship
- **Spearman Correlation**: Measures monotonic relationship
- **Significance Testing**: P-values < 0.05 considered significant
- **Lagged Analysis**: Tests predictive power of sentiment

---

## 📊 Results & KPIs

### Key Performance Indicators

#### **1. Sentiment Analysis Quality**
- ✅ Dual-method validation (TextBlob + VADER)
- ✅ Sentiment score range: -1 (negative) to +1 (positive)
- ✅ Category distribution: Positive/Neutral/Negative

#### **2. Correlation Strength**
- Pearson correlation coefficients for each stock
- P-values indicating statistical significance
- Sample sizes for reliability assessment

#### **3. Technical Indicator Accuracy**
- RSI: Identifies overbought (>70) and oversold (<30) conditions
- MACD: Signals bullish/bearish crossovers
- Bollinger Bands: Captures 95% of price movements

#### **4. Data Coverage**
- News articles analyzed: 1,000,000+ headlines
- Trading days covered: 2,500+ days
- Stocks analyzed: 6 major tech companies

---

## 🛠️ Technologies Used

### Programming & Data Analysis
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Natural Language Processing
- **TextBlob**: Sentiment analysis
- **VADER (vaderSentiment)**: Social media sentiment analysis
- **NLTK**: Natural language toolkit

### Technical Analysis
- **TA-Lib**: Technical analysis library

### Statistical Analysis
- **SciPy**: Statistical functions and tests
- **Statsmodels**: Advanced statistical modeling

### Visualization
- **Matplotlib**: Core plotting library
- **Seaborn**: Statistical data visualization

### Development Tools
- **Jupyter Notebook**: Interactive development
- **Git**: Version control
- **GitHub**: Repository hosting

---

## 📈 How to Interpret Results

### Correlation Coefficients
- **r > 0.3**: Moderate positive correlation (sentiment ↑ → returns ↑)
- **r < -0.3**: Moderate negative correlation (sentiment ↑ → returns ↓)
- **-0.3 < r < 0.3**: Weak or no correlation

### P-Values
- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)
- **p ≥ 0.05**: Not significant (ns)

### Technical Indicators
- **RSI > 70**: Overbought (potential sell signal)
- **RSI < 30**: Oversold (potential buy signal)
- **MACD > Signal**: Bullish trend
- **MACD < Signal**: Bearish trend

---

## 🔄 Workflow Summary

```
1. DATA LOADING
   ├── Load news headlines (CSV)
   └── Load stock prices (6 CSVs)
   
2. DATA PREPARATION
   ├── Normalize dates
   ├── Remove timezone info
   └── Align to trading days
   
3. SENTIMENT ANALYSIS
   ├── Apply TextBlob
   ├── Apply VADER
   └── Calculate average sentiment
   
4. TECHNICAL ANALYSIS
   ├── Calculate indicators (RSI, MACD, etc.)
   ├── Compute financial metrics
   └── Generate visualizations
   
5. AGGREGATION
   ├── Group by trading date
   ├── Calculate daily averages
   └── Handle multiple articles per day
   
6. CORRELATION ANALYSIS
   ├── Merge sentiment + returns
   ├── Calculate Pearson correlation
   ├── Test significance (p-values)
   └── Analyze lagged effects
   
7. VISUALIZATION & REPORTING
   ├── Scatter plots
   ├── Time-series charts
   └── Summary statistics
```

---

## 🌿 Git Branches

- **main**: Production-ready code
- **task-1**: Initial setup and EDA
- **task-2**: Technical analysis implementation
- **task-3**: Enhanced correlation analysis

---

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the 10 Academy training program.

---

## 👥 Authors

- **Data Analyst**: Nova Financial Solutions Team
- **Repository**: [nansamuel12](https://github.com/nansamuel12)

---

## 🙏 Acknowledgments

- **10 Academy**: Training and project guidance
- **Nova Financial Solutions**: Business requirements and objectives
- **Open Source Community**: Libraries and tools (TA-Lib, TextBlob, VADER)

---

## 📞 Contact & Support

For questions or support:
- Open an issue on GitHub
- Contact: [Your Email]

---

## 🔮 Future Enhancements

- [ ] Real-time sentiment analysis dashboard
- [ ] Machine learning models for price prediction
- [ ] Integration with live news APIs
- [ ] Extended stock universe (more tickers)
- [ ] Advanced NLP models (BERT, FinBERT)
- [ ] Automated trading signal generation

---

**Last Updated**: November 22, 2025