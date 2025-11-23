# Project Architecture

## Modular OOP Design

This project follows a clean, modular architecture with well-defined classes and clear APIs.

---

## 📦 Module Structure

```
src/
├── __init__.py                  # Package initialization with exposed APIs
├── sentiment_analyzer.py        # SentimentAnalyzer class
├── technical_analyzer.py        # TAProcessor class
├── data_loader.py              # DataLoader & DataPreprocessor classes
└── correlation_analyzer.py     # CorrelationAnalyzer class

tests/
├── __init__.py
├── test_sentiment_analyzer.py   # 15+ unit tests
├── test_technical_analyzer.py   # 20+ unit tests
├── test_data_loader.py         # 15+ unit tests
└── test_correlation_analyzer.py # 15+ unit tests
```

---

## 🎯 Core Classes

### 1. **SentimentAnalyzer**
**Purpose**: Perform sentiment analysis on text data

**Key Methods**:
- `analyze(text, method)` - Analyze single text
- `analyze_batch(texts)` - Batch analysis
- `analyze_dataframe(df, text_column)` - DataFrame integration
- `categorize_sentiment(score)` - Categorize as Positive/Neutral/Negative
- `get_sentiment_stats(scores)` - Statistical summary

**Features**:
- Dual-method support (TextBlob + VADER)
- Ensemble scoring
- Configurable thresholds
- Error handling

**Example Usage**:
```python
from src import SentimentAnalyzer

analyzer = SentimentAnalyzer(method='ensemble')
score = analyzer.analyze("Stock prices surge to new highs!")
category = analyzer.categorize_sentiment(score)
```

---

### 2. **TAProcessor**
**Purpose**: Calculate technical indicators and financial metrics using TA-Lib

**Key Methods**:
- `calculate_sma(data, period)` - Simple Moving Average
- `calculate_ema(data, period)` - Exponential Moving Average
- `calculate_rsi(data, period)` - Relative Strength Index
- `calculate_macd(data, fast, slow, signal)` - MACD
- `calculate_bollinger_bands(data, period, std)` - Bollinger Bands
- `calculate_atr(high, low, close, period)` - Average True Range
- `calculate_stochastic(high, low, close)` - Stochastic Oscillator
- `calculate_obv(close, volume)` - On-Balance Volume
- `calculate_daily_return(prices)` - Daily returns
- `calculate_volatility(returns, window)` - Rolling volatility
- `calculate_sharpe_ratio(returns, risk_free_rate)` - Sharpe ratio
- `process_stock_data(df)` - Apply all indicators at once
- `get_indicator_summary(df)` - Summary statistics

**Features**:
- TA-Lib integration
- 15+ technical indicators
- Financial metrics calculation
- Batch processing support

**Example Usage**:
```python
from src import TAProcessor

processor = TAProcessor()
processed_df = processor.process_stock_data(stock_df)
summary = processor.get_indicator_summary(processed_df)
```

---

### 3. **DataLoader**
**Purpose**: Load financial data from CSV files

**Key Methods**:
- `load_news_data(filename)` - Load news headlines
- `load_stock_data(ticker, subdir)` - Load single stock
- `load_multiple_stocks(tickers, subdir)` - Load multiple stocks

**Features**:
- Flexible path configuration
- Error handling for missing files
- Multi-ticker support

**Example Usage**:
```python
from src import DataLoader

loader = DataLoader('../data')
news_df = loader.load_news_data()
stock_data = loader.load_multiple_stocks(['AAPL', 'MSFT'])
```

---

### 4. **DataPreprocessor**
**Purpose**: Preprocess and transform financial data

**Key Methods**:
- `normalize_dates(df, date_column)` - Normalize dates to trading days
- `handle_missing_values(df, strategy)` - Handle NaN values
- `align_dates(df1, df2, date_col)` - Align two datasets by date
- `aggregate_by_date(df, date_col, agg_dict)` - Date-based aggregation
- `add_time_features(df, date_column)` - Add temporal features
- `merge_sentiment_stock(sentiment_df, stock_df)` - Merge datasets
- `create_lagged_features(df, columns, lags)` - Create lag features

**Features**:
- Timezone handling
- Multiple aggregation strategies
- Feature engineering
- Data alignment utilities

**Example Usage**:
```python
from src import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.normalize_dates(df, 'date')
df = preprocessor.add_time_features(df, 'date')
merged = preprocessor.merge_sentiment_stock(sentiment_df, stock_df)
```

---

### 5. **CorrelationAnalyzer**
**Purpose**: Analyze statistical correlations

**Key Methods**:
- `calculate_pearson(x, y)` - Pearson correlation
- `calculate_spearman(x, y)` - Spearman correlation
- `analyze_correlation(df, x_col, y_col, method)` - Full analysis
- `lagged_correlation(df, x_col, y_col, max_lag)` - Lagged analysis
- `batch_correlation_analysis(data_dict, x_col, y_col)` - Multiple datasets
- `plot_correlation_scatter(df, x_col, y_col)` - Visualization
- `plot_correlation_heatmap(df, columns)` - Heatmap
- `get_correlation_strength(corr)` - Categorize strength

**Features**:
- Multiple correlation methods
- Statistical significance testing
- Lagged correlation support
- Visualization tools
- Batch processing

**Example Usage**:
```python
from src import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
result = analyzer.analyze_correlation(df, 'sentiment', 'returns')
lagged = analyzer.lagged_correlation(df, 'sentiment', 'returns', max_lag=5)
```

---

## 🧪 Unit Tests

### Test Coverage

**test_sentiment_analyzer.py** (15 tests):
- Initialization tests
- TextBlob analysis tests
- VADER analysis tests
- Ensemble method tests
- Categorization tests
- Batch processing tests
- DataFrame integration tests
- Statistics calculation tests
- Error handling tests

**test_technical_analyzer.py** (20 tests):
- SMA/EMA calculation tests
- RSI calculation tests
- MACD calculation tests
- Bollinger Bands tests
- ATR calculation tests
- Stochastic Oscillator tests
- OBV calculation tests
- Return calculation tests
- Volatility tests
- Sharpe ratio tests
- Full processing pipeline tests
- Summary generation tests

**test_data_loader.py** (15 tests):
- Data loading tests
- Date normalization tests
- Missing value handling tests
- Date alignment tests
- Aggregation tests
- Time feature tests
- Merge operation tests
- Lagged feature tests

**test_correlation_analyzer.py** (15 tests):
- Pearson correlation tests
- Spearman correlation tests
- Significance testing
- Lagged correlation tests
- Batch analysis tests
- Strength categorization tests
- NaN handling tests
- Edge case tests

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_sentiment_analyzer

# Run with verbose output
python -m unittest discover tests -v
```

---

## 📚 API Documentation

### Package Import

```python
# Import all classes
from src import (
    SentimentAnalyzer,
    TAProcessor,
    DataLoader,
    DataPreprocessor,
    CorrelationAnalyzer
)

# Or import individually
from src.sentiment_analyzer import SentimentAnalyzer
from src.technical_analyzer import TAProcessor
```

### Complete Workflow Example

```python
# 1. Load data
loader = DataLoader('../data')
news_df = loader.load_news_data()
stock_df = loader.load_stock_data('AAPL')

# 2. Preprocess
preprocessor = DataPreprocessor()
news_df = preprocessor.normalize_dates(news_df, 'date')
stock_df = preprocessor.normalize_dates(stock_df, 'Date')

# 3. Sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
news_df = sentiment_analyzer.analyze_dataframe(news_df, 'headline')

# 4. Technical analysis
ta_processor = TAProcessor()
stock_df = ta_processor.process_stock_data(stock_df)

# 5. Aggregate sentiment
daily_sentiment = preprocessor.aggregate_by_date(
    news_df, 'trading_date', {'sentiment_score': ['mean', 'count']}
)

# 6. Merge datasets
merged_df = preprocessor.merge_sentiment_stock(daily_sentiment, stock_df)

# 7. Correlation analysis
corr_analyzer = CorrelationAnalyzer()
result = corr_analyzer.analyze_correlation(
    merged_df, 'sentiment_mean', 'Daily_Return'
)

print(f"Correlation: {result['correlation']:.4f}")
print(f"P-value: {result['p_value']:.6f}")
print(f"Significance: {result['significance']}")
```

---

## 🎨 Design Principles

### 1. **Single Responsibility**
Each class has one clear purpose:
- `SentimentAnalyzer` → Sentiment analysis only
- `TAProcessor` → Technical indicators only
- `DataLoader` → Data loading only
- `DataPreprocessor` → Data preprocessing only
- `CorrelationAnalyzer` → Correlation analysis only

### 2. **Encapsulation**
- All classes encapsulate their internal logic
- Clean public APIs exposed
- Private implementation details hidden

### 3. **Reusability**
- Classes can be used independently
- No tight coupling between modules
- Easy to import and use in any context

### 4. **Testability**
- Each class has comprehensive unit tests
- Methods are small and focused
- Easy to mock and test in isolation

### 5. **Extensibility**
- Easy to add new methods to existing classes
- Easy to create new classes following the same pattern
- Inheritance-friendly design

---

## 📊 Benefits of This Architecture

### For Development:
✅ **Modular**: Each component can be developed independently  
✅ **Testable**: Comprehensive unit test coverage  
✅ **Maintainable**: Clear separation of concerns  
✅ **Reusable**: Classes can be used across different notebooks  
✅ **Scalable**: Easy to add new features  

### For Users:
✅ **Simple API**: Easy to learn and use  
✅ **Well-documented**: Clear docstrings and examples  
✅ **Flexible**: Multiple ways to use each class  
✅ **Reliable**: Tested and validated  

### For Production:
✅ **Robust**: Error handling built-in  
✅ **Performant**: Optimized algorithms  
✅ **Professional**: Industry-standard design patterns  
✅ **Deployable**: Ready for production use  

---

## 🚀 Next Steps

### Potential Enhancements:
1. Add logging throughout modules
2. Implement caching for expensive operations
3. Add configuration file support
4. Create CLI interface
5. Add async support for batch operations
6. Implement data validation schemas
7. Add more visualization methods
8. Create performance benchmarks

---

**Version**: 1.0.0  
**Author**: Nova Financial Solutions  
**Last Updated**: November 23, 2025
