"""
Unit tests for TAProcessor class
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from technical_analyzer import TAProcessor


class TestTAProcessor(unittest.TestCase):
    """Test cases for TAProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = TAProcessor()
        
        # Create sample stock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.sample_df = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 102 + np.random.randn(100).cumsum(),
            'Low': 98 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
    
    def test_initialization(self):
        """Test TAProcessor initialization"""
        self.assertIsNotNone(self.processor)
    
    def test_calculate_sma(self):
        """Test SMA calculation"""
        sma = self.processor.calculate_sma(self.sample_df['Close'], 20)
        
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.sample_df))
        self.assertTrue(pd.isna(sma.iloc[0]))  # First values should be NaN
        self.assertFalse(pd.isna(sma.iloc[-1]))  # Last value should not be NaN
    
    def test_calculate_ema(self):
        """Test EMA calculation"""
        ema = self.processor.calculate_ema(self.sample_df['Close'], 12)
        
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(self.sample_df))
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = self.processor.calculate_rsi(self.sample_df['Close'])
        
        self.assertIsInstance(rsi, pd.Series)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        macd, signal, hist = self.processor.calculate_macd(self.sample_df['Close'])
        
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(hist, pd.Series)
        self.assertEqual(len(macd), len(self.sample_df))
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = self.processor.calculate_bollinger_bands(self.sample_df['Close'])
        
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # Upper should be >= Middle >= Lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        self.assertTrue((upper[valid_idx] >= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] >= lower[valid_idx]).all())
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        atr = self.processor.calculate_atr(
            self.sample_df['High'],
            self.sample_df['Low'],
            self.sample_df['Close']
        )
        
        self.assertIsInstance(atr, pd.Series)
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr >= 0).all())
    
    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation"""
        slowk, slowd = self.processor.calculate_stochastic(
            self.sample_df['High'],
            self.sample_df['Low'],
            self.sample_df['Close']
        )
        
        self.assertIsInstance(slowk, pd.Series)
        self.assertIsInstance(slowd, pd.Series)
        
        # Stochastic should be between 0 and 100
        valid_k = slowk.dropna()
        valid_d = slowd.dropna()
        self.assertTrue((valid_k >= 0).all() and (valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all() and (valid_d <= 100).all())
    
    def test_calculate_obv(self):
        """Test OBV calculation"""
        obv = self.processor.calculate_obv(
            self.sample_df['Close'],
            self.sample_df['Volume']
        )
        
        self.assertIsInstance(obv, pd.Series)
        self.assertEqual(len(obv), len(self.sample_df))
    
    def test_calculate_daily_return(self):
        """Test daily return calculation"""
        returns = self.processor.calculate_daily_return(self.sample_df['Close'])
        
        self.assertIsInstance(returns, pd.Series)
        self.assertTrue(pd.isna(returns.iloc[0]))  # First return should be NaN
    
    def test_calculate_log_return(self):
        """Test log return calculation"""
        log_returns = self.processor.calculate_log_return(self.sample_df['Close'])
        
        self.assertIsInstance(log_returns, pd.Series)
        self.assertTrue(pd.isna(log_returns.iloc[0]))
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        returns = self.processor.calculate_daily_return(self.sample_df['Close'])
        volatility = self.processor.calculate_volatility(returns, window=20)
        
        self.assertIsInstance(volatility, pd.Series)
        # Volatility should be positive
        valid_vol = volatility.dropna()
        self.assertTrue((valid_vol >= 0).all())
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = self.processor.calculate_daily_return(self.sample_df['Close'])
        sharpe = self.processor.calculate_sharpe_ratio(returns.dropna())
        
        self.assertIsInstance(sharpe, float)
    
    def test_process_stock_data(self):
        """Test full stock data processing"""
        result = self.processor.process_stock_data(self.sample_df)
        
        # Check that all indicators are added
        expected_columns = [
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'ATR', 'Stoch_K', 'Stoch_D', 'OBV',
            'Daily_Return', 'Log_Return', 'Volatility', 'Cumulative_Return'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")
    
    def test_get_indicator_summary(self):
        """Test indicator summary generation"""
        processed_df = self.processor.process_stock_data(self.sample_df)
        summary = self.processor.get_indicator_summary(processed_df)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('RSI', summary)
        self.assertIn('MACD', summary)
        
        # Check summary structure
        if 'RSI' in summary:
            self.assertIn('mean', summary['RSI'])
            self.assertIn('std', summary['RSI'])
            self.assertIn('current', summary['RSI'])


if __name__ == '__main__':
    unittest.main()
