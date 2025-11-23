"""
Unit tests for DataLoader and DataPreprocessor classes
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader, DataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader('../data')
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        self.assertIsNotNone(self.loader.data_dir)
        self.assertIsInstance(self.loader.data_dir, Path)
    
    def test_load_stock_data_structure(self):
        """Test that load_stock_data returns proper structure"""
        # This test assumes AAPL.csv exists
        try:
            df = self.loader.load_stock_data('AAPL')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
        except FileNotFoundError:
            self.skipTest("AAPL.csv not found in data directory")


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        self.sample_df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
    
    def test_initialization(self):
        """Test DataPreprocessor initialization"""
        self.assertIsNotNone(self.preprocessor)
    
    def test_normalize_dates(self):
        """Test date normalization"""
        result = self.preprocessor.normalize_dates(self.sample_df, 'date')
        
        self.assertIn('trading_date', result.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['trading_date']))
    
    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop strategy"""
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[5, 'value'] = np.nan
        
        result = self.preprocessor.handle_missing_values(df_with_nan, strategy='drop')
        
        self.assertEqual(len(result), len(df_with_nan) - 1)
        self.assertFalse(result['value'].isna().any())
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test invalid strategy raises error"""
        with self.assertRaises(ValueError):
            self.preprocessor.handle_missing_values(self.sample_df, strategy='invalid')
    
    def test_align_dates(self):
        """Test date alignment"""
        df1 = pd.DataFrame({
            'trading_date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'value1': range(10)
        })
        
        df2 = pd.DataFrame({
            'trading_date': pd.date_range('2020-01-05', periods=10, freq='D'),
            'value2': range(10)
        })
        
        aligned1, aligned2 = self.preprocessor.align_dates(df1, df2)
        
        # Should have 6 common dates (Jan 5-10)
        self.assertEqual(len(aligned1), 6)
        self.assertEqual(len(aligned2), 6)
    
    def test_aggregate_by_date(self):
        """Test date aggregation"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })
        df['date'] = df['date'].dt.date
        
        agg_dict = {'value': ['mean', 'count']}
        result = self.preprocessor.aggregate_by_date(df, 'date', agg_dict)
        
        self.assertLess(len(result), len(df))
        self.assertIn('date', result.columns)
    
    def test_add_time_features(self):
        """Test time feature addition"""
        result = self.preprocessor.add_time_features(self.sample_df, 'date')
        
        expected_features = ['year', 'month', 'day', 'day_of_week', 'quarter']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_merge_sentiment_stock(self):
        """Test sentiment-stock merge"""
        sentiment_df = pd.DataFrame({
            'trading_date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'sentiment': np.random.randn(10)
        })
        
        stock_df = pd.DataFrame({
            'trading_date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'price': 100 + np.random.randn(10),
            'return': np.random.randn(10)
        })
        
        merged = self.preprocessor.merge_sentiment_stock(sentiment_df, stock_df)
        
        self.assertIn('sentiment', merged.columns)
        self.assertIn('price', merged.columns)
        self.assertIn('return', merged.columns)
    
    def test_create_lagged_features(self):
        """Test lagged feature creation"""
        result = self.preprocessor.create_lagged_features(
            self.sample_df,
            columns=['value'],
            lags=[1, 2, 3]
        )
        
        self.assertIn('value_lag_1', result.columns)
        self.assertIn('value_lag_2', result.columns)
        self.assertIn('value_lag_3', result.columns)
        
        # Check that lag is correct
        self.assertEqual(result['value'].iloc[3], result['value_lag_3'].iloc[6])


if __name__ == '__main__':
    unittest.main()
