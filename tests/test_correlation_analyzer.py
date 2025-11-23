"""
Unit tests for CorrelationAnalyzer class
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from correlation_analyzer import CorrelationAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for CorrelationAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = CorrelationAnalyzer()
        
        # Create sample data with known correlation
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.7 * x + 0.3 * np.random.randn(n)  # Positive correlation
        
        self.sample_df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': np.random.randn(n)  # Independent variable
        })
    
    def test_initialization(self):
        """Test CorrelationAnalyzer initialization"""
        self.assertIsNotNone(self.analyzer)
    
    def test_calculate_pearson(self):
        """Test Pearson correlation calculation"""
        corr, p_value = self.analyzer.calculate_pearson(
            self.sample_df['x'],
            self.sample_df['y']
        )
        
        self.assertIsInstance(corr, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)
        self.assertGreater(corr, 0.5)  # Should be positive correlation
    
    def test_calculate_spearman(self):
        """Test Spearman correlation calculation"""
        corr, p_value = self.analyzer.calculate_spearman(
            self.sample_df['x'],
            self.sample_df['y']
        )
        
        self.assertIsInstance(corr, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)
    
    def test_analyze_correlation_pearson(self):
        """Test correlation analysis with Pearson method"""
        result = self.analyzer.analyze_correlation(
            self.sample_df,
            'x',
            'y',
            method='pearson'
        )
        
        self.assertIn('correlation', result)
        self.assertIn('p_value', result)
        self.assertIn('significance', result)
        self.assertIn('n_samples', result)
        self.assertIn('method', result)
        
        self.assertEqual(result['method'], 'pearson')
        self.assertEqual(result['n_samples'], 100)
    
    def test_analyze_correlation_spearman(self):
        """Test correlation analysis with Spearman method"""
        result = self.analyzer.analyze_correlation(
            self.sample_df,
            'x',
            'y',
            method='spearman'
        )
        
        self.assertEqual(result['method'], 'spearman')
    
    def test_analyze_correlation_invalid_method(self):
        """Test invalid correlation method"""
        with self.assertRaises(ValueError):
            self.analyzer.analyze_correlation(
                self.sample_df,
                'x',
                'y',
                method='invalid'
            )
    
    def test_significance_levels(self):
        """Test significance level categorization"""
        # Create data with strong correlation (low p-value)
        result = self.analyzer.analyze_correlation(
            self.sample_df,
            'x',
            'y'
        )
        
        self.assertIn(result['significance'], ['***', '**', '*', 'ns'])
    
    def test_lagged_correlation(self):
        """Test lagged correlation analysis"""
        result = self.analyzer.lagged_correlation(
            self.sample_df,
            'x',
            'y',
            max_lag=3
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 0 to 3 lags
        self.assertIn('lag', result.columns)
        self.assertIn('correlation', result.columns)
        self.assertIn('p_value', result.columns)
    
    def test_batch_correlation_analysis(self):
        """Test batch correlation analysis"""
        data_dict = {
            'dataset1': self.sample_df,
            'dataset2': self.sample_df.copy()
        }
        
        result = self.analyzer.batch_correlation_analysis(
            data_dict,
            'x',
            'y'
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('name', result.columns)
        self.assertIn('correlation', result.columns)
    
    def test_get_correlation_strength(self):
        """Test correlation strength categorization"""
        self.assertEqual(self.analyzer.get_correlation_strength(0.8), "Strong")
        self.assertEqual(self.analyzer.get_correlation_strength(0.5), "Moderate")
        self.assertEqual(self.analyzer.get_correlation_strength(0.3), "Weak")
        self.assertEqual(self.analyzer.get_correlation_strength(0.1), "Very Weak")
        self.assertEqual(self.analyzer.get_correlation_strength(-0.8), "Strong")
    
    def test_correlation_with_nan_values(self):
        """Test correlation handling with NaN values"""
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[5:10, 'y'] = np.nan
        
        result = self.analyzer.analyze_correlation(df_with_nan, 'x', 'y')
        
        self.assertIsInstance(result['correlation'], float)
        self.assertLess(result['n_samples'], 100)
    
    def test_correlation_with_insufficient_data(self):
        """Test correlation with insufficient data"""
        small_df = pd.DataFrame({'x': [1], 'y': [2]})
        
        corr, p_value = self.analyzer.calculate_pearson(small_df['x'], small_df['y'])
        
        # Should return default values
        self.assertEqual(corr, 0.0)
        self.assertEqual(p_value, 1.0)


if __name__ == '__main__':
    unittest.main()
