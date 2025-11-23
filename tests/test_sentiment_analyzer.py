"""
Unit tests for SentimentAnalyzer class
"""

import unittest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
        self.positive_text = "The stock price surged to new highs, excellent performance!"
        self.negative_text = "The company reported terrible losses and declining revenue."
        self.neutral_text = "The meeting was held on Tuesday."
    
    def test_initialization(self):
        """Test SentimentAnalyzer initialization"""
        self.assertIsNotNone(self.analyzer.vader_analyzer)
        self.assertEqual(self.analyzer.method, 'ensemble')
    
    def test_analyze_textblob_positive(self):
        """Test TextBlob analysis on positive text"""
        score = self.analyzer.analyze_textblob(self.positive_text)
        self.assertGreater(score, 0, "Positive text should have positive score")
        self.assertLessEqual(score, 1, "Score should be <= 1")
    
    def test_analyze_textblob_negative(self):
        """Test TextBlob analysis on negative text"""
        score = self.analyzer.analyze_textblob(self.negative_text)
        self.assertLess(score, 0, "Negative text should have negative score")
        self.assertGreaterEqual(score, -1, "Score should be >= -1")
    
    def test_analyze_vader_positive(self):
        """Test VADER analysis on positive text"""
        score = self.analyzer.analyze_vader(self.positive_text)
        self.assertGreater(score, 0, "Positive text should have positive score")
        self.assertLessEqual(score, 1, "Score should be <= 1")
    
    def test_analyze_vader_negative(self):
        """Test VADER analysis on negative text"""
        score = self.analyzer.analyze_vader(self.negative_text)
        self.assertLess(score, 0, "Negative text should have negative score")
        self.assertGreaterEqual(score, -1, "Score should be >= -1")
    
    def test_analyze_ensemble(self):
        """Test ensemble analysis"""
        score = self.analyzer.analyze(self.positive_text, method='ensemble')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)
    
    def test_analyze_invalid_method(self):
        """Test analyze with invalid method"""
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.positive_text, method='invalid')
    
    def test_categorize_sentiment_positive(self):
        """Test sentiment categorization for positive score"""
        category = self.analyzer.categorize_sentiment(0.5)
        self.assertEqual(category, 'Positive')
    
    def test_categorize_sentiment_negative(self):
        """Test sentiment categorization for negative score"""
        category = self.analyzer.categorize_sentiment(-0.5)
        self.assertEqual(category, 'Negative')
    
    def test_categorize_sentiment_neutral(self):
        """Test sentiment categorization for neutral score"""
        category = self.analyzer.categorize_sentiment(0.02)
        self.assertEqual(category, 'Neutral')
    
    def test_analyze_batch(self):
        """Test batch analysis"""
        texts = [self.positive_text, self.negative_text, self.neutral_text]
        results = self.analyzer.analyze_batch(texts)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 3)
        self.assertIn('sentiment_textblob', results.columns)
        self.assertIn('sentiment_vader', results.columns)
        self.assertIn('sentiment_score', results.columns)
    
    def test_analyze_dataframe(self):
        """Test DataFrame analysis"""
        df = pd.DataFrame({
            'headline': [self.positive_text, self.negative_text]
        })
        
        result = self.analyzer.analyze_dataframe(df, 'headline')
        
        self.assertIn('sentiment_textblob', result.columns)
        self.assertIn('sentiment_vader', result.columns)
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('sentiment_category', result.columns)
    
    def test_get_sentiment_stats(self):
        """Test sentiment statistics calculation"""
        scores = [0.5, -0.3, 0.1, 0.8, -0.2]
        stats = self.analyzer.get_sentiment_stats(scores)
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('positive_pct', stats)
        self.assertIn('negative_pct', stats)
        self.assertIn('neutral_pct', stats)
    
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        score = self.analyzer.analyze("")
        self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main()
