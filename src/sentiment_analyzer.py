"""
Sentiment Analysis Module

Provides sentiment analysis functionality using TextBlob and VADER.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from typing import Union, List, Dict


class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text data.
    
    Supports multiple sentiment analysis methods:
    - TextBlob (polarity-based)
    - VADER (compound score)
    - Ensemble (average of both)
    
    Attributes:
        vader_analyzer: VADER sentiment analyzer instance
        method: Default sentiment analysis method
    """
    
    def __init__(self, method: str = 'ensemble'):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            method: Sentiment analysis method ('textblob', 'vader', or 'ensemble')
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.method = method
        
    def analyze_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception as e:
            return 0.0
    
    def analyze_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Compound sentiment score (-1 to 1)
        """
        try:
            return self.vader_analyzer.polarity_scores(str(text))['compound']
        except Exception as e:
            return 0.0
    
    def analyze(self, text: str, method: str = None) -> float:
        """
        Analyze sentiment using specified method.
        
        Args:
            text: Input text to analyze
            method: Method to use (overrides default if provided)
            
        Returns:
            Sentiment score (-1 to 1)
        """
        method = method or self.method
        
        if method == 'textblob':
            return self.analyze_textblob(text)
        elif method == 'vader':
            return self.analyze_vader(text)
        elif method == 'ensemble':
            textblob_score = self.analyze_textblob(text)
            vader_score = self.analyze_vader(text)
            return (textblob_score + vader_score) / 2
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_batch(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List or Series of texts to analyze
            
        Returns:
            DataFrame with sentiment scores for each method
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        results = {
            'text': texts,
            'sentiment_textblob': [self.analyze_textblob(t) for t in texts],
            'sentiment_vader': [self.analyze_vader(t) for t in texts],
        }
        
        df = pd.DataFrame(results)
        df['sentiment_score'] = (df['sentiment_textblob'] + df['sentiment_vader']) / 2
        
        return df
    
    def categorize_sentiment(self, score: float, threshold: float = 0.05) -> str:
        """
        Categorize sentiment score into Positive/Neutral/Negative.
        
        Args:
            score: Sentiment score (-1 to 1)
            threshold: Threshold for neutral classification
            
        Returns:
            Sentiment category ('Positive', 'Neutral', or 'Negative')
        """
        if score > threshold:
            return 'Positive'
        elif score < -threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Add sentiment analysis columns to a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()
        
        df['sentiment_textblob'] = df[text_column].apply(self.analyze_textblob)
        df['sentiment_vader'] = df[text_column].apply(self.analyze_vader)
        df['sentiment_score'] = (df['sentiment_textblob'] + df['sentiment_vader']) / 2
        df['sentiment_category'] = df['sentiment_score'].apply(self.categorize_sentiment)
        
        return df
    
    def get_sentiment_stats(self, scores: Union[List[float], pd.Series]) -> Dict[str, float]:
        """
        Calculate statistics for sentiment scores.
        
        Args:
            scores: List or Series of sentiment scores
            
        Returns:
            Dictionary with mean, std, min, max, and median
        """
        scores = pd.Series(scores)
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'median': scores.median(),
            'positive_pct': (scores > 0.05).sum() / len(scores) * 100,
            'negative_pct': (scores < -0.05).sum() / len(scores) * 100,
            'neutral_pct': ((scores >= -0.05) & (scores <= 0.05)).sum() / len(scores) * 100
        }
