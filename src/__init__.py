"""
Financial Analysis Package

A comprehensive package for financial news sentiment analysis,
technical analysis, and correlation studies.

Modules:
    - sentiment_analyzer: Sentiment analysis using TextBlob and VADER
    - technical_analyzer: Technical indicators using TA-Lib
    - data_loader: Data loading and preprocessing utilities
    - correlation_analyzer: Statistical correlation analysis
"""

from .sentiment_analyzer import SentimentAnalyzer
from .technical_analyzer import TAProcessor
from .data_loader import DataLoader, DataPreprocessor
from .correlation_analyzer import CorrelationAnalyzer

__version__ = '1.0.0'
__author__ = 'Nova Financial Solutions'

__all__ = [
    'SentimentAnalyzer',
    'TAProcessor',
    'DataLoader',
    'DataPreprocessor',
    'CorrelationAnalyzer'
]
