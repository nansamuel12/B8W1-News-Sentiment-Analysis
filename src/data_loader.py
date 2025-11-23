"""
Data Loading and Preprocessing Module

Handles loading and preprocessing of news and stock data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DataLoader:
    """
    A class for loading financial data from CSV files.
    
    Handles:
    - News data loading
    - Stock price data loading
    - Multiple ticker support
    """
    
    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        
    def load_news_data(self, filename: str = 'raw_analyst_ratings.csv') -> pd.DataFrame:
        """
        Load news headlines data.
        
        Args:
            filename: Name of news data file
            
        Returns:
            DataFrame with news data
        """
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath)
        return df
    
    def load_stock_data(self, ticker: str, subdir: str = 'Data') -> pd.DataFrame:
        """
        Load stock price data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            subdir: Subdirectory containing stock data
            
        Returns:
            DataFrame with OHLCV data
        """
        filepath = self.data_dir / subdir / f'{ticker}.csv'
        df = pd.read_csv(filepath)
        return df
    
    def load_multiple_stocks(self, tickers: List[str], subdir: str = 'Data') -> Dict[str, pd.DataFrame]:
        """
        Load stock data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            subdir: Subdirectory containing stock data
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        stock_data = {}
        for ticker in tickers:
            try:
                stock_data[ticker] = self.load_stock_data(ticker, subdir)
            except FileNotFoundError:
                print(f"Warning: Data file not found for {ticker}")
        return stock_data


class DataPreprocessor:
    """
    A class for preprocessing financial data.
    
    Handles:
    - Date normalization
    - Missing value handling
    - Data alignment
    - Feature engineering
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        pass
    
    def normalize_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Normalize dates to trading days (remove time component).
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with normalized dates
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
        
        # Remove timezone if present
        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_localize(None)
        
        # Create trading_date column (date only)
        df['trading_date'] = pd.to_datetime(df[date_column].dt.date)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Strategy to use ('drop', 'ffill', 'bfill', 'mean')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'ffill':
            df = df.fillna(method='ffill')
        elif strategy == 'bfill':
            df = df.fillna(method='bfill')
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    def align_dates(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                   date_col: str = 'trading_date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two DataFrames by common dates.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            date_col: Name of date column
            
        Returns:
            Tuple of aligned DataFrames
        """
        common_dates = set(df1[date_col]).intersection(set(df2[date_col]))
        
        df1_aligned = df1[df1[date_col].isin(common_dates)].copy()
        df2_aligned = df2[df2[date_col].isin(common_dates)].copy()
        
        return df1_aligned, df2_aligned
    
    def aggregate_by_date(self, df: pd.DataFrame, 
                         date_col: str,
                         agg_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Aggregate data by date.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            agg_dict: Dictionary specifying aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        return df.groupby(date_col).agg(agg_dict).reset_index()
    
    def add_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Add time-based features to DataFrame.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['is_month_start'] = df[date_column].dt.is_month_start
        df['is_month_end'] = df[date_column].dt.is_month_end
        
        return df
    
    def merge_sentiment_stock(self, sentiment_df: pd.DataFrame, 
                            stock_df: pd.DataFrame,
                            date_col: str = 'trading_date') -> pd.DataFrame:
        """
        Merge sentiment and stock data by date.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            stock_df: DataFrame with stock data
            date_col: Name of date column
            
        Returns:
            Merged DataFrame
        """
        merged = pd.merge(stock_df, sentiment_df, on=date_col, how='inner')
        merged = merged.dropna()
        return merged
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to lag
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
