"""
Technical Analysis Module

Provides technical indicator calculations using TA-Lib.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, Tuple


class TAProcessor:
    """
    A class for calculating technical indicators and financial metrics.
    
    Supports:
    - Moving Averages (SMA, EMA)
    - Momentum Indicators (RSI, MACD, Stochastic)
    - Volatility Indicators (Bollinger Bands, ATR)
    - Volume Indicators (OBV)
    - Financial Metrics (Returns, Volatility, Sharpe Ratio)
    """
    
    def __init__(self):
        """Initialize the TAProcessor."""
        pass
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price series
            period: SMA period
            
        Returns:
            SMA values
        """
        return talib.SMA(data, timeperiod=period)
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA values
        """
        return talib.EMA(data, timeperiod=period)
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI values (0-100)
        """
        return talib.RSI(data, timeperiod=period)
    
    def calculate_macd(self, data: pd.Series, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast: Fast period (default: 12)
            slow: Slow period (default: 26)
            signal: Signal period (default: 9)
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        macd, signal_line, histogram = talib.MACD(data, 
                                                   fastperiod=fast,
                                                   slowperiod=slow, 
                                                   signalperiod=signal)
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, 
                                  period: int = 20, 
                                  std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price series
            period: Moving average period (default: 20)
            std: Standard deviation multiplier (default: 2)
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        upper, middle, lower = talib.BBANDS(data, 
                                            timeperiod=period,
                                            nbdevup=std, 
                                            nbdevdn=std)
        return upper, middle, lower
    
    def calculate_atr(self, high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default: 14)
            
        Returns:
            ATR values
        """
        return talib.ATR(high, low, close, timeperiod=period)
    
    def calculate_stochastic(self, high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series,
                           fastk_period: int = 14,
                           slowk_period: int = 3,
                           slowd_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            fastk_period: Fast K period (default: 14)
            slowk_period: Slow K period (default: 3)
            slowd_period: Slow D period (default: 3)
            
        Returns:
            Tuple of (slowk, slowd)
        """
        slowk, slowd = talib.STOCH(high, low, close,
                                   fastk_period=fastk_period,
                                   slowk_period=slowk_period,
                                   slowd_period=slowd_period)
        return slowk, slowd
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV values
        """
        return talib.OBV(close, volume)
    
    def calculate_daily_return(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily percentage returns.
        
        Args:
            prices: Price series
            
        Returns:
            Daily returns (%)
        """
        return prices.pct_change() * 100
    
    def calculate_log_return(self, prices: pd.Series) -> pd.Series:
        """
        Calculate logarithmic returns.
        
        Args:
            prices: Price series
            
        Returns:
            Log returns
        """
        return np.log(prices / prices.shift(1)) * 100
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Return series
            window: Rolling window size (default: 20)
            
        Returns:
            Volatility values
        """
        return returns.rolling(window=window).std()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Return series (in percentage)
            risk_free_rate: Annual risk-free rate (default: 0.02)
            periods_per_year: Trading periods per year (default: 252)
            
        Returns:
            Sharpe ratio
        """
        returns_decimal = returns / 100
        mean_return = returns_decimal.mean() * periods_per_year
        std_return = returns_decimal.std() * np.sqrt(periods_per_year)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    def process_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to stock DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        df['SMA_50'] = self.calculate_sma(df['Close'], 50)
        df['SMA_200'] = self.calculate_sma(df['Close'], 200)
        df['EMA_12'] = self.calculate_ema(df['Close'], 12)
        df['EMA_26'] = self.calculate_ema(df['Close'], 26)
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # ATR
        df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        
        # OBV
        df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # Returns and Metrics
        df['Daily_Return'] = self.calculate_daily_return(df['Close'])
        df['Log_Return'] = self.calculate_log_return(df['Close'])
        df['Volatility'] = self.calculate_volatility(df['Daily_Return'])
        df['Cumulative_Return'] = (1 + df['Daily_Return'] / 100).cumprod()
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with statistics for each indicator
        """
        indicators = ['RSI', 'MACD', 'ATR', 'Daily_Return', 'Volatility']
        summary = {}
        
        for indicator in indicators:
            if indicator in df.columns:
                summary[indicator] = {
                    'mean': df[indicator].mean(),
                    'std': df[indicator].std(),
                    'min': df[indicator].min(),
                    'max': df[indicator].max(),
                    'current': df[indicator].iloc[-1] if len(df) > 0 else None
                }
        
        return summary
