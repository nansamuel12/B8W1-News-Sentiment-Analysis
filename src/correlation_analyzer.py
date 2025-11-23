"""
Correlation Analysis Module

Provides statistical correlation analysis between sentiment and stock returns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelationAnalyzer:
    """
    A class for analyzing correlations between sentiment and stock movements.
    
    Supports:
    - Pearson correlation
    - Spearman correlation
    - Lagged correlation analysis
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize the CorrelationAnalyzer."""
        pass
    
    def calculate_pearson(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation, p_value)
        """
        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2:
            return 0.0, 1.0
        
        corr, p_value = stats.pearsonr(x_clean, y_clean)
        return corr, p_value
    
    def calculate_spearman(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """
        Calculate Spearman correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation, p_value)
        """
        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2:
            return 0.0, 1.0
        
        corr, p_value = stats.spearmanr(x_clean, y_clean)
        return corr, p_value
    
    def analyze_correlation(self, df: pd.DataFrame, 
                          x_col: str, 
                          y_col: str,
                          method: str = 'pearson') -> Dict[str, float]:
        """
        Analyze correlation between two variables.
        
        Args:
            df: Input DataFrame
            x_col: Name of first variable column
            y_col: Name of second variable column
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Dictionary with correlation results
        """
        if method == 'pearson':
            corr, p_value = self.calculate_pearson(df[x_col], df[y_col])
        elif method == 'spearman':
            corr, p_value = self.calculate_spearman(df[x_col], df[y_col])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine significance
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'significance': significance,
            'n_samples': len(df.dropna(subset=[x_col, y_col])),
            'method': method
        }
    
    def lagged_correlation(self, df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          max_lag: int = 5,
                          method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate lagged correlations.
        
        Args:
            df: Input DataFrame
            x_col: Name of predictor variable column
            y_col: Name of target variable column
            max_lag: Maximum lag to test
            method: Correlation method
            
        Returns:
            DataFrame with lagged correlation results
        """
        results = []
        
        for lag in range(max_lag + 1):
            df_lagged = df.copy()
            df_lagged[f'{x_col}_lagged'] = df_lagged[x_col].shift(lag)
            
            corr_result = self.analyze_correlation(
                df_lagged,
                f'{x_col}_lagged',
                y_col,
                method=method
            )
            
            results.append({
                'lag': lag,
                'correlation': corr_result['correlation'],
                'p_value': corr_result['p_value'],
                'significance': corr_result['significance']
            })
        
        return pd.DataFrame(results)
    
    def batch_correlation_analysis(self, 
                                  data_dict: Dict[str, pd.DataFrame],
                                  x_col: str,
                                  y_col: str,
                                  method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlations for multiple datasets.
        
        Args:
            data_dict: Dictionary mapping names to DataFrames
            x_col: Name of first variable column
            y_col: Name of second variable column
            method: Correlation method
            
        Returns:
            DataFrame with results for all datasets
        """
        results = []
        
        for name, df in data_dict.items():
            corr_result = self.analyze_correlation(df, x_col, y_col, method)
            corr_result['name'] = name
            results.append(corr_result)
        
        return pd.DataFrame(results)
    
    def plot_correlation_scatter(self, df: pd.DataFrame,
                                x_col: str,
                                y_col: str,
                                title: str = None,
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create scatter plot with regression line.
        
        Args:
            df: Input DataFrame
            x_col: Name of x-axis variable
            y_col: Name of y-axis variable
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Remove NaN values
        plot_df = df[[x_col, y_col]].dropna()
        
        # Scatter plot
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.5)
        
        # Regression line
        z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
        p = np.poly1d(z)
        ax.plot(plot_df[x_col], p(plot_df[x_col]), "r--", linewidth=2)
        
        # Calculate correlation
        corr, p_value = self.calculate_pearson(plot_df[x_col], plot_df[y_col])
        
        # Labels
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f'{x_col} vs {y_col}\nr={corr:.3f}, p={p_value:.4f}')
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                columns: List[str],
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create correlation heatmap.
        
        Args:
            df: Input DataFrame
            columns: List of columns to include
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        corr_matrix = df[columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=1)
        
        ax.set_title('Correlation Heatmap')
        
        return fig
    
    def get_correlation_strength(self, corr: float) -> str:
        """
        Categorize correlation strength.
        
        Args:
            corr: Correlation coefficient
            
        Returns:
            Strength category
        """
        abs_corr = abs(corr)
        
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
