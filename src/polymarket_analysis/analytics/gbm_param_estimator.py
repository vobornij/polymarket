import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import spans

class GBMParameterEstimator:
    def __init__(self, symbol="BTC-USD", period="1y"):
        """
        Initialize GBM parameter estimator for Bitcoin
        
        Args:
            symbol: Yahoo Finance symbol for Bitcoin
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        self.symbol = symbol
        self.period = period
        self.prices = None
        self.returns = None
        self.mu = None  # drift parameter
        self.sigma = None  # volatility parameter
        
    def fetch_data(self):
        """Fetch Bitcoin price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=self.period)
            self.prices = data['Close']
            print(f"Fetched {len(self.prices)} price points for {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_returns(self):
        """Calculate log returns from price data"""
        if self.prices is None:
            raise ValueError("No price data available. Call fetch_data() first.")
        
        # Calculate log returns
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        print(f"Calculated {len(self.returns)} daily returns")
        
    def estimate_parameters(self):
        """Estimate GBM parameters (mu and sigma) from historical data using daily model"""
        if self.returns is None:
            raise ValueError("No returns data available. Call calculate_returns() first.")
        
        # Daily parameters (not annualized)
        self.mu = self.returns.mean()  # Daily drift
        self.sigma = self.returns.std()  # Daily volatility
        
        # For reference, calculate annualized versions
        self.mu_annualized = self.mu * 365
        self.sigma_annualized = self.sigma * np.sqrt(365)
        
        return {
            'mu_daily': self.mu,
            'sigma_daily': self.sigma,
            'mu_annualized': self.mu_annualized,
            'sigma_annualized': self.sigma_annualized
        }
    
    def get_parameters_summary(self):
        """Get a summary of estimated parameters"""
        if self.mu is None or self.sigma is None:
            raise ValueError("Parameters not estimated. Call estimate_parameters() first.")
        
        # Calculate confidence intervals
        n = len(self.returns)
        
        # 95% confidence interval for mean
        se_mean = self.returns.std() / np.sqrt(n)
        ci_mean = stats.t.interval(0.95, n-1, loc=self.returns.mean(), scale=se_mean)
        
        # 95% confidence interval for std
        chi2_lower = stats.chi2.ppf(0.025, n-1)
        chi2_upper = stats.chi2.ppf(0.975, n-1)
        ci_std = (
            self.returns.std() * np.sqrt((n-1) / chi2_upper),
            self.returns.std() * np.sqrt((n-1) / chi2_lower)
        )
        
        summary = {
            'symbol': self.symbol,
            'period': self.period,
            'n_observations': len(self.returns),
            'current_price': self.prices.iloc[-1],
            'daily_drift': self.mu,
            'daily_volatility': self.sigma,
            'annualized_drift': self.mu_annualized,
            'annualized_volatility': self.sigma_annualized,
            'daily_return_mean': self.returns.mean(),
            'daily_return_std': self.returns.std(),
            'sharpe_ratio_daily': self.mu / self.sigma if self.sigma != 0 else 0,
            'sharpe_ratio_annualized': self.mu_annualized / self.sigma_annualized if self.sigma_annualized != 0 else 0,
            'confidence_intervals': {
                'daily_mean_95ci': ci_mean,
                'daily_std_95ci': ci_std,
                'annual_vol_95ci': (ci_std[0] * np.sqrt(365), ci_std[1] * np.sqrt(365))
            }
        }
        
        return summary
        """
        Simulate future price paths using GBM with daily parameters
        
        Args:
            days_ahead: Number of days to simulate
            n_simulations: Number of Monte Carlo simulations
            S0: Starting price (uses last available price if None)
        
        Returns:
            Array of simulated final prices
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Parameters not estimated. Call estimate_parameters() first.")
        
        if S0 is None:
            S0 = self.prices.iloc[-1]
        
        # Generate random shocks for each day
        Z = np.random.standard_normal(n_simulations)
        
        # GBM formula for daily model: S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # For daily parameters, T = days_ahead (not converted to years)
        log_returns = (self.mu - 0.5 * self.sigma**2) * days_ahead + self.sigma * np.sqrt(days_ahead) * Z
        
        # Calculate final prices
        final_prices = S0 * np.exp(log_returns)
        
        return final_prices
    
    def probability_analysis(self, current_price, target_price, days_ahead):
        """
        Calculate probability of reaching target price using analytical GBM formula
        
        Args:
            target_price: Target Bitcoin price
            days_ahead: Number of days ahead
        
        Returns:
            Dictionary with probability analysis
        """
        from scipy.stats import norm
        
        if self.mu is None or self.sigma is None:
            raise ValueError("Parameters not estimated. Call estimate_parameters() first.")
        
        # Analytical GBM probability calculation
        # For GBM: ln(S_T/S_0) ~ Normal((μ - 0.5*σ²)*T, σ²*T)
        
        # Calculate parameters for the log return distribution
        log_drift = (self.mu - 0.5 * self.sigma**2) * days_ahead
        log_volatility = self.sigma * np.sqrt(days_ahead)
        
        # Calculate log ratio: ln(target_price / current_price)
        log_target_ratio = np.log(target_price / current_price)
        
        # Probability that final price >= target price
        # P(S_T >= target) = P(ln(S_T/S_0) >= ln(target/S_0))
        # = P(Z >= (ln(target/S_0) - log_drift) / log_volatility)
        # = 1 - Φ((ln(target/S_0) - log_drift) / log_volatility)
        
        if log_volatility > 0:
            z_score = (log_target_ratio - log_drift) / log_volatility
            prob_above_target = 1 - norm.cdf(z_score)
            prob_below_target = norm.cdf(z_score)
        else:
            # Edge case: no volatility
            prob_above_target = 1.0 if log_drift >= log_target_ratio else 0.0
            prob_below_target = 1.0 - prob_above_target
        
        # Expected price: E[S_T] = S_0 * exp(μ * T)
        expected_price = current_price * np.exp(self.mu * days_ahead)
        
        # Median price: S_0 * exp((μ - 0.5*σ²) * T)
        median_price = current_price * np.exp(log_drift)
        
        # Price percentiles using analytical formula
        percentiles_z = [-1.645, -1.282, -0.674, 0, 0.674, 1.282, 1.645]  # 5%, 10%, 25%, 50%, 75%, 90%, 95%
        percentiles = []
        
        for z in percentiles_z:
            log_price = np.log(current_price) + log_drift + log_volatility * z
            percentiles.append(np.exp(log_price))
        
        # Standard deviation of final price
        # Var[S_T] = S_0² * exp(2μT) * (exp(σ²T) - 1)
        variance = (current_price**2) * np.exp(2 * self.mu * days_ahead) * (np.exp(self.sigma**2 * days_ahead) - 1)
        std_dev = np.sqrt(variance)
        
        return {
            'current_price': current_price,
            'target_price': target_price,
            'days_ahead': days_ahead,
            'probability_above_target': prob_above_target,
            'probability_below_target': prob_below_target,
            'expected_price': expected_price,
            'median_price': median_price,
            'price_percentiles': {
                '5%': percentiles[0],
                '10%': percentiles[1],
                '25%': percentiles[2],
                '50%': percentiles[3],
                '75%': percentiles[4],
                '90%': percentiles[5],
                '95%': percentiles[6]
            },
            'std_dev': std_dev,
            'log_drift': log_drift,
            'log_volatility': log_volatility,
            'z_score': z_score if log_volatility > 0 else None
        }

    def p_above_target(self, current_price, target_price, days_ahead):
        """
        Calculate the probability of the price being above a target price after a certain number of days.
        
        Args:
            current_price: Current Bitcoin price
            target_price: Target Bitcoin price
            days_ahead: Number of days to project
        
        Returns:
            Probability of the price being above the target price
        """
        analysis = self.probability_analysis(current_price, target_price, days_ahead)
        return analysis['probability_above_target']

    def p_below_target(self, current_price, target_price, days_ahead):
        analysis = self.probability_analysis(current_price, target_price, days_ahead)
        return analysis['probability_below_target']
    
    def p_between_targets(self, current_price, lower, upper, days_ahead):
        lower_analysis = self.probability_analysis(current_price, lower, days_ahead)
        upper_analysis = self.probability_analysis(current_price, upper, days_ahead)

        return lower_analysis['probability_above_target'] + upper_analysis['probability_below_target'] - 1

    def p_in_range(self, current_price, target_range: spans.floatrange, days_ahead) -> float:
        if(target_range.lower is None):
            return self.p_below_target(current_price, target_range.upper, days_ahead)
        elif(target_range.upper is None):
            return self.p_above_target(current_price, target_range.lower, days_ahead)
        else:
            return self.p_between_targets(current_price, target_range.lower, target_range.upper, days_ahead)


