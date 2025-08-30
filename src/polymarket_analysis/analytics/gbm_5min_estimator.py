from datetime import datetime, time, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats
import pytz
from polymarket_analysis.data.binance import Binance


class GBMParameter5MinEstimator:
    def __init__(self, data_file_path: str = "/Users/vobornij/projects/polymarket/data/btc_5min_data.json"):
        """
        Initialize the GBM Parameter Hourly Estimator.
        
        Args:
            data_file_path: Path to the JSON file containing Bitcoin 5-minute data
        """
        self.data_file_path = data_file_path
        self.btc_data: Optional[pd.DataFrame] = None
        self.weekly_template: Optional[pd.DataFrame] = None
        self.overall_mean_log_return: float = 0.0
        self.lookback_window = 1 * 24 * 12  # 1 day of 5-min intervals
        
        self._load_and_prepare_data()
        
    def _load_and_prepare_data(self):
        """Load Bitcoin 5-minute data from file and prepare templates."""
        # Load Bitcoin 5-minute data from JSON file
        try:
            self.btc_data = pd.read_json(self.data_file_path, lines=True)
            self.btc_data['timestamp'] = pd.to_datetime(self.btc_data['timestamp'])
            self.btc_data = self.btc_data[['timestamp', 'price']].copy()
            print(f"Loaded {len(self.btc_data)} data points from {self.data_file_path}")
        except Exception as e:
            print(f"Error loading data from {self.data_file_path}: {e}")
            print("Falling back to Binance API...")
            # Fallback to Binance API if file loading fails
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self.btc_data = Binance.load_bitcon_5min(
                from_date=start_date,
                to_date=end_date,
            )[['timestamp', 'price']].copy()
        
        # Calculate returns
        self.btc_data['prev_price'] = self.btc_data['price'].shift(1)
        self.btc_data['return'] = self.btc_data['price'] / self.btc_data['prev_price']
        self.btc_data['log_return'] = np.log(self.btc_data['return'])
        
        # Calculate overall mean log return
        self.overall_mean_log_return = self.btc_data['log_return'].mean()
        
        # Create weekly template
        self._create_weekly_template()
        
    def _create_weekly_template(self):
        """Create weekly template with rolling smoothed statistics."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
            
        result_weekly = []
        
        # Calculate statistics for each day/hour/5-min interval
        for day_of_week in range(7):  # 0=Monday, 6=Sunday
            for hour in range(24):
                for interval in range(12):  # 12 5-minute intervals per hour
                    mask = (
                        (self.btc_data['timestamp'].dt.dayofweek == day_of_week) &
                        (self.btc_data['timestamp'].dt.hour == hour) & 
                        (self.btc_data['timestamp'].dt.minute == interval * 5)
                    )
                    
                    if mask.any():
                        bucket_data = self.btc_data.loc[mask].copy()
                        
                        # Filter extreme values (0.5% from each tail)
                        q_low = bucket_data['log_return'].quantile(0.005)
                        q_high = bucket_data['log_return'].quantile(0.995)
                        filtered_data = bucket_data[
                            (bucket_data['log_return'] >= q_low) & 
                            (bucket_data['log_return'] <= q_high)
                        ]
                        
                        if len(filtered_data) > 0:
                            log_return_std = filtered_data['log_return'].std()
                            
                            result_weekly.append({
                                'day_of_week': day_of_week,
                                'hour': hour,
                                'interval': interval,
                                'log_return_std': log_return_std,
                                'n_observations': len(filtered_data)
                            })
        
        result_weekly = pd.DataFrame(result_weekly)
        
        # Apply circular rolling smoothing
        window_size = 12  # 1 hour window
        result_weekly['log_return_std_rolling'] = self._circular_rolling(
            result_weekly['log_return_std'], window_size
        )
        
        self.weekly_template = result_weekly
        
    def _circular_rolling(self, series, window):
        """Apply rolling mean with circular boundary conditions."""
        half_window = window // 2
        
        # Create circular extension
        extended = pd.concat([
            series.iloc[-half_window:],
            series,
            series.iloc[:half_window]
        ], ignore_index=True)
        
        # Apply rolling and extract middle portion
        rolled = extended.rolling(window=window, center=True).mean()
        return rolled.iloc[half_window:-half_window].reset_index(drop=True)
        
    def _get_template_std(self, timestamp: datetime) -> float:
        """Get template standard deviation for a given timestamp."""
        if self.weekly_template is None:
            raise ValueError("Weekly template not created")
            
        day_of_week = timestamp.weekday()
        hour = timestamp.hour
        minute_interval = timestamp.minute // 5
        
        template_row = self.weekly_template[
            (self.weekly_template['day_of_week'] == day_of_week) &
            (self.weekly_template['hour'] == hour) &
            (self.weekly_template['interval'] == minute_interval)
        ]
        
        if len(template_row) > 0:
            return template_row.iloc[0]['log_return_std_rolling']
        else:
            raise ValueError("Bitcoin data not loaded")
            # Fallback to overall std if not found
            # if self.btc_data is None:
            #     raise ValueError("Bitcoin data not loaded")
            # return self.btc_data['log_return'].std()
    
    def _calculate_scaling_factor(self, current_time: datetime) -> float:
        """Calculate scaling factor based on recent actual volatility vs template."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
        
        # Ensure timezone compatibility
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        elif current_time.tzinfo != pytz.UTC:
            current_time = current_time.astimezone(pytz.UTC)
            
        # Get recent data (last lookback_window intervals)
        recent_data = self.btc_data[
            self.btc_data['timestamp'] <= current_time
        ].tail(self.lookback_window)
        
        if len(recent_data) < self.lookback_window // 2:
            raise ValueError("recent_data is too short for scaling factor calculation")
            # return 1.0  # Default scaling if insufficient data
            
        # Calculate actual rolling std
        actual_std = recent_data['log_return'].std()
        
        # Calculate template RMS over the same period
        template_stds = []
        for _, row in recent_data.iterrows():
            template_std = self._get_template_std(row['timestamp'])
            template_stds.append(template_std)
        
        template_rms = np.sqrt(np.mean(np.array(template_stds)**2))
        
        if template_rms > 0:
            return actual_std / template_rms
        else:
            return 1.0
            
    def p_above_target(self, current_time: datetime, current_price: float, 
                      target_price: float, hours_ahead: float) -> float:
        """
        Calculate the probability of the price being above a target price.
        
        Args:
            current_time: Current timestamp
            current_price: Current Bitcoin price
            target_price: Target Bitcoin price
            hours_ahead: Number of hours to project
            
        Returns:
            Probability of the price being above the target price
        """
        scaling_factor = self._calculate_scaling_factor(current_time)
        
        # Project forward using template with scaling
        total_log_drift = 0.0
        total_log_variance = 0.0
        
        intervals_ahead = int(hours_ahead * 12)  # 5-min intervals
        current_timestamp = current_time
        
        for i in range(intervals_ahead):
            # Get template std for this interval and scale it
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Add to cumulative drift and variance
            total_log_drift += self.overall_mean_log_return
            total_log_variance += scaled_std ** 2
            
            # Move to next 5-minute interval
            current_timestamp += timedelta(minutes=5)
        
        total_log_std = np.sqrt(total_log_variance)
        
        # Calculate probability using normal distribution
        log_target_ratio = np.log(target_price / current_price)
        standardized = (log_target_ratio - total_log_drift) / total_log_std
        
        return float(1 - stats.norm.cdf(standardized))
    
    def p_between_targets(self, current_time: datetime, current_price: float, 
                         lower: float, upper: float, hours_ahead: float) -> float:
        """
        Calculate the probability of the price being between two target prices.
        
        Args:
            current_time: Current timestamp
            current_price: Current Bitcoin price
            lower: Lower bound price
            upper: Upper bound price
            hours_ahead: Number of hours to project
            
        Returns:
            Probability of the price being between the target prices
        """
        scaling_factor = self._calculate_scaling_factor(current_time)
        
        # Project forward using template with scaling
        total_log_drift = 0.0
        total_log_variance = 0.0
        
        intervals_ahead = int(hours_ahead * 12)  # 5-min intervals
        current_timestamp = current_time
        
        for i in range(intervals_ahead):
            # Get template std for this interval and scale it
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Add to cumulative drift and variance
            total_log_drift += self.overall_mean_log_return
            total_log_variance += scaled_std ** 2
            
            # Move to next 5-minute interval
            current_timestamp += timedelta(minutes=5)
        
        total_log_std = np.sqrt(total_log_variance)
        
        # Calculate probabilities for both bounds
        log_lower_ratio = np.log(lower / current_price)
        log_upper_ratio = np.log(upper / current_price)
        
        standardized_lower = (log_lower_ratio - total_log_drift) / total_log_std
        standardized_upper = (log_upper_ratio - total_log_drift) / total_log_std
        
        prob_below_upper = stats.norm.cdf(standardized_upper)
        prob_below_lower = stats.norm.cdf(standardized_lower)
        
        return float(prob_below_upper - prob_below_lower)
    
    def get_current_price(self) -> float:
        """Get the most recent Bitcoin price from loaded data."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
        return float(self.btc_data['price'].iloc[-1])
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the loaded data."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
            
        return {
            'data_points': len(self.btc_data),
            'date_range': {
                'start': self.btc_data['timestamp'].min(),
                'end': self.btc_data['timestamp'].max()
            },
            'price_range': {
                'min': self.btc_data['price'].min(),
                'max': self.btc_data['price'].max(),
                'current': self.btc_data['price'].iloc[-1]
            },
            'overall_mean_log_return': self.overall_mean_log_return,
            'template_intervals': len(self.weekly_template) if self.weekly_template is not None else 0
        }


