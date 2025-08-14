from datetime import datetime, time, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats
import pytz
from polymarket_analysis.data.binance import Binance
import math

class GBMParameterHourlyTemplateEstimator:
    def __init__(self, data_file_path: str = "/Users/kate/projects/polymarket/data/btc_5min_data.json"):
        """
        Initialize the GBM Parameter Hourly Template Estimator.
        This estimator builds templates from 5-minute data but aggregates them into hourly templates
        for faster calculation by taking the mean of the 5-minute standard deviations within each hour.
        
        Args:
            data_file_path: Path to the JSON file containing Bitcoin 5-minute data
        """
        self.data_file_path = data_file_path
        self.btc_data: Optional[pd.DataFrame] = None
        self.hourly_template: Optional[pd.DataFrame] = None
        self.overall_mean_log_return: float = 0.0
        self.lookback_window = 1 * 24  # 1 day of hourly intervals
        
        self._load_and_prepare_data()
        
    def _load_and_prepare_data(self):
        """Load Bitcoin 5-minute data from file and prepare hourly templates."""
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
        
        # Calculate overall mean log return (scale to hourly)
        self.overall_mean_log_return = self.btc_data['log_return'].mean() * 12  # Scale 5-min to hourly
        
        # Create hourly template from 5-minute data
        self._create_hourly_template()
        
    def _create_hourly_template(self):
        """Create hourly template by aggregating 5-minute statistics."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
        
        # First, create 5-minute template similar to the original estimator
        result_5min = []
        
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
                            
                            result_5min.append({
                                'day_of_week': day_of_week,
                                'hour': hour,
                                'interval': interval,
                                'log_return_std': log_return_std,
                                'n_observations': len(filtered_data)
                            })
        
        result_5min_df = pd.DataFrame(result_5min)
        
        # Apply circular rolling smoothing to 5-minute data
        window_size = 12  # 1 hour window
        result_5min_df['log_return_std_rolling'] = self._circular_rolling(
            result_5min_df['log_return_std'], window_size
        )
        
        # Now aggregate to hourly template by taking mean of 5-minute stds within each hour
        result_hourly = []
        
        for day_of_week in range(7):
            for hour in range(24):
                hour_data = result_5min_df[
                    (result_5min_df['day_of_week'] == day_of_week) &
                    (result_5min_df['hour'] == hour)
                ]
                
                if len(hour_data) > 0:
                    # Take mean of smoothed 5-minute stds to get hourly std
                    hourly_std = hour_data['log_return_std_rolling'].mean()
                    # Scale to hourly variance then back to std
                    # Variance scales linearly with time, so hourly variance = 12 * 5min variance
                    hourly_std_scaled = hourly_std * np.sqrt(12)
                    total_observations = hour_data['n_observations'].sum()
                    
                    result_hourly.append({
                        'day_of_week': day_of_week,
                        'hour': hour,
                        'log_return_std': hourly_std_scaled,
                        'n_observations': total_observations
                    })
        
        result_hourly_df = pd.DataFrame(result_hourly)
        
        # Apply additional smoothing to hourly template
        window_size_hourly = 1
        result_hourly_df['log_return_std_rolling'] = result_hourly_df['log_return_std']
        
        self.hourly_template = result_hourly_df
        
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
        if self.hourly_template is None:
            raise ValueError("Hourly template not created")
            
        day_of_week = timestamp.weekday()
        hour = timestamp.hour
        
        template_row = self.hourly_template[
            (self.hourly_template['day_of_week'] == day_of_week) &
            (self.hourly_template['hour'] == hour)
        ]
        
        if len(template_row) > 0:
            return template_row.iloc[0]['log_return_std']
        else:
            raise ValueError("Template not found for timestamp")
    
    def _calculate_scaling_factor(self, current_time: datetime) -> float:
        """Calculate scaling factor based on recent actual volatility vs template."""
        if self.btc_data is None:
            raise ValueError("Bitcoin data not loaded")
        
        # Ensure timezone compatibility
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        elif current_time.tzinfo != pytz.UTC:
            current_time = current_time.astimezone(pytz.UTC)
            
        # Get recent data and aggregate to hourly
        recent_5min_data = self.btc_data[
            self.btc_data['timestamp'] <= current_time
        ].tail(self.lookback_window * 12)  # Convert hourly window to 5-min intervals
        
        if len(recent_5min_data) < (self.lookback_window * 12) // 2:
            raise ValueError("recent_data is too short for scaling factor calculation")
            
        # Aggregate to hourly data for scaling calculation
        recent_5min_data['hour_floor'] = recent_5min_data['timestamp'].dt.floor('h')
        hourly_aggregated = recent_5min_data.groupby('hour_floor').agg({
            'log_return': 'sum',  # Sum log returns for the hour
            'timestamp': 'first'
        }).reset_index()
        
        if len(hourly_aggregated) < self.lookback_window // 2:
            raise ValueError("Insufficient hourly data for scaling factor calculation")
            
        # Calculate actual hourly std
        actual_std = hourly_aggregated['log_return'].std()
        
        # Calculate template RMS over the same period
        template_stds = []
        for _, row in hourly_aggregated.iterrows():
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
        
        # Project forward using hourly template with scaling
        total_log_drift = 0.0
        total_log_variance = 0.0
        
        # For fractional hours, we need to handle the calculation carefully
        full_hours = int(hours_ahead)
        remaining_fraction = hours_ahead - full_hours
        
        current_timestamp = current_time
        
        # Process full hours
        for i in range(full_hours):
            # Get template std for this hour and scale it
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Add to cumulative drift and variance
            total_log_drift += self.overall_mean_log_return
            total_log_variance += scaled_std ** 2
            
            # Move to next hour
            current_timestamp += timedelta(hours=1)
        
        # Handle remaining fraction of hour
        if remaining_fraction > 0:
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Scale by the fraction of the hour
            total_log_drift += self.overall_mean_log_return * remaining_fraction
            total_log_variance += (scaled_std ** 2) * remaining_fraction
        
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
        if (math.isinf(upper)):
            return self.p_above_target(
                current_time=current_time,
                current_price=current_price,
                target_price=lower,
                hours_ahead=hours_ahead
            )
        if(lower <= 0):
            lower = 0
        


        scaling_factor = self._calculate_scaling_factor(current_time)
        
        # Project forward using hourly template with scaling
        total_log_drift = 0.0
        total_log_variance = 0.0
        
        # For fractional hours, we need to handle the calculation carefully
        full_hours = int(hours_ahead)
        remaining_fraction = hours_ahead - full_hours
        
        current_timestamp = current_time
        
        # Process full hours
        for i in range(full_hours):
            # Get template std for this hour and scale it
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Add to cumulative drift and variance
            total_log_drift += self.overall_mean_log_return
            total_log_variance += scaled_std ** 2
            
            # Move to next hour
            current_timestamp += timedelta(hours=1)
        
        # Handle remaining fraction of hour
        if remaining_fraction > 0:
            template_std = self._get_template_std(current_timestamp)
            scaled_std = template_std * scaling_factor
            
            # Scale by the fraction of the hour
            total_log_drift += self.overall_mean_log_return * remaining_fraction
            total_log_variance += (scaled_std ** 2) * remaining_fraction
        
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
            'template_intervals': len(self.hourly_template) if self.hourly_template is not None else 0,
            'template_type': 'hourly_aggregated_from_5min'
        }
    
    def get_template_comparison(self) -> pd.DataFrame:
        """Get the hourly template for inspection."""
        if self.hourly_template is None:
            raise ValueError("Hourly template not created")
        return self.hourly_template.copy()
