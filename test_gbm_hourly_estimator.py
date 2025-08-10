#!/usr/bin/env python3
"""
Test script for GBMParameterHourlyEstimator
"""

from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polymarket_analysis.analytics.gbm_5min_estimator import GBMParameter5MinEstimator

def test_gbm_hourly_estimator():
    """Test the GBMParameterHourlyEstimator implementation."""
    
    print("Testing GBMParameterHourlyEstimator...")
    
    # Create estimator with recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Use 3 months of data for testing
    
    try:
        estimator = GBMParameter5MinEstimator(
            data_start_date=start_date,
            data_end_date=end_date
        )
        
        print(f"✓ Successfully initialized estimator")
        print(f"  - Data loaded: {len(estimator.btc_data)} rows")
        print(f"  - Weekly template: {len(estimator.weekly_template)} intervals")
        print(f"  - Overall mean log return: {estimator.overall_mean_log_return:.8f}")
        
        # Test probability calculations
        current_time = datetime.now()
        current_price = 50000.0  # Example BTC price
        target_price = 55000.0   # 10% higher
        hours_ahead = 24.0       # 1 day ahead
        
        prob_above = estimator.p_above_target(
            current_time=current_time,
            current_price=current_price,
            target_price=target_price,
            hours_ahead=hours_ahead
        )
        
        print(f"✓ P(price > {target_price} in {hours_ahead}h): {prob_above:.4f}")
        
        # Test probability between targets
        lower_price = 48000.0
        upper_price = 52000.0
        
        prob_between = estimator.p_between_targets(
            current_time=current_time,
            current_price=current_price,
            lower=lower_price,
            upper=upper_price,
            hours_ahead=hours_ahead
        )
        
        print(f"✓ P({lower_price} < price < {upper_price} in {hours_ahead}h): {prob_between:.4f}")
        
        # Test scaling factor calculation
        scaling_factor = estimator._calculate_scaling_factor(current_time)
        print(f"✓ Current scaling factor: {scaling_factor:.4f}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gbm_hourly_estimator()
