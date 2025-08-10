#!/usr/bin/env python3
"""
Demo script showing the difference between standard GBM and time-varying variance GBM
"""

from datetime import datetime, timedelta
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polymarket_analysis.analytics.gbm_5min_estimator import GBMParameter5MinEstimator

def demo_time_varying_variance():
    """Demonstrate the time-varying variance approach."""
    
    print("Demonstrating Time-Varying Variance GBM...")
    
    # Create estimator using default data file
    estimator = GBMParameter5MinEstimator()
    
    # Get data summary
    summary = estimator.get_data_summary()
    print(f"Loaded {summary['data_points']} data points")
    print(f"Created template with {summary['template_intervals']} intervals")
    
    # Test probability calculations at different times of day
    current_price = 50000.0
    target_price = 51000.0  # 2% move instead of 10%
    hours_ahead = 24.0
    
    # Test at different hours to see how probabilities change
    base_time = datetime(2025, 8, 6, 0, 0, 0)  # Start at midnight
    
    results = []
    
    for hour in range(24):
        test_time = base_time.replace(hour=hour)
        
        prob_above = estimator.p_above_target(
            current_time=test_time,
            current_price=current_price,
            target_price=target_price,
            hours_ahead=hours_ahead
        )
        
        scaling_factor = estimator._calculate_scaling_factor(test_time)
        template_std = estimator._get_template_std(test_time)
        
        results.append({
            'hour': hour,
            'prob_above': prob_above,
            'scaling_factor': scaling_factor,
            'template_std': template_std,
            'scaled_std': template_std * scaling_factor
        })
        
        print(f"Hour {hour:2d}: P(>{target_price}) = {prob_above:.4f}, "
              f"Scale = {scaling_factor:.3f}, "
              f"Template σ = {template_std:.6f}, "
              f"Scaled σ = {template_std * scaling_factor:.6f}")
    
    # Show summary statistics
    probs = [r['prob_above'] for r in results]
    scales = [r['scaling_factor'] for r in results]
    template_stds = [r['template_std'] for r in results]
    
    print(f"\nSummary over 24 hours:")
    print(f"Probability range: {min(probs):.4f} - {max(probs):.4f}")
    print(f"Scaling factor range: {min(scales):.3f} - {max(scales):.3f}")
    print(f"Template σ range: {min(template_stds):.6f} - {max(template_stds):.6f}")
    
    print(f"\nDemonstration completed!")
    print(f"The time-varying approach uses:")
    print(f"1. Weekly template of volatility patterns")
    print(f"2. Real-time scaling based on recent market conditions") 
    print(f"3. Time-specific probability calculations")

if __name__ == "__main__":
    demo_time_varying_variance()
