import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from .gbm_param_estimator import GBMParameterEstimator
import warnings
warnings.filterwarnings('ignore')

class GBMBacktester:
    def __init__(self, symbol="BTC-USD", lookback_days=365, test_days=180):
        """
        Initialize GBM backtester
        
        Args:
            symbol: Trading symbol
            lookback_days: Days to use for parameter estimation
            test_days: Days to backtest predictions
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.test_days = test_days
        self.full_data = None
        self.backtest_results = []
        
    def fetch_extended_data(self):
        """Fetch extended historical data for backtesting"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=f"{self.lookback_days}d")
            self.full_data = data['Close']
            print(f"Fetched {len(self.full_data)} price points for backtesting")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def run_backtest(self, forecast_horizons=[7, 14, 30], target_changes=[0.05, 0.1, 0.15, -0.05, -0.1]):
        """
        Run backtest using only real historical data
        Records actual outcomes for each target change and horizon
        Uses non-overlapping intervals for each horizon
        """
        if self.full_data is None:
            raise ValueError("No data available. Call fetch_extended_data() first.")
        
        print("Running backtest on historical data (no simulations)")
        
        results = []
        
        # For each horizon, create non-overlapping test intervals
        for horizon in forecast_horizons:
            print(f"\nTesting {horizon}-day horizon...")
            
            # Start after training period
            start_idx = self.lookback_days
            end_idx = len(self.full_data) - horizon
            
            # Create non-overlapping intervals
            interval_starts = range(start_idx, min(end_idx, start_idx + self.test_days), horizon)
            
            for interval_start in interval_starts:
                interval_end = interval_start + horizon
                
                if interval_end >= len(self.full_data):
                    break
                
                start_price = self.full_data.iloc[interval_start]
                end_price = self.full_data.iloc[interval_end]
                actual_return = (end_price / start_price) - 1
                start_date = self.full_data.index[interval_start]
                
                # Test each target change for this interval
                for target_change in target_changes:
                    # Check if target was actually achieved
                    actual_achieved = actual_return >= target_change
                    
                    result = {
                        'date': start_date,
                        'horizon_days': horizon,
                        'target_change': target_change,
                        'start_price': start_price,
                        'end_price': end_price,
                        'actual_return': actual_return,
                        'actual_achieved': actual_achieved
                    }
                    
                    results.append(result)
            
            print(f"  Tested {len(interval_starts)} non-overlapping intervals")
        
        self.backtest_results = pd.DataFrame(results)
        print(f"\nBacktest completed with {len(self.backtest_results)} observations")
        return self.backtest_results
    
    def analyze_calibration(self, with_predictions=True):
        """Analyze probability calibration - compare predicted vs actual frequencies"""
        if len(self.backtest_results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        if with_predictions:
            df = self.get_gbm_predictions()
        else:
            df = self.backtest_results.copy()
        
        calibration_results = []
        
        # Analyze by horizon and target change
        for horizon in sorted(df['horizon_days'].unique()):
            for target_change in sorted(df['target_change'].unique()):
                subset = df[(df['horizon_days'] == horizon) & 
                           (df['target_change'] == target_change)]
                
                if len(subset) < 5:
                    continue
                
                actual_frequency = subset['actual_achieved'].mean()
                n_observations = len(subset)
                
                result = {
                    'horizon_days': horizon,
                    'target_change': target_change,
                    'actual_frequency': actual_frequency,
                    'n_observations': n_observations
                }
                
                if with_predictions and 'predicted_probability' in subset.columns:
                    predicted_prob = subset['predicted_probability'].mean()
                    result['predicted_probability'] = predicted_prob
                    result['difference'] = actual_frequency - predicted_prob
                
                calibration_results.append(result)
        
        return pd.DataFrame(calibration_results)
    
    def calculate_brier_score(self):
        """Calculate Brier score for probability predictions"""
        df = self.get_gbm_predictions()
        
        # Brier Score = mean((predicted_prob - actual_outcome)^2)
        brier_score = np.mean((df['predicted_probability'] - 
                             df['actual_achieved'].astype(int))**2)
        
        return brier_score
    
    def generate_simple_report(self):
        """Generate simple backtest report"""
        if len(self.backtest_results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        df = self.backtest_results
        print("=" * 60)
        print("HISTORICAL DATA ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"Total Observations: {len(df)}")
        
        # Historical frequency analysis
        calibration = self.analyze_calibration(with_predictions=False)
        
        print("\nActual Historical Frequencies:")
        print("=" * 40)
        
        for _, row in calibration.iterrows():
            horizon = row['horizon_days']
            target = row['target_change']
            actual_freq = row['actual_frequency']
            n = row['n_observations']
            
            print(f"\n{horizon}-day, {target:+.1%} target:")
            print(f"  Actual Frequency: {actual_freq:.3f} ({actual_freq:.1%})")
            print(f"  Sample Size: {n}")
        
        print("\n" + "="*60)
        print("GBM PROBABILITY CALIBRATION COMPARISON")
        print("="*60)
        
        gbm_calibration = self.analyze_calibration(with_predictions=True)
        brier_score = self.calculate_brier_score()
        print(f"Overall Brier Score: {brier_score:.4f}")
        
        print("\nGBM vs Historical Comparison:")
        print("=" * 40)
        
        for _, row in gbm_calibration.iterrows():
            horizon = row['horizon_days']
            target = row['target_change']
            pred_prob = row['predicted_probability']
            actual_freq = row['actual_frequency']
            diff = row['difference']
            n = row['n_observations']
            
            print(f"\n{horizon}-day, {target:+.1%} target:")
            print(f"  GBM Predicted: {pred_prob:.3f} ({pred_prob:.1%})")
            print(f"  Actual Frequency: {actual_freq:.3f} ({actual_freq:.1%})")
            print(f"  Difference: {diff:+.3f} ({diff:+.1%})")
            print(f"  Sample Size: {n}")
            
            if abs(diff) < 0.05:
                print(f"  ✓ Well calibrated")
            elif diff > 0:
                print(f"  ⚠ GBM underestimated (actual > predicted)")
            else:
                print(f"  ⚠ GBM overestimated (predicted > actual)")
    
    def plot_relative_errors(self):
        """
        Plot calibration results using Plotly - predicted probability vs actual frequency
        """
        if len(self.backtest_results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        calibration_df = self.analyze_calibration()
        
        if len(calibration_df) == 0:
            print("No calibration data available")
            return None
        
        # Create subplots for each horizon
        horizons = sorted(calibration_df['horizon_days'].unique())
        n_horizons = len(horizons)
        
        fig = make_subplots(
            rows=n_horizons, cols=1,
            shared_xaxes=True,
            subplot_titles=[f'{horizon}-day Forecast Horizon' for horizon in horizons],
            vertical_spacing=0.1
        )
        
        # Color palette
        target_changes = sorted(calibration_df['target_change'].unique())
        colors = px.colors.qualitative.Set1[:len(target_changes)]
        
        for i, horizon in enumerate(horizons, 1):
            subset = calibration_df[calibration_df['horizon_days'] == horizon]
            
            for j, target_change in enumerate(target_changes):
                target_subset = subset[subset['target_change'] == target_change]
                
                if len(target_subset) > 0:
                    row = target_subset.iloc[0]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[row['predicted_probability']],
                            y=[row['actual_frequency']],
                            mode='markers',
                            name=f"Target: {target_change:+.1%}",
                            marker=dict(
                                color=colors[j], 
                                size=max(10, min(30, row['n_observations']/2)),  # Size based on sample size
                                opacity=0.7
                            ),
                            showlegend=(i == 1),  # Only show legend for first subplot
                            hovertemplate=(
                                f"<b>Horizon: {horizon} days</b><br>" +
                                f"Target: {target_change:+.1%}<br>" +
                                "Predicted Prob: %{x:.3f}<br>" +
                                "Actual Freq: %{y:.3f}<br>" +
                                f"Sample Size: {row['n_observations']}<br>" +
                                f"Difference: {row['difference']:+.3f}<br>" +
                                "<extra></extra>"
                            )
                        ),
                        row=i, col=1
                    )
            
            # Add perfect calibration line (diagonal)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='red', width=2),
                    name='Perfect Calibration',
                    showlegend=(i == 1),
                    hoverinfo='skip'
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title="GBM Probability Calibration: Predicted vs Actual Frequencies",
            height=200 * n_horizons + 100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Predicted Probability", row=n_horizons, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Actual Frequency", range=[0, 1])
        
        return fig
    
    def get_gbm_predictions(self):
        """
        Get GBM probability predictions for the backtest results
        This is separate from the backtest data collection
        """
        if len(self.backtest_results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Estimate GBM parameters using last lookback period (most recent data)
        train_prices = self.full_data.iloc[-self.lookback_days:]
        
        estimator = GBMParameterEstimator(self.symbol)
        estimator.prices = train_prices
        estimator.calculate_returns()
        estimator.estimate_parameters()
        
        print(f"Using GBM parameters: μ={estimator.mu:.4f}, σ={estimator.sigma:.4f}")
        
        # Add predictions to existing results
        df = self.backtest_results.copy()
        predictions = []
        
        for _, row in df.iterrows():
            target_price = row['start_price'] * (1 + row['target_change'])
            
            # Temporarily set the estimator's current price for this prediction
            original_last_price = estimator.prices.iloc[-1] if len(estimator.prices) > 0 else None
            
            # Create a price series that ends with our start_price
            temp_prices = estimator.prices.copy()
            temp_prices.iloc[-1] = row['start_price']
            estimator.prices = temp_prices
            
            prob_result = estimator.probability_analysis(
                target_price=target_price, 
                current_price=row['start_price'],
                days_ahead=row['horizon_days']
            )
            predictions.append(prob_result['probability_above_target'])
            
            # Restore original prices if needed
            if original_last_price is not None:
                temp_prices.iloc[-1] = original_last_price
                estimator.prices = temp_prices
        
        # Create new DataFrame with predictions
        result_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            row_dict = row.to_dict()
            row_dict['predicted_probability'] = predictions[i]
            row_dict['mu'] = estimator.mu
            row_dict['sigma'] = estimator.sigma
            result_data.append(row_dict)
        
        return pd.DataFrame(result_data)