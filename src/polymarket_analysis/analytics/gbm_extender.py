from datetime import datetime, timezone
import pandas as pd
# from .gbm_param_estimator import GBMParameterEstimator
from .gbm_hourly_template_estimator import GBMParameterHourlyTemplateEstimator
from polymarket_analysis.data.model.polymarket_contract import PriceHistory

class GbmExtender:
    def __init__(self, gbm: GBMParameterHourlyTemplateEstimator):
        self.gbm = gbm

    def extend_price_history(self, polymarket_history: PriceHistory, price_df: pd.DataFrame):
        """
        Extend price history with estimated probabilities - fully vectorized.
        """
        
        # Use merge_asof for efficient timestamp matching
        merged = pd.merge_asof(
            polymarket_history.price_df,
            price_df[['timestamp', 'price']].rename(columns={'price': 'btc_price'}),
            on='timestamp',
            direction='backward'
        )
        
        # Create mask for valid calculations
        valid_mask = (
            ~merged['btc_price'].isna()  
            # (days_ahead > 0)
        )
        
        def get_price_estimate(row) -> float:
            # p = self.gbm.p_in_range(
            #         row['btc_price'], 
            #         polymarket_history.contract.price_range, 
            #         (polymarket_history.contract.target_time - row['timestamp']).total_seconds() / (24 * 3600)
            #     )
            hours_ahead=(polymarket_history.contract.target_time - row['timestamp']).total_seconds() / (24 * 3600)
            if hours_ahead <= 0:
                if(polymarket_history.contract.outcome == "Yes"):
                    return 1
                elif(polymarket_history.contract.outcome == "No"):
                    return 0

            p = self.gbm.p_between_targets(
                current_time=row['timestamp'],
                current_price=row['btc_price'], 
                lower=polymarket_history.contract.price_range.lower,
                upper=polymarket_history.contract.price_range.upper, 
                hours_ahead=(polymarket_history.contract.target_time - row['timestamp']).total_seconds() / (24 * 3600)
            )
            if polymarket_history.contract.outcome == "Yes":
                return p
            elif polymarket_history.contract.outcome == "No":
                return 1.0 - p
            else:
                raise ValueError(f"Unknown outcome: {polymarket_history.contract.outcome}. Expected 'Yes' or 'No'.")

        
        # Calculate probabilities only for valid rows
        if valid_mask.any():
            valid_data = merged[valid_mask]
            
            probabilities = valid_data.apply(
                get_price_estimate,
                axis=1
            )
            
            merged.loc[valid_mask, 'fair_price'] = probabilities
        
        old_len = len(polymarket_history.price_df)
        # Update the price_df in history
        polymarket_history.price_df = merged.drop('btc_price', axis=1)

        assert len(polymarket_history.price_df) == old_len, "Length of price_df should remain unchanged after merging."
        
        return polymarket_history