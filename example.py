#!/usr/bin/env python3
"""
Polymarket Analysis Example Script

A simple example demonstrating the core functionality of the Polymarket analysis framework.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from polymarket_analysis.data.data_collector import DataCollector
from polymarket_analysis.data.data_processor import DataProcessor
from polymarket_analysis.strategies.arbitrage_detector import ArbitrageDetector
from polymarket_analysis.strategies.strategy_backtester import StrategyBacktester
from polymarket_analysis.visualization.dashboard import PolymarketVisualizer
from polymarket_analysis.utils.config import config
from polymarket_analysis.utils.logger import get_default_logger


async def main():
    """Run a complete analysis workflow."""
    logger = get_default_logger()
    logger.info("Starting Polymarket analysis example...")
    
    try:
        # Ensure directories exist
        config.ensure_directories()
        
        # Initialize components
        collector = DataCollector()
        processor = DataProcessor()
        detector = ArbitrageDetector()
        backtester = StrategyBacktester(initial_capital=10000)
        visualizer = PolymarketVisualizer()
        
        logger.info("Components initialized successfully")
        
        # Note: This example uses mock data since real API calls require keys
        logger.info("Creating mock data for demonstration...")
        
        # Create mock data (replace with real data collection in production)
        from datetime import datetime, timedelta
        import random
        import pandas as pd
        import numpy as np
        from polymarket_analysis.api.polymarket_client import Market, PricePoint
        
        # Mock markets
        markets = [
            Market(
                id=f"market_{i}",
                question=f"Will Bitcoin reach ${40000 + i*1000} by end of 2025?",
                description="Bitcoin price prediction market",
                end_date=datetime.now() + timedelta(days=30),
                outcome_prices={"Yes": 0.6 + random.uniform(-0.2, 0.2), "No": 0.4 + random.uniform(-0.2, 0.2)},
                volume=random.uniform(1000, 10000),
                liquidity=random.uniform(500, 5000),
                active=True,
                tags=["crypto", "bitcoin"]
            )
            for i in range(5)
        ]
        
        # Mock price histories
        histories = {}
        base_time = datetime.now() - timedelta(days=7)
        
        for market in markets:
            price_points = []
            current_price = 0.5
            
            for hour in range(168):  # 7 days * 24 hours
                timestamp = base_time + timedelta(hours=hour)
                current_price += random.uniform(-0.05, 0.05)
                current_price = max(0.01, min(0.99, current_price))
                
                price_points.append(PricePoint(
                    market_id=market.id,
                    outcome="Yes",
                    price=current_price,
                    timestamp=timestamp,
                    volume=random.uniform(10, 100)
                ))
            
            histories[market.id] = price_points
        
        # Mock crypto prices
        timestamps = pd.date_range(base_time, periods=168, freq='H')
        crypto_prices = pd.DataFrame({
            'BTC-USD': np.cumsum(np.random.randn(168) * 100) + 50000,
            'ETH-USD': np.cumsum(np.random.randn(168) * 50) + 3000,
        }, index=timestamps)
        
        logger.info(f"Created mock data: {len(markets)} markets, {sum(len(h) for h in histories.values())} price points")
        
        # Process data
        logger.info("Processing data...")
        markets_df = processor.create_market_dataframe(markets)
        price_df = processor.create_price_dataframe(histories, outcome_filter="Yes")
        price_features_df = processor.calculate_price_features(price_df)
        aligned_df = processor.align_crypto_prices(price_features_df, crypto_prices)
        
        logger.info(f"Data processed: {price_features_df.shape[0]} price records with {price_features_df.shape[1]} features")
        
        # Detect arbitrage signals
        logger.info("Detecting arbitrage opportunities...")
        
        mean_reversion_signals = detector.detect_mean_reversion_opportunities(price_features_df)
        momentum_signals = detector.detect_momentum_opportunities(price_features_df)
        
        all_signals = mean_reversion_signals + momentum_signals
        logger.info(f"Detected {len(all_signals)} arbitrage signals ({len(mean_reversion_signals)} mean reversion, {len(momentum_signals)} momentum)")
        
        # Backtest strategy
        if all_signals:
            logger.info("Running strategy backtest...")
            performance = backtester.backtest_strategy(all_signals, price_features_df)
            
            logger.info("=== BACKTEST RESULTS ===")
            logger.info(f"Total Trades: {performance.total_trades}")
            logger.info(f"Win Rate: {performance.win_rate:.1%}")
            logger.info(f"Total Return: {performance.total_return:.1%}")
            logger.info(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {performance.max_drawdown:.1%}")
        else:
            logger.info("No signals generated - skipping backtest")
            performance = None
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        dashboard_figures = visualizer.create_dashboard(
            markets_df=markets_df,
            price_df=price_df,
            signals=all_signals,
            performance=performance,
            save_dir=config.processed_data_dir / "example_output"
        )
        
        logger.info(f"Created {len(dashboard_figures)} visualizations")
        
        # Save processed data
        logger.info("Saving results...")
        processor.save_processed_data(
            market_df=markets_df,
            price_df=price_features_df,
            aligned_df=aligned_df if not aligned_df.empty else None
        )
        
        logger.info("Example analysis completed successfully!")
        logger.info(f"Check {config.processed_data_dir} for output files")
        logger.info(f"Check {config.processed_data_dir / 'example_output'} for visualizations")
        
        return {
            'markets': len(markets),
            'signals': len(all_signals),
            'performance': performance,
            'visualizations': len(dashboard_figures)
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    result = asyncio.run(main())
    print("\n🎉 Example completed successfully!")
    print(f"📊 Analyzed {result['markets']} markets")
    print(f"🔍 Found {result['signals']} arbitrage signals")
    if result['performance']:
        print(f"💰 Strategy return: {result['performance'].total_return:.1%}")
    print(f"📈 Created {result['visualizations']} visualizations")
    print("\nNext steps:")
    print("1. Set up your API keys in .env file")
    print("2. Run the Jupyter notebook: notebooks/polymarket_analysis_demo.ipynb")
    print("3. Explore the generated visualizations and data files")
