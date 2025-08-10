# Polymarket Crypto Analysis Framework

A comprehensive Python framework for analyzing Polymarket historical data to identify statistical arbitrage opportunities in crypto pricing contracts.

## 🚀 Features

- **Data Collection**: Automated collection from Polymarket API and crypto reference prices
- **Signal Detection**: Multiple arbitrage detection strategies (mean reversion, momentum, price divergence)
- **Strategy Backtesting**: Comprehensive backtesting framework with performance metrics
- **Visualization**: Interactive dashboards and charts using Plotly and Matplotlib
- **Machine Learning**: ML-based price prediction models for enhanced signal generation
- **Risk Management**: Built-in position sizing, stop-loss, and take-profit mechanisms

## 📁 Project Structure

```
polymarket-analysis/
├── src/polymarket_analysis/          # Main package
│   ├── api/                          # Polymarket API client
│   ├── data/                         # Data collection and processing
│   ├── strategies/                   # Arbitrage detection and backtesting
│   ├── utils/                        # Configuration and logging utilities
│   └── visualization/                # Plotting and dashboard creation
├── notebooks/                        # Jupyter notebooks for analysis
├── data/                            # Data storage
│   ├── raw/                         # Raw collected data
│   ├── processed/                   # Processed datasets
│   └── models/                      # Trained ML models
├── tests/                           # Unit tests
├── config/                          # Configuration files
└── logs/                           # Application logs
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd polymarket-analysis
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Verify installation**:
   ```bash
   python -c "from polymarket_analysis import Config; print('Installation successful!')"
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# API Configuration
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_BASE_URL=https://clob.polymarket.com
API_RATE_LIMIT=10
API_TIMEOUT=30

# Analysis Settings
LOOKBACK_DAYS=30
MIN_VOLUME_THRESHOLD=1000
ARBITRAGE_THRESHOLD=0.02
CONFIDENCE_LEVEL=0.95

# Crypto symbols for reference prices
CRYPTO_SYMBOLS=BTC-USD,ETH-USD,ADA-USD,SOL-USD,DOGE-USD
```

### API Setup

1. **Polymarket API**: Get your API key from [Polymarket](https://polymarket.com)
2. **Yahoo Finance**: No API key required (uses yfinance library)

## 📊 Quick Start

### 1. Basic Data Collection

```python
import asyncio
from polymarket_analysis.data.data_collector import DataCollector

# Initialize collector
collector = DataCollector()

# Collect crypto markets and price data
async def collect_data():
    markets, histories, crypto_prices = await collector.collect_complete_dataset(
        days_back=30,
        min_volume=1000
    )
    return markets, histories, crypto_prices

# Run collection
markets, histories, crypto_prices = asyncio.run(collect_data())
```

### 2. Signal Detection

```python
from polymarket_analysis.strategies.arbitrage_detector import ArbitrageDetector
from polymarket_analysis.data.data_processor import DataProcessor

# Process data
processor = DataProcessor()
price_df = processor.create_price_dataframe(histories)
price_features_df = processor.calculate_price_features(price_df)

# Detect arbitrage signals
detector = ArbitrageDetector()
signals = detector.detect_mean_reversion_opportunities(price_features_df)
```

### 3. Strategy Backtesting

```python
from polymarket_analysis.strategies.strategy_backtester import StrategyBacktester

# Initialize backtester
backtester = StrategyBacktester(
    initial_capital=10000,
    max_position_size=0.1,
    transaction_cost=0.01
)

# Run backtest
performance = backtester.backtest_strategy(signals, price_features_df)
print(f"Total Return: {performance.total_return:.1%}")
print(f"Win Rate: {performance.win_rate:.1%}")
print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
```

### 4. Visualization

```python
from polymarket_analysis.visualization.dashboard import PolymarketVisualizer

# Create visualizations
visualizer = PolymarketVisualizer()
markets_df = processor.create_market_dataframe(markets)

# Generate dashboard
dashboard = visualizer.create_dashboard(
    markets_df=markets_df,
    price_df=price_df,
    signals=signals,
    performance=performance
)
```

## 📓 Jupyter Notebooks

The `notebooks/` directory contains example analyses:

- **`polymarket_analysis_demo.ipynb`**: Complete workflow demonstration
- **`signal_optimization.ipynb`**: Parameter optimization examples
- **`market_analysis.ipynb`**: Market-specific deep dives

To run notebooks:
```bash
poetry run jupyter lab notebooks/
```

## 🔍 Analysis Strategies

### 1. Mean Reversion Strategy
Identifies when market prices deviate significantly from their historical mean and generates signals expecting reversion.

### 2. Momentum Strategy
Detects strong price trends with volume confirmation and generates signals expecting continuation.

### 3. Price Divergence Strategy
Compares Polymarket prices with crypto reference prices to identify arbitrage opportunities.

### 4. Machine Learning Strategy
Uses trained models to predict future prices and generate signals based on expected price movements.

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility
- **Trade Metrics**: Win rate, profit factor, average trade duration
- **Advanced Metrics**: Calmar ratio, Sortino ratio, tail ratio

## 🧪 Testing

Run the test suite:
```bash
poetry run pytest tests/
```

Run tests with coverage:
```bash
poetry run pytest tests/ --cov=polymarket_analysis --cov-report=html
```

## 🚀 Advanced Usage

### Parameter Optimization

```python
# Optimize strategy parameters
parameter_ranges = {
    'max_position_size': [0.05, 0.10, 0.15],
    'stop_loss': [0.03, 0.05, 0.07],
    'take_profit': [0.08, 0.10, 0.12]
}

optimization_result = backtester.optimize_parameters(
    signals=signals,
    price_data=price_df,
    parameter_ranges=parameter_ranges
)
```

### Walk-Forward Analysis

```python
# Test strategy robustness over time
wf_results = backtester.run_walk_forward_analysis(
    signals=signals,
    price_data=price_df,
    train_period_days=30,
    test_period_days=7
)
```

### Custom Signal Development

```python
# Extend ArbitrageDetector for custom strategies
class CustomDetector(ArbitrageDetector):
    def detect_custom_opportunities(self, data):
        # Implement your custom logic
        signals = []
        # ... custom signal logic ...
        return signals
```

## 🔧 Configuration Options

### Data Collection Settings
- `LOOKBACK_DAYS`: Historical data period
- `MIN_VOLUME_THRESHOLD`: Minimum market volume filter
- `API_RATE_LIMIT`: API request rate limiting

### Strategy Settings
- `ARBITRAGE_THRESHOLD`: Minimum price discrepancy
- `CONFIDENCE_LEVEL`: Statistical confidence threshold
- Position sizing methods: 'fixed', 'proportional', 'kelly'

### Risk Management
- Stop-loss and take-profit levels
- Maximum position sizes
- Maximum holding periods

## 📚 Dependencies

Core dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **aiohttp**: Async HTTP client
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning
- **yfinance**: Crypto price data
- **jupyter**: Interactive notebooks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `poetry run pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This framework is for educational and research purposes only. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the `docs/` directory for detailed documentation
- **Examples**: See `notebooks/` for usage examples

## 🏗️ Roadmap

- [ ] Real-time data streaming
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Multi-exchange arbitrage detection
- [ ] Portfolio optimization
- [ ] Automated execution framework
- [ ] Web-based dashboard
- [ ] Mobile notifications

---

**Happy Trading! 🚀📈**
