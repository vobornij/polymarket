"""
Visualization module for Polymarket analysis results.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from ..strategies.arbitrage_detector import ArbitrageSignal
from ..strategies.strategy_backtester import StrategyPerformance, Trade
from ..utils.logger import get_default_logger
from ..utils.config import config


class PolymarketVisualizer:
    """
    Creates visualizations for Polymarket analysis results.
    """
    
    def __init__(self, style: str = 'plotly_dark'):
        """
        Initialize the visualizer.
        
        Args:
            style: Plotting style to use
        """
        self.logger = get_default_logger()
        self.style = style
        
        # Set matplotlib style
        plt.style.use('dark_background' if 'dark' in style else 'default')
        
        # Set seaborn style
        sns.set_theme(style='darkgrid' if 'dark' in style else 'whitegrid')
    
    def plot_market_overview(
        self,
        markets_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an overview visualization of markets.
        
        Args:
            markets_df: DataFrame with market information
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating market overview visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Volume Distribution',
                'Liquidity vs Volume',
                'Markets by Tag',
                'Active vs Inactive Markets'
            ],
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Volume distribution
        fig.add_trace(
            go.Histogram(
                x=markets_df['volume'],
                nbinsx=30,
                name='Volume Distribution',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # Liquidity vs Volume scatter
        fig.add_trace(
            go.Scatter(
                x=markets_df['volume'],
                y=markets_df['liquidity'],
                mode='markers',
                name='Liquidity vs Volume',
                marker=dict(
                    size=8,
                    color=markets_df['avg_price'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Price")
                ),
                text=markets_df['question'].str[:50] + '...',
                hovertemplate='<b>%{text}</b><br>Volume: %{x}<br>Liquidity: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Markets by tag (top 10)
        if 'tags' in markets_df.columns:
            tag_counts = markets_df['tags'].str.split(', ').explode().value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=tag_counts.values,
                    y=tag_counts.index,
                    orientation='h',
                    name='Markets by Tag',
                    marker_color='lightcoral'
                ),
                row=2, col=1
            )
        
        # Active vs Inactive
        active_counts = markets_df['active'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Active', 'Inactive'],
                values=[active_counts.get(True, 0), active_counts.get(False, 0)],
                name='Market Status',
                marker_colors=['green', 'red']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Polymarket Overview Dashboard',
            template=self.style,
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Volume", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Volume", row=1, col=2)
        fig.update_yaxes(title_text="Liquidity", row=1, col=2)
        fig.update_xaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Tag", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Market overview saved to {save_path}")
        
        return fig
    
    def plot_price_history(
        self,
        price_df: pd.DataFrame,
        market_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot price history for selected markets.
        
        Args:
            price_df: DataFrame with price data
            market_ids: List of market IDs to plot (plots top 5 by volume if None)
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating price history visualization...")
        
        if market_ids is None:
            # Select top markets by volume
            volume_by_market = price_df.groupby('market_id')['volume'].sum()
            market_ids = volume_by_market.nlargest(5).index.tolist()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, market_id in enumerate(market_ids):
            market_data = price_df[price_df['market_id'] == market_id].copy()
            
            if market_data.empty:
                continue
            
            market_data = market_data.sort_index()
            
            # Plot price line
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['price'],
                    mode='lines',
                    name=f'Market {market_id[:8]}...',
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate='<b>Market %{fullData.name}</b><br>Time: %{x}<br>Price: %{y:.4f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='Price History - Top Markets by Volume',
            xaxis_title='Time',
            yaxis_title='Price',
            template=self.style,
            height=600,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Price history saved to {save_path}")
        
        return fig
    
    def plot_arbitrage_signals(
        self,
        signals: List[ArbitrageSignal],
        price_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize arbitrage signals on price charts.
        
        Args:
            signals: List of arbitrage signals
            price_df: Price data
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating arbitrage signals visualization...")
        
        # Group signals by market
        signals_by_market = {}
        for signal in signals:
            if signal.market_id not in signals_by_market:
                signals_by_market[signal.market_id] = []
            signals_by_market[signal.market_id].append(signal)
        
        # Select top markets with signals
        market_ids = list(signals_by_market.keys())[:3]  # Top 3 for clarity
        
        fig = make_subplots(
            rows=len(market_ids), cols=1,
            subplot_titles=[f'Market {mid[:12]}...' for mid in market_ids],
            shared_xaxes=True
        )
        
        colors = {
            'price_divergence': 'red',
            'mean_reversion': 'blue',
            'momentum': 'green',
            'ml_prediction': 'purple'
        }
        
        for i, market_id in enumerate(market_ids, 1):
            # Plot price history
            market_data = price_df[price_df['market_id'] == market_id].copy()
            market_data = market_data.sort_index()
            
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['price'],
                    mode='lines',
                    name=f'Price',
                    line=dict(color='white', width=1),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Add signals
            market_signals = signals_by_market[market_id]
            
            for signal_type in colors.keys():
                type_signals = [s for s in market_signals if s.signal_type == signal_type]
                
                if type_signals:
                    fig.add_trace(
                        go.Scatter(
                            x=[s.timestamp for s in type_signals],
                            y=[s.current_price for s in type_signals],
                            mode='markers',
                            name=signal_type.replace('_', ' ').title(),
                            marker=dict(
                                color=colors[signal_type],
                                size=10,
                                symbol='triangle-up'
                            ),
                            showlegend=(i == 1),
                            hovertemplate=f'<b>{signal_type}</b><br>'
                                        'Time: %{x}<br>'
                                        'Price: %{y:.4f}<br>'
                                        'Confidence: %{customdata[0]:.2f}<br>'
                                        'Potential Return: %{customdata[1]:.2%}<extra></extra>',
                            customdata=[[s.confidence, s.potential_return] for s in type_signals]
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            title='Arbitrage Signals on Price Charts',
            template=self.style,
            height=200 * len(market_ids) + 100,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        for i in range(1, len(market_ids) + 1):
            fig.update_yaxes(title_text="Price", row=i, col=1)
        
        fig.update_xaxes(title_text="Time", row=len(market_ids), col=1)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Arbitrage signals plot saved to {save_path}")
        
        return fig
    
    def plot_strategy_performance(
        self,
        performance: StrategyPerformance,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive strategy performance visualization.
        
        Args:
            performance: Strategy performance metrics
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating strategy performance visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Equity Curve',
                'Trade P&L Distribution',
                'Win/Loss Analysis',
                'Monthly Returns',
                'Risk Metrics',
                'Trade Duration Distribution'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}, {'type': 'histogram'}]]
        )
        
        if performance.trade_details:
            # Extract trade data
            trade_df = pd.DataFrame([
                {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'pnl': trade.pnl,
                    'duration': (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else 0,
                    'return': trade.pnl / (trade.position_size * trade.entry_price) if trade.position_size * trade.entry_price > 0 else 0
                }
                for trade in performance.trade_details
            ])
            
            # 1. Equity Curve (simulated)
            cumulative_pnl = trade_df['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=trade_df['entry_time'],
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # 2. P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=trade_df['pnl'],
                    nbinsx=20,
                    name='P&L Distribution',
                    marker_color='skyblue'
                ),
                row=1, col=2
            )
            
            # 3. Win/Loss Analysis
            wins = trade_df[trade_df['pnl'] > 0]['pnl'].sum()
            losses = abs(trade_df[trade_df['pnl'] < 0]['pnl'].sum())
            
            fig.add_trace(
                go.Bar(
                    x=['Wins', 'Losses'],
                    y=[wins, losses],
                    name='Win/Loss',
                    marker_color=['green', 'red']
                ),
                row=1, col=3
            )
            
            # 4. Monthly Returns (if enough data)
            if len(trade_df) > 5:
                trade_df['month'] = trade_df['entry_time'].dt.to_period('M')
                monthly_pnl = trade_df.groupby('month')['pnl'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_pnl.index.astype(str),
                        y=monthly_pnl.values,
                        name='Monthly P&L',
                        marker_color='orange'
                    ),
                    row=2, col=1
                )
            
            # 5. Risk Metrics
            metrics_names = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate']
            metrics_values = [
                performance.sharpe_ratio,
                abs(performance.max_drawdown),
                performance.win_rate
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    name='Risk Metrics',
                    marker_color=['blue', 'red', 'green']
                ),
                row=2, col=2
            )
            
            # 6. Trade Duration Distribution
            fig.add_trace(
                go.Histogram(
                    x=trade_df['duration'],
                    nbinsx=15,
                    name='Duration (hours)',
                    marker_color='purple'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=f'Strategy Performance Dashboard<br>'
                  f'<sub>Total Return: {performance.total_return:.1%} | '
                  f'Win Rate: {performance.win_rate:.1%} | '
                  f'Sharpe: {performance.sharpe_ratio:.2f}</sub>',
            template=self.style,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Strategy performance plot saved to {save_path}")
        
        return fig
    
    def plot_correlation_analysis(
        self,
        aligned_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create correlation analysis between Polymarket and crypto prices.
        
        Args:
            aligned_df: DataFrame with aligned price data
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating correlation analysis visualization...")
        
        # Calculate correlations
        crypto_columns = [col for col in aligned_df.columns if col.startswith('crypto_')]
        
        if not crypto_columns:
            self.logger.warning("No crypto price columns found for correlation analysis")
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Price Correlation Heatmap',
                'Price vs Crypto Scatter',
                'Rolling Correlation',
                'Price Ratio Analysis'
            ]
        )
        
        # 1. Correlation Heatmap
        corr_data = aligned_df[['price'] + crypto_columns].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. Scatter plot with most correlated crypto
        if len(crypto_columns) > 0:
            # Find most correlated crypto
            correlations = aligned_df['price'].corr(aligned_df[crypto_columns])
            best_crypto = correlations.abs().idxmax()
            
            fig.add_trace(
                go.Scatter(
                    x=aligned_df[best_crypto],
                    y=aligned_df['price'],
                    mode='markers',
                    name=f'Price vs {best_crypto}',
                    marker=dict(size=4, opacity=0.6),
                    hovertemplate=f'{best_crypto}: %{{x}}<br>Polymarket Price: %{{y}}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Rolling Correlation
            rolling_corr = aligned_df['price'].rolling(window=24).corr(aligned_df[best_crypto])
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name=f'Rolling Correlation with {best_crypto}',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            # 4. Price Ratio
            price_ratio = aligned_df['price'] / aligned_df[best_crypto]
            
            fig.add_trace(
                go.Scatter(
                    x=price_ratio.index,
                    y=price_ratio.values,
                    mode='lines',
                    name=f'Price Ratio (Polymarket/{best_crypto})',
                    line=dict(color='green')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Polymarket vs Crypto Price Correlation Analysis',
            template=self.style,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Correlation analysis plot saved to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        markets_df: pd.DataFrame,
        price_df: pd.DataFrame,
        signals: List[ArbitrageSignal],
        performance: Optional[StrategyPerformance] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            markets_df: Market information DataFrame
            price_df: Price data DataFrame
            signals: List of arbitrage signals
            performance: Strategy performance (optional)
            save_dir: Directory to save plots
        
        Returns:
            Dictionary of figure objects
        """
        self.logger.info("Creating comprehensive dashboard...")
        
        config.ensure_directories()
        
        if save_dir is None:
            save_dir = config.processed_data_dir / "visualizations"
        
        # Ensure the save directory exists
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # Market Overview
        figures['market_overview'] = self.plot_market_overview(
            markets_df,
            save_path=f"{save_dir}/market_overview.html" if save_dir else None
        )
        
        # Price History
        figures['price_history'] = self.plot_price_history(
            price_df,
            save_path=f"{save_dir}/price_history.html" if save_dir else None
        )
        
        # Arbitrage Signals
        figures['arbitrage_signals'] = self.plot_arbitrage_signals(
            signals,
            price_df,
            save_path=f"{save_dir}/arbitrage_signals.html" if save_dir else None
        )
        
        # Strategy Performance (if provided)
        if performance:
            figures['strategy_performance'] = self.plot_strategy_performance(
                performance,
                save_path=f"{save_dir}/strategy_performance.html" if save_dir else None
            )
        
        self.logger.info(f"Dashboard created with {len(figures)} visualizations")
        return figures
