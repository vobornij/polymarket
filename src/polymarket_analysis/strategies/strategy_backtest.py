"""
Strategy backtesting module for Polymarket trading strategies.

This module provides comprehensive backtesting functionality that tracks positions,
realized PnL, and performance metrics over time for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

from .strategy import TradingStrategy, Order, OrderSide


@dataclass
class Trade:
    """Represents an executed trade with entry and exit information."""
    entry_timestamp: datetime
    entry_price: float
    entry_volume: float
    side: OrderSide
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class PositionState:
    """Represents the current position state at a point in time."""
    timestamp: datetime
    position: float  # Net position (positive = long, negative = short)
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    cash: float
    total_value: float
    current_price: float


class StrategyBacktester:
    """
    Comprehensive backtester for trading strategies.
    
    Tracks positions, PnL, and performance metrics over time, handling
    partial fills, position management, and final settlement.
    """
    
    def __init__(self, commission_rate: float = 0.0):
        """
        Initialize the backtester.
        
        Args:
            commission_rate: Commission rate per trade (e.g., 0.01 for 1%)
        """
        self.commission_rate = commission_rate
        
        # State tracking
        self.position = 0.0
        self.realized_pnl = 0.0
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        
        # Time series data
        self.position_history: List[PositionState] = []
        self.trade_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = {}
    
    def backtest(self, strategy: TradingStrategy, market_data: pd.DataFrame, 
                 final_outcome: float) -> pd.DataFrame:
        """
        Run a comprehensive backtest of the strategy.
        
        Args:
            strategy: Trading strategy to test
            market_data: Historical market data with required columns:
                - timestamp: Time of observation
                - price: Market price (0-1)
                - estimated_probability: Strategy's probability estimate
                - Additional columns as needed by strategy
            final_outcome: Final settlement value (0 or 1)
            
        Returns:
            DataFrame with time series of positions, PnL, and metrics
        """
        # Reset state
        self._reset_state()
        
        orders = strategy.generate_orders(market_data)

        # Process orders
        self._process_orders(orders)
        
        # Final settlement
        self._final_settlement(final_outcome, market_data.iloc[-1]['timestamp'])
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Return results as DataFrame
        return self._create_results_dataframe()
    
    
    def _reset_state(self):
        """Reset backtester state for new run."""
        self.cash = 0
        self.position = 0.0
        self.realized_pnl = 0.0
        self.open_trades = []
        self.closed_trades = []
        self.position_history = []
        self.trade_history = []
        self.metrics = {}
    
    def _process_orders(self, orders: List[Order]):
        """Process a single timestep in the backtest."""
        
        # Execute orders
        for order in orders:
            self._execute_order(order)

            print(f"Processing order: {order.side.value} {order.volume} at {order.price} on {order.timestamp}")
        
            # Calculate unrealized PnL
            unrealized_pnl = self._calculate_unrealized_pnl(order.price)
            total_pnl = self.realized_pnl + unrealized_pnl
            total_value = self.cash + self.position * order.price
            
            # Record position state
            position_state = PositionState(
                timestamp=order.timestamp,
                position=self.position,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.realized_pnl,
                total_pnl=total_pnl,
                cash=self.cash,
                total_value=total_value,
                current_price=order.price
            )
            
            self.position_history.append(position_state)
    
    def _execute_order(self, order: Order):
        """Execute an order and update positions."""
        # Calculate net cash flow
        if order.side == OrderSide.BUY:
            cash_flow = -(order.volume * order.price)
            position_change = order.volume
        else:  # SELL
            cash_flow = order.volume * order.price
            position_change = -order.volume
        
        # Update cash and position
        self.cash += cash_flow
        old_position = self.position
        self.position += position_change
        
        # Handle position changes and PnL calculation
        self._handle_position_change(order, old_position)
        
        # Record trade
        trade_record = {
            'timestamp': order.timestamp,
            'side': order.side.value,
            'volume': order.volume,
            'price': order.price,
            'cash_flow': cash_flow,
            'position_before': old_position,
            'position_after': self.position
        }
        
        self.trade_history.append(trade_record)
    
    def _handle_position_change(self, order: Order, old_position: float):
        """Handle position changes and calculate realized PnL."""
        
        if order.side == OrderSide.BUY:
            # print(f"Executing BUY order: {order.volume} at {order.price} on {order.timestamp}")
            trade = Trade(
                entry_timestamp=order.timestamp,
                entry_price=order.price,
                entry_volume=order.volume,
                side=order.side
            )
            self.open_trades.append(trade)
        
        else:
            # Reducing or reversing position - realize some PnL
            self._realize_pnl(order.volume, order.price, order.timestamp)
    
    def _realize_pnl(self, volume: float, exit_price: float, timestamp: datetime):
        """Realize PnL when closing positions (FIFO basis)."""
        remaining_volume = volume
        
        while remaining_volume > 0 and self.open_trades:
            trade = self.open_trades[0]
            
            if trade.entry_volume <= remaining_volume:
                # Close entire trade
                pnl = self._calculate_trade_pnl(trade, exit_price, trade.entry_volume)
                self.realized_pnl += pnl
                
                # Mark trade as closed
                trade.exit_timestamp = timestamp
                trade.exit_price = exit_price
                trade.realized_pnl = pnl
                
                print(f"Closing {trade}")

                self.closed_trades.append(trade)
                self.open_trades.pop(0)
                
                remaining_volume -= trade.entry_volume
            
            else:
                # Partial close
                pnl = self._calculate_trade_pnl(trade, exit_price, remaining_volume)
                self.realized_pnl += pnl
                
                # Create closed trade record
                closed_trade = Trade(
                    entry_timestamp=trade.entry_timestamp,
                    entry_price=trade.entry_price,
                    entry_volume=remaining_volume,
                    side=trade.side,
                    exit_timestamp=timestamp,
                    exit_price=exit_price,
                    realized_pnl=pnl
                )
                
                print(f"Partial close: {closed_trade}")

                self.closed_trades.append(closed_trade)
                
                # Reduce open trade volume
                trade.entry_volume -= remaining_volume
                remaining_volume = 0
    
    def _calculate_trade_pnl(self, trade: Trade, exit_price: float, volume: float) -> float:
        """Calculate PnL for a trade."""
        if trade.side == OrderSide.BUY:
            return volume * (exit_price - trade.entry_price)
        else:
            return volume * (trade.entry_price - exit_price)
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for open positions."""
        unrealized = 0.0
        
        for trade in self.open_trades:
            if trade.side == OrderSide.BUY:
                unrealized += trade.entry_volume * (current_price - trade.entry_price)
            else:
                unrealized += trade.entry_volume * (trade.entry_price - current_price)
        
        return unrealized
    
    def _final_settlement(self, final_outcome: float, final_timestamp: datetime):
        """Handle final settlement of all positions."""
        if self.position != 0:
            # Close all remaining positions at final outcome price
            total_volume = abs(self.position)
            
            # Realize PnL for all open trades
            for trade in self.open_trades:
                pnl = self._calculate_trade_pnl(trade, final_outcome, trade.entry_volume)
                self.realized_pnl += pnl
                
                trade.exit_timestamp = final_timestamp
                trade.exit_price = final_outcome
                trade.realized_pnl = pnl
                
                self.closed_trades.append(trade)
            
            self.cash += self.position * final_outcome
            print(f"Finalizing position: {self.position} at price {final_outcome}, final cash {self.cash}")
            
            # Clear positions
            self.open_trades = []
            self.position = 0.0
            
            # Final position state
            final_state = PositionState(
                timestamp=final_timestamp,
                position=0.0,
                unrealized_pnl=0.0,
                realized_pnl=self.realized_pnl,
                total_pnl=self.realized_pnl,
                cash=self.cash,
                total_value=self.cash,
                current_price=final_outcome
            )
            
            self.position_history.append(final_state)
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.position_history:
            return
        
        # Extract time series data
        total_values = [state.total_value for state in self.position_history]
        total_pnls = [state.total_pnl for state in self.position_history]
        
        # Trading metrics
        num_trades = len(self.trade_history)
        winning_trades = [t for t in self.closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        # Store metrics
        self.metrics = {
            'total_pnl': total_pnls[-1],
            'realized_pnl': self.realized_pnl,
            'num_trades': num_trades,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'total_invested': sum(abs(t['volume'] * t['price']) for t in self.trade_history),
            'win_rate': win_rate,
        }
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create results DataFrame with time series of positions and PnL."""
        results_data = []
        
        for state in self.position_history:
            results_data.append({
                'timestamp': state.timestamp,
                'position': state.position,
                'unrealized_pnl': state.unrealized_pnl,
                'realized_pnl': state.realized_pnl,
                'total_pnl': state.total_pnl,
                'cash': state.cash,
                'total_value': state.total_value,
                'current_price': state.current_price
            })
        
        return pd.DataFrame(results_data)
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all executed trades."""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_closed_trades_summary(self) -> pd.DataFrame:
        """Get summary of all closed trades with PnL."""
        if not self.closed_trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'entry_timestamp': trade.entry_timestamp,
                'exit_timestamp': trade.exit_timestamp,
                'side': trade.side.value,
                'volume': trade.entry_volume,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'realized_pnl': trade.realized_pnl,
                'return': trade.realized_pnl / (trade.entry_volume * trade.entry_price),
                'duration': (trade.exit_timestamp - trade.entry_timestamp).total_seconds() / 3600  # hours
            })
        
        return pd.DataFrame(trades_data)
    
    def print_performance_summary(self):
        """Print a summary of backtest performance."""
        if not self.metrics:
            print("No backtest results available.")
            return
        
        print("=" * 60)
        print("STRATEGY BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"Initial Cash:        ${self.metrics['initial_cash']:,.2f}")
        print(f"Final Value:         ${self.metrics['final_value']:,.2f}")
        print(f"Total Return:        {self.metrics['total_return']:.2%}")
        print(f"Total PnL:           ${self.metrics['total_pnl']:,.2f}")
        print(f"Realized PnL:        ${self.metrics['realized_pnl']:,.2f}")
        print()
        
        print(f"Volatility (Daily.):   {self.metrics['volatility']:.2%}")
        print(f"Sharpe Ratio:        {self.metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {self.metrics['max_drawdown']:.2%}")
        print()
        
        print(f"Total Trades:        {self.metrics['num_trades']}")
        print(f"Winning Trades:      {self.metrics['num_winning_trades']}")
        print(f"Losing Trades:       {self.metrics['num_losing_trades']}")
        print(f"Win Rate:            {self.metrics['win_rate']:.2%}")
        print(f"Avg Win:             ${self.metrics['avg_win']:,.2f}")
        print(f"Avg Loss:            ${self.metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:       {self.metrics['profit_factor']:.2f}")
        print("=" * 60)

