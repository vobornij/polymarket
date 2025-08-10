"""
Strategy interface for Polymarket trading based on price predictions and probabilities.

This module provides abstract base classes and concrete implementations for trading strategies
that analyze market data and generate trade orders based on estimated probabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import math

## ?
class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"



@dataclass
class Order:
    """Represents a trade order to be executed."""
    side: OrderSide
    volume: float 
    price: float
    timestamp: datetime



class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    A strategy takes market data including prices and estimated probabilities,
    and generates a series of trade orders based on its logic.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self.orders_generated = []
        self.performance_metrics = {}
    
    @abstractmethod
    def generate_orders(self, market_data: pd.DataFrame) -> List[Order]:
        """
        Generate trade orders based on market data.
        
        Args:
            market_data: DataFrame with columns including:
                - market_id: Market identifier
                - token_id: Token identifier  
                - current_price: Current market price (0-1)
                - fair_price: Model's estimated probability (0-1)
                - timestamp: When the data was collected
                - expiry: When the market expires
                - Additional columns as needed by strategy
        
        Returns:
            List of TradeOrder objects
        """
        pass
    


class EdgeBasedStrategy(TradingStrategy):
    """
    Strategy that trades based on edge between estimated and market probabilities.
    
    Buys when estimated probability > market probability + threshold
    Sells when estimated probability < market probability - threshold
    """

    def __init__(self, name: str = "EdgeBased", **kwargs):
        """
        Initialize edge-based strategy.
        
        Parameters:
            min_edge: Minimum edge required to trade (default: 0.05)
            max_backoff_edge: Maximum edge required to back off (default: 0.01)
            max_position_size: Maximum position size (default: 100.0)
            max_order_size: Maximum order size (default: 10.0)
            use_kelly: Whether to use Kelly criterion for sizing (default: False)
            kelly_fraction: Fraction of Kelly to use (default: 0.25)
            min_confidence: Minimum confidence required (default: 0.0)
            use_limit_orders: Whether to use limit orders (default: True)
            limit_price_offset: Offset from current price for limits (default: 0.01)
        """
        super().__init__(name, **kwargs)
        
        # Strategy parameters with defaults
        self.min_edge = kwargs.get('min_edge', 0.05)
        self.max_backoff_edge = kwargs.get('min_edge', 0.01)
        self.max_position_size = kwargs.get('max_position_size', 100.0)
        self.max_order_size = kwargs.get('max_order_size', 10.0)
        self.use_kelly = kwargs.get('use_kelly', False)
        self.kelly_fraction = kwargs.get('kelly_fraction', 0.25)
        self.min_confidence = kwargs.get('min_confidence', 0.0)
        self.current_position = 0.0
    
    def generate_orders(self, market_data: pd.DataFrame) -> List[Order]:
        """Generate orders based on edge analysis."""
        orders = []
        
        for idx, row in market_data.iterrows():
            if self.should_trade(row):
                order = self._create_order(row)
                if order:
                    orders.append(order)
            elif self.should_exit(row):
                order = self._create_backoff_order(row)
                if order:
                    orders.append(order)
        
        return orders
    
    def _create_backoff_order(self, row: pd.Series) -> Optional[Order]:
        """Create a trade order for the given market data."""
        volume = self.current_position

        side = OrderSide.SELL

        cropped_volume = min(abs(volume), self.max_order_size)

        if cropped_volume <= 0:        
            return None
        
        self.current_position -= cropped_volume
        order = Order(
            side=side,
            volume=cropped_volume,
            price=row['price'],
            timestamp=row['timestamp']
        )
    
        # print(f"Generated order: {order.side.value} {cropped_volume} at {row['price']} fair price {row['fair_price']} on {row['timestamp']}")
        return order

    
    def should_exit(self, row: pd.Series) -> bool:
        """Check if we should back off based on edge."""
        # Calculate edge
        market_prob = row['price']
        fair_price = row['fair_price']
        
        edge = fair_price - market_prob
        
        if edge < self.max_backoff_edge:
            return True
        
        return False
    
    def should_trade(self, row: pd.Series) -> bool:
        """Check if we should trade based on edge."""
        # Calculate edge
        market_price = row['price']
        fair_price = row['fair_price']
        
        edge = fair_price - market_price
        
        if edge < self.min_edge:
            return False
        
        return True
    
    def _create_order(self, row: pd.Series) -> Optional[Order]:
        """Create a trade order for the given market data."""
        market_price = row['price']
        fair_price = row['fair_price']

        if(math.isnan(fair_price)):
            print(f"Skipping order generation for at {row['timestamp']} due to NaN fair price")
            return None

        edge = fair_price - market_price
        
        # Calculate position size
        desired_position = self.calculate_desire_position(row, edge)

        #print(f"Desired position: {desired_position} for edge {edge} at {row['timestamp']}")

        volume = desired_position - self.current_position

        side = OrderSide.BUY if(volume > 0) else OrderSide.SELL

        cropped_volume = min(abs(volume), self.max_order_size)

        self.current_position += cropped_volume if side == OrderSide.BUY else -cropped_volume

        if cropped_volume < 1:        
            return None
        order = Order(
            side=side,
            volume=cropped_volume,
            price=row['price'],
            timestamp=row['timestamp']
        )

        # print(f"Generated order: {order.side.value} {cropped_volume} at {row['price']} fair price {row['fair_price']} on {row['timestamp']}")

        return order
    
    def calculate_desire_position(self, row: pd.Series, edge: float) -> float:
        """Calculate position size using Kelly criterion or fixed sizing."""
        if self.use_kelly:
            return self._kelly_desired_position(row, edge)
        else:
            return min(self.parameters.get('default_size', 10.0), self.max_position_size)
    
    def _kelly_desired_position(self, row: pd.Series, edge: float) -> float:
        """Calculate Kelly criterion position size."""
        market_prob = row['price']
        fair_price = row['fair_price']
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = lose probability
        if edge > 0:  # Buying
            b = (1 - market_prob) / market_prob  # Odds received
            p = fair_price  # Win probability
            q = 1 - p  # Lose probability
        else:  # Selling
            b = market_prob / (1 - market_prob)  # Odds received
            p = 1 - fair_price  # Win probability (market resolves No)
            q = 1 - p  # Lose probability
        
        if b <= 0:
            return 0.0
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fraction and cap
        position_size = kelly_fraction * self.kelly_fraction * 100  # Assuming 100 unit bankroll
        return min(max(position_size, 0), self.max_position_size)
    