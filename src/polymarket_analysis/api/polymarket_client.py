"""
Polymarket API client using the official py-clob-client library.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BookParams, TradeParams
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from ..utils.logger import get_default_logger


@dataclass
class Market:
    """Represents a Polymarket market."""
    id: str
    question: str
    description: str
    end_date: datetime
    outcome_prices: Dict[str, float]
    volume: float
    liquidity: float
    active: bool
    tags: List[str]
    condition_id: Optional[str] = None
    tokens: Optional[List[Dict[str, Any]]] = None


@dataclass
class PricePoint:
    """Represents a price point for a market outcome."""
    market_id: str
    outcome: str
    price: float
    timestamp: datetime
    volume: float


class PolymarketClient:
    """
    Wrapper around the official py-clob-client for Polymarket API interactions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://clob.polymarket.com",
        private_key: Optional[str] = None,
        api_creds: Optional[Dict[str, str]] = None,
        chain_id: int = POLYGON
    ):
        """
        Initialize the Polymarket client.
        
        Args:
            api_key: API key for authenticated requests (deprecated, use api_creds)
            base_url: Base URL for the API
            private_key: Private key for wallet operations
            api_creds: Dictionary with api_key, api_secret, api_passphrase
            chain_id: Blockchain chain ID (default: Polygon mainnet)
        """
        self.base_url = base_url.rstrip('/')
        self.logger = get_default_logger()
        
        # Initialize credentials
        creds = None
        if api_creds:
            # Validate that all required credentials are present
            api_key_val = api_creds.get("api_key")
            api_secret_val = api_creds.get("api_secret")
            api_passphrase_val = api_creds.get("api_passphrase")
            
            if api_key_val and api_secret_val and api_passphrase_val:
                creds = ApiCreds(
                    api_key=api_key_val,
                    api_secret=api_secret_val,
                    api_passphrase=api_passphrase_val
                )
            else:
                self.logger.warning("Incomplete API credentials provided. Client will operate in read-only mode.")
        elif api_key:
            # For backward compatibility, create minimal creds
            self.logger.warning("Using deprecated api_key parameter. Consider using api_creds instead.")
        
        # Initialize the official client
        if private_key and creds:
            self.client = ClobClient(
                host=self.base_url,
                key=private_key,
                creds=creds,
                chain_id=chain_id
            )
        elif private_key:
            self.client = ClobClient(
                host=self.base_url,
                key=private_key,
                chain_id=chain_id
            )
        elif creds:
            self.client = ClobClient(
                host=self.base_url,
                creds=creds,
                chain_id=chain_id
            )
        else:
            # Read-only client
            self.client = ClobClient(
                host=self.base_url,
                chain_id=chain_id
            )
    
    async def close(self):
        """Close any resources (for compatibility with async context manager)."""
        pass
    
    def _parse_market_data(self, market_data: Dict[str, Any]) -> Market:
        """Parse raw market data into Market object."""
        try:
            # Extract end date
            end_date_str = market_data.get("end_date_iso", market_data.get("end_date", ""))
            if end_date_str:
                if end_date_str.endswith("Z"):
                    end_date_str = end_date_str[:-1] + "+00:00"
                end_date = datetime.fromisoformat(end_date_str)
            else:
                # Default to far future if no end date
                end_date = datetime.now() + timedelta(days=365)
            
            # Extract outcome prices from tokens
            outcome_prices = {}
            tokens = market_data.get("tokens", [])
            for token in tokens:
                outcome = token.get("outcome", "")
                # Try to get price from various possible fields
                price = None
                if "price" in token:
                    price = float(token["price"])
                elif "last_price" in token:
                    price = float(token["last_price"])
                
                if outcome and price is not None:
                    outcome_prices[outcome] = price
            
            return Market(
                id=market_data.get("id", ""),
                question=market_data.get("question", ""),
                description=market_data.get("description", ""),
                end_date=end_date,
                outcome_prices=outcome_prices,
                volume=float(market_data.get("volume", 0)),
                liquidity=float(market_data.get("liquidity", 0)),
                active=market_data.get("active", True),
                tags=market_data.get("tags", []),
                condition_id=market_data.get("condition_id"),
                tokens=tokens
            )
        except Exception as e:
            self.logger.error(f"Error parsing market data: {e}")
            # Return a minimal market object
            return Market(
                id=market_data.get("id", "unknown"),
                question=market_data.get("question", "Unknown"),
                description="",
                end_date=datetime.now() + timedelta(days=365),
                outcome_prices={},
                volume=0.0,
                liquidity=0.0,
                active=False,
                tags=[]
            )
    
    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[Market]:
        """
        Fetch markets from Polymarket.
        
        Args:
            active: Whether to fetch only active markets
            limit: Maximum number of markets to fetch (handled via pagination)
            offset: Offset for pagination
            tags: Filter by tags (handled via keyword filtering)
        
        Returns:
            List of Market objects
        """
        try:
            self.logger.info("Fetching markets from Polymarket...")
            
            # Get simplified markets (easier to parse)
            response = self.client.get_simplified_markets()
            
            markets = []
            market_count = 0
            
            # Process markets from response
            raw_markets = []
            if isinstance(response, dict):
                raw_markets = response.get("data", [])
            elif isinstance(response, list):
                raw_markets = response
            else:
                self.logger.warning(f"Unexpected response format: {type(response)}")
                return []
            
            for market_data in raw_markets:
                # Apply offset
                if market_count < offset:
                    market_count += 1
                    continue
                
                # Apply limit
                if len(markets) >= limit:
                    break
                
                # Parse market
                market = self._parse_market_data(market_data)
                
                # Apply filters
                if active and not market.active:
                    continue
                
                if tags:
                    # Check if any of the requested tags are in the market tags
                    if not any(tag.lower() in [t.lower() for t in market.tags] for tag in tags):
                        # Also check question and description for tag keywords
                        text_to_search = f"{market.question} {market.description}".lower()
                        if not any(tag.lower() in text_to_search for tag in tags):
                            continue
                
                markets.append(market)
                market_count += 1
            
            self.logger.info(f"Fetched {len(markets)} markets")
            return markets
            
        except Exception as e:
            self.logger.error(f"Failed to fetch markets: {e}")
            # Return empty list instead of raising to maintain compatibility
            return []
    
    async def get_crypto_markets(self) -> List[Market]:
        """
        Fetch crypto-related markets specifically.
        Uses a comprehensive set of crypto-related keywords.
        
        Returns:
            List of crypto Market objects
        """
        crypto_keywords = [
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "BTC", "ETH", 
            "DeFi", "Web3", "blockchain", "altcoin", "dogecoin", "DOGE",
            "solana", "SOL", "cardano", "ADA", "polkadot", "DOT",
            "binance", "BNB", "ripple", "XRP", "litecoin", "LTC",
            "chainlink", "LINK", "uniswap", "UNI", "avalanche", "AVAX",
            "price", "trading", "exchange", "coin", "token", "digital asset",
            "meme coin", "shiba", "pepe", "usdc", "usdt", "stablecoin",
            "market cap", "coinbase", "fantom", "FTM", "cosmos", "ATOM",
            "polygon", "MATIC", "stellar", "XLM", "tron", "TRX"
        ]
        
        # Get all markets and filter for crypto-related ones
        all_markets = await self.get_markets(limit=1000, active=True)
        
        crypto_markets = []
        for market in all_markets:
            # Check question and description for crypto keywords
            text_to_search = f"{market.question} {market.description}".lower()
            
            # More lenient matching - any crypto keyword found
            if any(keyword.lower() in text_to_search for keyword in crypto_keywords):
                crypto_markets.append(market)
            # Also check tags if available
            elif any(keyword.lower() in [tag.lower() for tag in market.tags] for keyword in crypto_keywords):
                crypto_markets.append(market)
        
        self.logger.info(f"Found {len(crypto_markets)} crypto-related markets out of {len(all_markets)} total markets")
        return crypto_markets
    
    async def get_market_history(
        self,
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> List[PricePoint]:
        """
        Fetch historical price data for a market using trade data.
        
        Args:
            market_id: Market condition_id
            start_date: Start date for historical data (as Unix timestamp)
            end_date: End date for historical data (as Unix timestamp)  
            resolution: Time resolution (not used - returns all trades)
        
        Returns:
            List of PricePoint objects from actual trades
            
        Note: Requires Level 2 authentication for trade data
        """
        try:
            self.logger.info(f"Fetching historical trade data for market {market_id}")
            
            # Convert dates to Unix timestamps if provided
            after_ts = None
            before_ts = None
            if start_date:
                after_ts = int(start_date.timestamp())
            if end_date:
                before_ts = int(end_date.timestamp())
            
            # Get trade data for the market
            # Create trade params - all parameters should be provided or defaults used
            if after_ts is not None and before_ts is not None:
                trade_params = TradeParams(
                    market=market_id,
                    after=after_ts,
                    before=before_ts
                )
            elif after_ts is not None:
                trade_params = TradeParams(
                    market=market_id,
                    after=after_ts
                )
            elif before_ts is not None:
                trade_params = TradeParams(
                    market=market_id,
                    before=before_ts
                )
            else:
                trade_params = TradeParams(
                    market=market_id
                )
            
            trades = self.client.get_trades(params=trade_params)
            
            price_points = []
            for trade in trades:
                if isinstance(trade, dict):
                    # Extract trade information
                    price_point = PricePoint(
                        market_id=market_id,
                        outcome=trade.get("outcome", ""),
                        price=float(trade.get("price", 0)),
                        timestamp=datetime.fromtimestamp(trade.get("timestamp", 0) / 1000),  # Convert from ms
                        volume=float(trade.get("size", 0))
                    )
                    price_points.append(price_point)
            
            # Sort by timestamp for chronological order
            price_points.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Fetched {len(price_points)} historical price points for market {market_id}")
            return price_points
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market history for {market_id}: {e}")
            return []

    async def get_market_trade_events(
        self,
        condition_id: str
    ) -> List[PricePoint]:
        """
        Fetch market trade events using the market trades events endpoint.
        
        Args:
            condition_id: Market condition ID
        
        Returns:
            List of PricePoint objects from trade events
        """
        try:
            self.logger.info(f"Fetching trade events for condition {condition_id}")
            
            # Get trade events for the market (no auth required)
            trade_events = self.client.get_market_trades_events(condition_id)
            
            price_points = []
            events_data = trade_events.get("data", []) if isinstance(trade_events, dict) else trade_events
            
            for event in events_data:
                if isinstance(event, dict):
                    price_point = PricePoint(
                        market_id=condition_id,
                        outcome=event.get("outcome", ""),
                        price=float(event.get("price", 0)),
                        timestamp=datetime.fromtimestamp(event.get("timestamp", 0) / 1000),
                        volume=float(event.get("size", 0))
                    )
                    price_points.append(price_point)
            
            # Sort by timestamp
            price_points.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Fetched {len(price_points)} trade events for condition {condition_id}")
            return price_points
            
        except Exception as e:
            self.logger.error(f"Failed to fetch trade events for {condition_id}: {e}")
            return []

    async def get_current_market_prices(
        self,
        market_id: str
    ) -> List[PricePoint]:
        """
        Get current market prices using last trade prices and order book data.
        
        Args:
            market_id: Market condition_id
        
        Returns:
            List of current PricePoint objects
        """
        try:
            self.logger.info(f"Fetching current prices for market {market_id}")
            
            # Get market data to find tokens
            market_data = self.client.get_market(market_id)
            
            price_points = []
            current_time = datetime.now()
            
            # Extract tokens and get their current prices
            if isinstance(market_data, dict):
                tokens = market_data.get("tokens", [])
            else:
                self.logger.warning(f"Unexpected market data format for {market_id}: {type(market_data)}")
                return []
            for token in tokens:
                token_id = token.get("token_id")
                outcome = token.get("outcome", "")
                
                if token_id:
                    try:
                        # Try multiple price sources
                        price = None
                        volume = 0.0
                        
                        # 1. Try last trade price
                        try:
                            last_trade = self.client.get_last_trade_price(token_id)
                            if isinstance(last_trade, dict) and "price" in last_trade:
                                price = float(last_trade["price"])
                        except:
                            pass
                        
                        # 2. Try midpoint price if no trade price
                        if price is None:
                            try:
                                midpoint_data = self.client.get_midpoint(token_id)
                                if isinstance(midpoint_data, dict) and "mid" in midpoint_data:
                                    price = float(midpoint_data["mid"])
                            except:
                                pass
                        
                        # 3. Try order book for price and volume
                        try:
                            order_book = self.client.get_order_book(token_id)
                            if order_book:
                                # Get volume from order book
                                if hasattr(order_book, 'bids') and order_book.bids:
                                    volume += sum(float(bid.size) for bid in order_book.bids)
                                if hasattr(order_book, 'asks') and order_book.asks:
                                    volume += sum(float(ask.size) for ask in order_book.asks)
                                
                                # Use mid price from order book if still no price
                                if price is None and order_book.bids and order_book.asks:
                                    best_bid = max(float(bid.price) for bid in order_book.bids)
                                    best_ask = min(float(ask.price) for ask in order_book.asks)
                                    price = (best_bid + best_ask) / 2
                        except:
                            pass
                        
                        if price is not None and price > 0:
                            price_point = PricePoint(
                                market_id=market_id,
                                outcome=outcome,
                                price=price,
                                timestamp=current_time,
                                volume=volume
                            )
                            price_points.append(price_point)
                            
                    except Exception as e:
                        self.logger.debug(f"Could not get price for token {token_id}: {e}")
            
            self.logger.info(f"Fetched {len(price_points)} current price points for market {market_id}")
            return price_points
            
        except Exception as e:
            self.logger.error(f"Failed to fetch current prices for {market_id}: {e}")
            return []
    
    async def get_multiple_market_histories(
        self,
        market_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> Dict[str, List[PricePoint]]:
        """
        Fetch historical data for multiple markets.
        
        Args:
            market_ids: List of market identifiers
            start_date: Start date for historical data
            end_date: End date for historical data
            resolution: Time resolution
        
        Returns:
            Dictionary mapping market_id to list of PricePoint objects
        """
        history_data = {}
        
        for market_id in market_ids:
            try:
                price_points = await self.get_market_history(
                    market_id, start_date, end_date, resolution
                )
                history_data[market_id] = price_points
            except Exception as e:
                self.logger.error(f"Failed to fetch history for {market_id}: {e}")
                history_data[market_id] = []
        
        return history_data
    
    def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a specific token.
        
        Args:
            token_id: Token ID to get order book for
        
        Returns:
            Order book data as dictionary
        """
        try:
            order_book = self.client.get_order_book(token_id)
            
            # Convert OrderBookSummary to dict format
            if hasattr(order_book, 'bids') and hasattr(order_book, 'asks'):
                return {
                    "bids": [
                        {"price": float(bid.price), "size": float(bid.size)}
                        for bid in order_book.bids
                    ] if order_book.bids else [],
                    "asks": [
                        {"price": float(ask.price), "size": float(ask.size)}
                        for ask in order_book.asks
                    ] if order_book.asks else []
                }
            else:
                # If it's already a dict, return as-is
                return order_book if isinstance(order_book, dict) else {}
        except Exception as e:
            self.logger.error(f"Failed to get order book for {token_id}: {e}")
            return {}
    
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get midpoint price for a token.
        
        Args:
            token_id: Token ID
        
        Returns:
            Midpoint price or None if unavailable
        """
        try:
            result = self.client.get_midpoint(token_id)
            if isinstance(result, dict):
                return float(result.get("mid", 0)) if "mid" in result else None
            elif isinstance(result, (int, float)):
                return float(result)
            else:
                return None
        except Exception as e:
            self.logger.debug(f"Could not get midpoint for {token_id}: {e}")
            return None
    
    async def get_order_book_snapshot(
        self,
        market_id: str
    ) -> Dict[str, Any]:
        """
        Get a comprehensive order book snapshot for a market with liquidity analysis.
        
        Args:
            market_id: Market condition_id
        
        Returns:
            Dictionary with order book data, spreads, and liquidity metrics
        """
        try:
            self.logger.info(f"Fetching order book snapshot for market {market_id}")
            
            # Get market data to find tokens
            market_data = self.client.get_market(market_id)
            
            # Handle different response formats
            if isinstance(market_data, dict):
                tokens = market_data.get("tokens", [])
            else:
                self.logger.warning(f"Unexpected market data format for {market_id}: {type(market_data)}")
                return {}
            
            snapshot = {
                "market_id": market_id,
                "timestamp": datetime.now(),
                "tokens": {}
            }
            
            for token in tokens:
                token_id = token.get("token_id")
                outcome = token.get("outcome", "")
                
                if token_id:
                    try:
                        # Get order book
                        order_book = self.client.get_order_book(token_id)
                        
                        # Calculate metrics
                        token_data = {
                            "outcome": outcome,
                            "token_id": token_id,
                            "bids": [],
                            "asks": [],
                            "best_bid": None,
                            "best_ask": None,
                            "spread": None,
                            "mid_price": None,
                            "total_bid_volume": 0.0,
                            "total_ask_volume": 0.0,
                            "liquidity_score": 0.0
                        }
                        
                        # Process bids and asks
                        if hasattr(order_book, 'bids') and order_book.bids:
                            token_data["bids"] = [
                                {"price": float(bid.price), "size": float(bid.size)}
                                for bid in order_book.bids
                            ]
                            token_data["best_bid"] = max(float(bid.price) for bid in order_book.bids)
                            token_data["total_bid_volume"] = sum(float(bid.size) for bid in order_book.bids)
                        
                        if hasattr(order_book, 'asks') and order_book.asks:
                            token_data["asks"] = [
                                {"price": float(ask.price), "size": float(ask.size)}
                                for ask in order_book.asks
                            ]
                            token_data["best_ask"] = min(float(ask.price) for ask in order_book.asks)
                            token_data["total_ask_volume"] = sum(float(ask.size) for ask in order_book.asks)
                        
                        # Calculate derived metrics
                        if token_data["best_bid"] and token_data["best_ask"]:
                            token_data["spread"] = token_data["best_ask"] - token_data["best_bid"]
                            token_data["mid_price"] = (token_data["best_bid"] + token_data["best_ask"]) / 2
                            
                            # Simple liquidity score: total volume / spread
                            total_volume = token_data["total_bid_volume"] + token_data["total_ask_volume"]
                            if token_data["spread"] > 0:
                                token_data["liquidity_score"] = total_volume / token_data["spread"]
                        
                        snapshot["tokens"][outcome] = token_data
                        
                    except Exception as e:
                        self.logger.debug(f"Could not get order book for token {token_id}: {e}")
            
            self.logger.info(f"Generated order book snapshot for market {market_id} with {len(snapshot['tokens'])} tokens")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to get order book snapshot for {market_id}: {e}")
            return {}

    async def get_comprehensive_market_data(
        self,
        market_id: str,
        include_history: bool = True,
        history_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get comprehensive market data including current prices, order book, and optional history.
        
        Args:
            market_id: Market condition_id
            include_history: Whether to fetch historical trade data
            history_days: Number of days of history to fetch
        
        Returns:
            Dictionary with comprehensive market data
        """
        try:
            self.logger.info(f"Fetching comprehensive data for market {market_id}")
            
            # Get basic market info
            market_data = self.client.get_market(market_id)
            
            # Get current prices
            current_prices = await self.get_current_market_prices(market_id)
            
            # Get order book snapshot
            order_book_snapshot = await self.get_order_book_snapshot(market_id)
            
            comprehensive_data = {
                "market_id": market_id,
                "timestamp": datetime.now(),
                "market_info": market_data,
                "current_prices": current_prices,
                "order_book": order_book_snapshot,
                "history": []
            }
            
            # Optionally fetch historical data
            if include_history:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=history_days)
                
                # Try trade events first (no auth required)
                try:
                    history = await self.get_market_trade_events(market_id)
                    # Filter by date range
                    history = [
                        point for point in history
                        if start_date <= point.timestamp <= end_date
                    ]
                    comprehensive_data["history"] = history
                    comprehensive_data["history_source"] = "trade_events"
                except Exception as e:
                    self.logger.debug(f"Could not fetch trade events: {e}")
                    
                    # Fallback to authenticated trades if available
                    try:
                        history = await self.get_market_history(market_id, start_date, end_date)
                        comprehensive_data["history"] = history
                        comprehensive_data["history_source"] = "user_trades"
                    except Exception as e2:
                        self.logger.debug(f"Could not fetch trade history: {e2}")
                        comprehensive_data["history_source"] = "none"
            
            self.logger.info(f"Generated comprehensive data for market {market_id}")
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive data for {market_id}: {e}")
            return {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()



    def make_order(self) :
        client = self.client
        client.set_api_creds(client.create_or_derive_api_creds()) 
        order_args = OrderArgs(
            price=0.01,
            size=5.0,
            side=BUY,
            token_id="", #Token ID you want to purchase goes here. 
        )
        signed_order = client.create_order(order_args)

        resp = client.post_order(signed_order)
        


