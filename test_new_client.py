#!/usr/bin/env python3
"""
Test script for the updated Polymarket client using py-clob-client.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_client():
    """Test the new client implementation."""
    print("Testing new Polymarket client...")
    
    try:
        from polymarket_analysis.api.polymarket_client import PolymarketClient
        
        # Initialize client (no credentials needed for public endpoints)
        client = PolymarketClient()
        
        async with client:
            # Test getting markets
            print("Fetching markets...")
            markets = await client.get_markets(limit=100)  # Increased limit to find crypto markets
            print(f"Retrieved {len(markets)} markets")
            
            for i, market in enumerate(markets[:3]):
                print(f"Market {i+1}: {market.question[:50]}...")
                print(f"  Volume: {market.volume}")
                print(f"  Active: {market.active}")
                print(f"  Outcomes: {list(market.outcome_prices.keys())}")
            
            # Test getting crypto markets
            print("\nFetching crypto markets...")
            crypto_markets = await client.get_crypto_markets()
            print(f"Retrieved {len(crypto_markets)} crypto markets")
            
            for i, market in enumerate(crypto_markets[:2]):
                print(f"Crypto Market {i+1}: {market.question[:50]}...")
        
        print("Client test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_client())
    sys.exit(0 if success else 1)
