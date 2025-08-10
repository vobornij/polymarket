"""
Configuration management for Polymarket analysis project.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration class for managing environment variables and settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration."""
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load environment variables
        if config_file:
            load_dotenv(config_file)
        else:
            load_dotenv(self.project_root / ".env")
    
    @property
    def polymarket_api_key(self) -> Optional[str]:
        """Get Polymarket API key from environment."""
        return os.getenv("POLYMARKET_API_KEY")
    
    @property
    def polymarket_api_secret(self) -> Optional[str]:
        """Get Polymarket API secret from environment."""
        return os.getenv("POLYMARKET_API_SECRET")
    
    @property
    def polymarket_api_passphrase(self) -> Optional[str]:
        """Get Polymarket API passphrase from environment."""
        return os.getenv("POLYMARKET_API_PASSPHRASE")
    
    @property
    def polymarket_private_key(self) -> Optional[str]:
        """Get Polymarket private key from environment."""
        return os.getenv("POLYMARKET_PRIVATE_KEY")
    
    @property
    def polymarket_api_creds(self) -> Optional[Dict[str, str]]:
        """Get Polymarket API credentials as a dictionary."""
        api_key = self.polymarket_api_key
        api_secret = self.polymarket_api_secret
        api_passphrase = self.polymarket_api_passphrase
        
        if api_key and api_secret and api_passphrase:
            return {
                "api_key": api_key,
                "api_secret": api_secret,
                "api_passphrase": api_passphrase
            }
        return None
    
    @property
    def database_url(self) -> str:
        """Get database URL."""
        return os.getenv("DATABASE_URL", f"sqlite:///{self.project_root}/data/polymarket.db")
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory path."""
        return self.data_dir / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        return self.data_dir / "processed"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.data_dir / "models"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.project_root / "logs"
    
    @property
    def api_settings(self) -> Dict[str, Any]:
        """Get API settings."""
        return {
            "base_url": os.getenv("POLYMARKET_BASE_URL", "https://clob.polymarket.com"),
            "rate_limit": int(os.getenv("API_RATE_LIMIT", "10")),  # requests per second
            "timeout": int(os.getenv("API_TIMEOUT", "30"))  # seconds
        }
    
    @property
    def analysis_settings(self) -> Dict[str, Any]:
        """Get analysis settings."""
        return {
            "lookback_days": int(os.getenv("LOOKBACK_DAYS", "30")),
            "min_volume_threshold": float(os.getenv("MIN_VOLUME_THRESHOLD", "1000")),
            "arbitrage_threshold": float(os.getenv("ARBITRAGE_THRESHOLD", "0.02")),  # 2%
            "confidence_level": float(os.getenv("CONFIDENCE_LEVEL", "0.95"))
        }
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
