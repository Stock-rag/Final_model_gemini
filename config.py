"""
Shared configuration for the unified RAG system
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration management"""

    # API Keys - Use environment variables
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present"""
        if not cls.FINNHUB_API_KEY:
            raise ValueError("FINNHUB_API_KEY environment variable is required. Please set it in .env file or environment.")

    # Model configurations - Using Llama-3.2-3B (2024, public, faster)
    DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Quantization settings
    QUANTIZATION_CONFIG = {
        "load_in_4bit": True,
        "llm_int8_enable_fp32_cpu_offload": False
    }

    # Generation parameters
    GENERATION_PARAMS = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "do_sample": True
    }

    # Market analysis settings
    MARKET_CONFIG = {
        "default_prediction_days": 30,
        "ml_training_days": 730,
        "cache_duration_hours": 24,
        "default_investment": 10000
    }

    # Technical analysis parameters
    TECHNICAL_INDICATORS = {
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "bollinger_window": 20,
        "sma_windows": [5, 10, 20, 50],
        "stoch_window": 14
    }

    # FAISS search settings
    SEARCH_CONFIG = {
        "top_k": 3,
        "news_limit": 100
    }

    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "finnhub_api_key": cls.FINNHUB_API_KEY,
            "llm_model": cls.DEFAULT_LLM_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "quantization": cls.QUANTIZATION_CONFIG,
            "generation": cls.GENERATION_PARAMS,
            "market": cls.MARKET_CONFIG,
            "technical": cls.TECHNICAL_INDICATORS,
            "search": cls.SEARCH_CONFIG
        }