# RAG-Based Market Analysis System

A RAG (Retrieval-Augmented Generation) system for market analysis combining LLMs, technical indicators, and machine learning predictions.

## Prerequisites

- Python 3.8+
- pip package manager
- Finnhub API key (for market data)
- ~8GB RAM minimum for running LLM models

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAGZ-C
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
FINNHUB_API_KEY=your_api_key_here
```

Get your free Finnhub API key from: https://finnhub.io/

### 4. Verify Installation
Run the component test to verify everything is installed correctly:
```bash
python test_all_components.py
```

## Running the System

### Main Integration Test
Test the complete RAG pipeline with market analysis:
```bash
python test_integration.py
```

### Market Analysis Components

**1. Advanced Market Analysis Engine:**
```bash
python RAG-Market_analysis_test/ML_e.py
```

**2. Market Analysis Integration:**
```bash
python RAG-Market_analysis_test/Integration.py
```

**3. Gemini Integration (if using Google's Gemini):**
```bash
python RAG-Market_analysis_test/gemini_integration.py
```

### Evaluation & Backtesting

**ML Model Evaluation:**
```bash
python ml_evaluation.py
```

**Historical Backtesting:**
```bash
python historical_backtesting.py
```

**Simple Backtest:**
```bash
python simple_backtest.py
```

**RAG Historical Evaluation:**
```bash
python rag_historical_evaluation.py
```

### LLM Implementations

**Test Different LLM Backends:**
```bash
python llm_implementations.py
```

**Test Gemini Models:**
```bash
python test_gemini.py
# or
python test_gemini_simple.py
```

## Project Structure

```
RAGZ-C/
├── config.py                      # Centralized configuration
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
│
├── RAG-Market_analysis_test/     # Market analysis modules
│   ├── ML_e.py                   # Advanced market analysis engine
│   ├── Integration.py            # RAG integration
│   ├── data_sources.py           # Data source utilities
│   └── gemini_integration.py     # Google Gemini integration
│
├── utils/                        # Utility modules
│   ├── llm_factory.py           # LLM factory pattern
│   └── embedding_utils.py       # Embedding utilities
│
├── market/                       # Market analysis
│   └── analysis_engine.py       # Market analysis engine
│
├── RAG-zul-general/             # General RAG implementations
│   ├── langchain_llm2.py        # LangChain integration
│   └── api-caller.py            # API utilities
│
└── test_*.py                    # Various test scripts
```

## Configuration

Edit `config.py` to customize:
- LLM models (default: Llama-3.2-3B-Instruct)
- Embedding models (default: all-MiniLM-L6-v2)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Market analysis parameters
- Quantization settings for model optimization

## Key Features

- **RAG-based market analysis**: Combines retrieval with LLM generation
- **Multiple LLM backends**: Support for HuggingFace, Gemini, and more
- **Technical indicators**: RSI, MACD, Bollinger Bands, SMA, Stochastic
- **ML predictions**: Random Forest-based price predictions
- **Historical backtesting**: Evaluate strategies on historical data
- **FAISS vector search**: Fast similarity search for news/context
- **Quantization support**: Run large models efficiently with 4-bit quantization

## Usage Example

```python
from config import Config
from RAG-Market_analysis_test.ML_e import AdvancedMarketAnalysisEngine

# Initialize engine
Config.validate_api_keys()
engine = AdvancedMarketAnalysisEngine(Config.FINNHUB_API_KEY)

# Analyze a stock
analysis = engine.comprehensive_analysis_with_prediction(
    symbol="AAPL",
    days_to_predict=30
)
print(analysis)
```

## Troubleshooting

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**API Key Issues:**
- Verify `.env` file exists and contains `FINNHUB_API_KEY`
- Check API key is valid at https://finnhub.io/

**Memory Issues:**
- Reduce model size in `config.py`
- Enable 4-bit quantization (already default)
- Close other applications to free RAM

**Model Loading Errors:**
- First run may take time to download models
- Check internet connection
- Ensure sufficient disk space (~5GB for models)

## Testing

Run the comprehensive test suite:
```bash
python test_all_components.py
```

This tests:
- All library imports
- Market analysis engine
- LangChain integration
- File syntax validation
- Basic LLM functionality

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
