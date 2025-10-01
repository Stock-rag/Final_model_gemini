"""
Multi-source market data provider with fallback support
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MultiMarketDataProvider:
    """Unified market data provider with multiple sources and fallback"""

    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_api_key = finnhub_api_key
        self.finnhub_base_url = "https://finnhub.io/api/v1"
        self.headers = {"X-Finnhub-Token": finnhub_api_key} if finnhub_api_key else None

    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price with fallback sources"""

        # Try Yahoo Finance first (always works, no API key needed)
        try:
            print(f"[DATA] Fetching current price for {symbol} from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                change = hist['Close'].iloc[-1] - hist['Open'].iloc[-1] if len(hist) > 0 else 0
                change_percent = (change / hist['Open'].iloc[-1] * 100) if hist['Open'].iloc[-1] > 0 else 0

                return {
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "change": float(change),
                    "change_percent": float(change_percent),
                    "open": float(hist['Open'].iloc[-1]),
                    "high": float(hist['High'].iloc[-1]),
                    "low": float(hist['Low'].iloc[-1]),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"[DATA] Yahoo Finance failed: {e}")

        # Try Finnhub as backup (if API key available)
        if self.finnhub_api_key and self.headers:
            try:
                print(f"[DATA] Trying Finnhub for {symbol}...")
                url = f"{self.finnhub_base_url}/quote"
                params = {"symbol": symbol}
                response = requests.get(url, headers=self.headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if 'c' in data and data['c'] > 0:  # 'c' is current price
                        return {
                            "symbol": symbol,
                            "current_price": data['c'],
                            "change": data['d'],
                            "change_percent": data['dp'],
                            "open": data['o'],
                            "high": data['h'],
                            "low": data['l'],
                            "volume": data.get('v', 0),
                            "source": "Finnhub",
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"[DATA] Finnhub failed: {e}")

        # Return error if all sources fail
        return {"error": f"Could not fetch data for {symbol} from any source"}

    def get_historical_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get historical data with fallback sources"""

        # Try Yahoo Finance first
        try:
            print(f"[DATA] Fetching {days} days of history for {symbol} from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")

            if not hist.empty:
                # Calculate all technical indicators
                close_prices = hist['Close']

                # RSI
                hist['RSI'] = self.calculate_rsi(close_prices)

                # MACD
                macd_data = self.calculate_macd(close_prices)
                hist['MACD'] = macd_data['macd_line']
                hist['MACD_Signal'] = macd_data['signal_line']
                hist['MACD_Histogram'] = macd_data['histogram']

                # Bollinger Bands
                bb_data = self.calculate_bollinger_bands(close_prices)
                hist['BB_Upper'] = bb_data['upper_band']
                hist['BB_Middle'] = bb_data['middle_band']
                hist['BB_Lower'] = bb_data['lower_band']

                # Moving Averages
                ma_data = self.calculate_moving_averages(close_prices)
                hist['SMA_5'] = ma_data['sma_5']
                hist['SMA_10'] = ma_data['sma_10']
                hist['SMA_20'] = ma_data['sma_20']
                hist['SMA_50'] = ma_data['sma_50']
                hist['EMA_12'] = ma_data['ema_12']
                hist['EMA_26'] = ma_data['ema_26']

                # Get latest values for summary
                latest_values = {
                    'rsi': hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else None,
                    'macd': hist['MACD'].iloc[-1] if not pd.isna(hist['MACD'].iloc[-1]) else None,
                    'macd_signal': hist['MACD_Signal'].iloc[-1] if not pd.isna(hist['MACD_Signal'].iloc[-1]) else None,
                    'sma_20': hist['SMA_20'].iloc[-1] if not pd.isna(hist['SMA_20'].iloc[-1]) else None,
                    'sma_50': hist['SMA_50'].iloc[-1] if not pd.isna(hist['SMA_50'].iloc[-1]) else None,
                    'bb_upper': hist['BB_Upper'].iloc[-1] if not pd.isna(hist['BB_Upper'].iloc[-1]) else None,
                    'bb_lower': hist['BB_Lower'].iloc[-1] if not pd.isna(hist['BB_Lower'].iloc[-1]) else None
                }

                return {
                    "symbol": symbol,
                    "historical_data": hist.to_dict('records'),
                    "technical_indicators": latest_values,
                    "source": "Yahoo Finance",
                    "period_days": days,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"[DATA] Yahoo Finance historical failed: {e}")

        # Try Finnhub historical (if available)
        if self.finnhub_api_key and self.headers:
            try:
                print(f"[DATA] Trying Finnhub historical for {symbol}...")
                from_date = int((datetime.now() - timedelta(days=days)).timestamp())
                to_date = int(datetime.now().timestamp())

                url = f"{self.finnhub_base_url}/stock/candle"
                params = {
                    "symbol": symbol,
                    "resolution": "D",
                    "from": from_date,
                    "to": to_date
                }
                response = requests.get(url, headers=self.headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('s') == 'ok' and 'c' in data:
                        # Convert to DataFrame format
                        df = pd.DataFrame({
                            'timestamp': [datetime.fromtimestamp(t) for t in data['t']],
                            'Open': data['o'],
                            'High': data['h'],
                            'Low': data['l'],
                            'Close': data['c'],
                            'Volume': data['v']
                        })
                        return {
                            "symbol": symbol,
                            "historical_data": df.to_dict('records'),
                            "source": "Finnhub",
                            "period_days": days,
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"[DATA] Finnhub historical failed: {e}")

        return {"error": f"Could not fetch historical data for {symbol}"}

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD technical indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band
        }

    def calculate_moving_averages(self, prices: pd.Series) -> Dict:
        """Calculate various moving averages"""
        return {
            'sma_5': prices.rolling(window=5).mean(),
            'sma_10': prices.rolling(window=10).mean(),
            'sma_20': prices.rolling(window=20).mean(),
            'sma_50': prices.rolling(window=50).mean(),
            'ema_12': prices.ewm(span=12).mean(),
            'ema_26': prices.ewm(span=26).mean()
        }

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information"""
        try:
            print(f"[DATA] Fetching company info for {symbol}...")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "market_cap": info.get('marketCap', 0),
                "employees": info.get('fullTimeEmployees', 0),
                "description": info.get('longBusinessSummary', 'N/A')[:500] + "...",
                "source": "Yahoo Finance",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Could not fetch company info for {symbol}: {e}"}

    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis combining all data sources"""
        print(f"[DATA] Starting comprehensive analysis for {symbol}...")

        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data_sources": []
        }

        # Get current price
        price_data = self.get_current_price(symbol)
        if "error" not in price_data:
            analysis["current_data"] = price_data
            analysis["data_sources"].append(price_data["source"])
        else:
            analysis["current_data"] = {"error": price_data["error"]}

        # Get historical data
        historical = self.get_historical_data(symbol, days=30)
        if "error" not in historical:
            analysis["historical_data"] = historical
            if historical["source"] not in analysis["data_sources"]:
                analysis["data_sources"].append(historical["source"])
        else:
            analysis["historical_data"] = {"error": historical["error"]}

        # Get company info
        company = self.get_company_info(symbol)
        if "error" not in company:
            analysis["company_info"] = company
            if company["source"] not in analysis["data_sources"]:
                analysis["data_sources"].append(company["source"])
        else:
            analysis["company_info"] = {"error": company["error"]}

        return analysis

# Test function
def test_data_provider():
    """Test the multi-source data provider"""
    provider = MultiMarketDataProvider()  # No API key needed for Yahoo Finance

    # Test with AAPL
    print("Testing with AAPL...")
    analysis = provider.get_comprehensive_analysis("AAPL")

    print(f"Data sources used: {analysis.get('data_sources', [])}")
    if "current_data" in analysis and "error" not in analysis["current_data"]:
        current = analysis["current_data"]
        print(f"Current price: ${current['current_price']:.2f}")
        print(f"Change: {current['change_percent']:.2f}%")

    return analysis

if __name__ == "__main__":
    test_data_provider()