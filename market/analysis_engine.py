"""
Refactored market analysis engine with unified configuration
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta

from config import Config

class MarketAnalysisEngine:
    """Unified market analysis engine"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.headers = {"X-Finnhub-Token": self.api_key}
        self.models = {}
        self.scalers = {}
        self.cache = {}

    def get_historical_data(self, symbol: str, days: int = None) -> Dict:
        """Get historical stock data"""
        days = days or Config.MARKET_CONFIG["ml_training_days"]

        from_date = int((datetime.now() - timedelta(days=days)).timestamp())
        to_date = int(datetime.now().timestamp())

        url = f"{self.base_url}/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": from_date,
            "to": to_date
        }

        try:
            response = requests.get(url, params=params, headers=self.headers)
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators using config"""
        config = Config.TECHNICAL_INDICATORS

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=config["rsi_window"])

        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=config["macd_slow"], window_fast=config["macd_fast"])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=config["bollinger_window"])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=config["stoch_window"])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

        # CCI
        df['cci'] = ta.momentum.cci(df['high'], df['low'], df['close'])

        # Moving averages
        for window in config["sma_windows"]:
            df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)

        # Exponential Moving Averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

        # Price momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()

        # Support and resistance
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()

        return df

    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional ML features"""
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()

        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Trend indicators
        df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)

        # Volatility features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']

        return df

    def prepare_dataset(self, historical_data: Dict) -> pd.DataFrame:
        """Convert raw data to ML-ready dataframe"""
        if 'c' not in historical_data or not historical_data['c']:
            return pd.DataFrame()

        df = pd.DataFrame({
            'timestamp': historical_data['t'],
            'close': historical_data['c'],
            'high': historical_data['h'],
            'low': historical_data['l'],
            'open': historical_data['o'],
            'volume': historical_data['v']
        })

        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        df = self.calculate_technical_indicators(df)
        df = self.create_ml_features(df)
        df.dropna(inplace=True)

        return df

    def train_models(self, df: pd.DataFrame, target_days: int = 1) -> Dict:
        """Train ML models for prediction"""
        if len(df) < 100:
            return {"error": "Insufficient data for training"}

        feature_cols = [col for col in df.columns if col not in ['close', 'timestamp', 'high', 'low', 'open', 'volume']]
        X = df[feature_cols].values
        y = df['close'].shift(-target_days).dropna().values
        X = X[:-target_days if target_days > 0 else len(X)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {}
        results = {}

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        models['linear'] = lr
        results['linear'] = {
            'mae': mean_absolute_error(y_test, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
        }

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        models['random_forest'] = rf
        results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
        }

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        models['gradient_boost'] = gb
        results['gradient_boost'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred))
        }

        self.models = models
        self.scalers['features'] = scaler

        return {
            'models_trained': list(models.keys()),
            'performance': results,
            'feature_columns': feature_cols,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def predict_prices(self, df: pd.DataFrame, days: int = None) -> Dict:
        """Predict future prices"""
        days = days or Config.MARKET_CONFIG["default_prediction_days"]

        if not self.models:
            return {"error": "Models not trained"}

        latest_data = df.iloc[-1:]
        feature_cols = [col for col in df.columns if col not in ['close', 'timestamp', 'high', 'low', 'open', 'volume']]
        X_latest = latest_data[feature_cols].values

        predictions = {}

        for model_name, model in self.models.items():
            model_predictions = []
            current_features = X_latest.copy()

            for day in range(days):
                if model_name in ['linear', 'gradient_boost']:
                    scaled_features = self.scalers['features'].transform(current_features)
                    pred = model.predict(scaled_features)[0]
                else:
                    pred = model.predict(current_features)[0]

                model_predictions.append(pred)
                current_features = current_features.copy()

            predictions[model_name] = model_predictions

        # Ensemble prediction
        ensemble_pred = []
        for i in range(days):
            day_predictions = [predictions[model][i] for model in predictions.keys()]
            ensemble_pred.append(np.mean(day_predictions))

        predictions['ensemble'] = ensemble_pred

        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]

        return {
            'predictions': predictions,
            'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'current_price': df['close'].iloc[-1],
            'models_used': list(self.models.keys())
        }

    def calculate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals"""
        latest = df.iloc[-1]
        signals = {}

        # RSI signals
        if latest['rsi'] < 30:
            signals['rsi'] = 'BUY'
        elif latest['rsi'] > 70:
            signals['rsi'] = 'SELL'
        else:
            signals['rsi'] = 'HOLD'

        # MACD signals
        if latest['macd'] > latest['macd_signal']:
            signals['macd'] = 'BUY'
        else:
            signals['macd'] = 'SELL'

        # Bollinger Bands
        if latest['close'] < latest['bb_lower']:
            signals['bollinger'] = 'BUY'
        elif latest['close'] > latest['bb_upper']:
            signals['bollinger'] = 'SELL'
        else:
            signals['bollinger'] = 'HOLD'

        # Moving averages
        if latest['close'] > latest['sma_20'] and latest['sma_20'] > latest['sma_50']:
            signals['moving_avg'] = 'BUY'
        elif latest['close'] < latest['sma_20'] and latest['sma_20'] < latest['sma_50']:
            signals['moving_avg'] = 'SELL'
        else:
            signals['moving_avg'] = 'HOLD'

        # Stochastic
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals['stochastic'] = 'BUY'
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals['stochastic'] = 'SELL'
        else:
            signals['stochastic'] = 'HOLD'

        # Overall signal
        buy_votes = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_votes = sum(1 for signal in signals.values() if signal == 'SELL')

        if buy_votes > sell_votes:
            signals['overall'] = 'BUY'
        elif sell_votes > buy_votes:
            signals['overall'] = 'SELL'
        else:
            signals['overall'] = 'HOLD'

        return signals

    def comprehensive_analysis(self, symbol: str, prediction_days: int = None) -> Dict:
        """Perform complete analysis"""
        prediction_days = prediction_days or Config.MARKET_CONFIG["default_prediction_days"]

        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis": {}
        }

        # Get historical data
        historical = self.get_historical_data(symbol, days=Config.MARKET_CONFIG["ml_training_days"])

        if 'error' in historical:
            return {"error": historical['error']}

        # Prepare dataset
        df = self.prepare_dataset(historical)

        if df.empty:
            return {"error": "Failed to prepare dataset"}

        # Technical indicators
        latest_indicators = {
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'macd_signal': df['macd_signal'].iloc[-1],
            'bollinger_upper': df['bb_upper'].iloc[-1],
            'bollinger_lower': df['bb_lower'].iloc[-1],
            'sma_20': df['sma_20'].iloc[-1],
            'sma_50': df['sma_50'].iloc[-1],
            'atr': df['atr'].iloc[-1],
            'stochastic_k': df['stoch_k'].iloc[-1],
            'williams_r': df['williams_r'].iloc[-1]
        }
        analysis["analysis"]["technical_indicators"] = latest_indicators

        # Trading signals
        signals = self.calculate_trading_signals(df)
        analysis["analysis"]["trading_signals"] = signals

        # Train models
        training_results = self.train_models(df, target_days=1)
        analysis["analysis"]["model_training"] = training_results

        # Predictions
        if 'error' not in training_results:
            predictions = self.predict_prices(df, days=prediction_days)
            analysis["analysis"]["price_predictions"] = predictions

            # Portfolio projections
            current_price = df['close'].iloc[-1]
            initial_investment = Config.MARKET_CONFIG["default_investment"]
            shares = initial_investment / current_price

            portfolio_projections = {}
            for model_name, pred_prices in predictions['predictions'].items():
                final_price = pred_prices[-1]
                final_value = shares * final_price
                profit_loss = final_value - initial_investment
                profit_loss_pct = (profit_loss / initial_investment) * 100

                portfolio_projections[model_name] = {
                    'initial_value': initial_investment,
                    'final_value': final_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'shares': shares
                }

            analysis["analysis"]["portfolio_projections"] = portfolio_projections

        return analysis

    def format_analysis_for_llm(self, analysis_data: Dict) -> str:
        """Format analysis for LLM consumption"""
        symbol = analysis_data.get("symbol", "Unknown")
        formatted_text = f"MARKET ANALYSIS REPORT FOR {symbol}\n"
        formatted_text += "=" * 50 + "\n\n"

        if "technical_indicators" in analysis_data["analysis"]:
            indicators = analysis_data["analysis"]["technical_indicators"]
            formatted_text += "TECHNICAL INDICATORS:\n"
            formatted_text += f"RSI (14): {indicators.get('rsi', 'N/A'):.2f}\n"
            formatted_text += f"MACD: {indicators.get('macd', 'N/A'):.2f}\n"
            formatted_text += f"MACD Signal: {indicators.get('macd_signal', 'N/A'):.2f}\n"
            formatted_text += f"Bollinger Upper: ${indicators.get('bollinger_upper', 'N/A'):.2f}\n"
            formatted_text += f"Bollinger Lower: ${indicators.get('bollinger_lower', 'N/A'):.2f}\n"
            formatted_text += f"SMA 20: ${indicators.get('sma_20', 'N/A'):.2f}\n"
            formatted_text += f"SMA 50: ${indicators.get('sma_50', 'N/A'):.2f}\n\n"

        if "trading_signals" in analysis_data["analysis"]:
            signals = analysis_data["analysis"]["trading_signals"]
            formatted_text += "TRADING SIGNALS:\n"
            for indicator, signal in signals.items():
                formatted_text += f"{indicator.upper()}: {signal}\n"
            formatted_text += "\n"

        if "portfolio_projections" in analysis_data["analysis"]:
            projections = analysis_data["analysis"]["portfolio_projections"]
            formatted_text += f"{Config.MARKET_CONFIG['default_prediction_days']}-DAY PORTFOLIO PROJECTIONS (${Config.MARKET_CONFIG['default_investment']:,} INVESTMENT):\n"
            for model, projection in projections.items():
                formatted_text += f"{model.upper()}:\n"
                formatted_text += f"  Final Value: ${projection['final_value']:.2f}\n"
                formatted_text += f"  Profit/Loss: ${projection['profit_loss']:.2f} ({projection['profit_loss_pct']:+.1f}%)\n"
            formatted_text += "\n"

        return formatted_text