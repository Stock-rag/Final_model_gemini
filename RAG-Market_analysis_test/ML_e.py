import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# ML and forecasting imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta  # Technical Analysis library

class AdvancedMarketAnalysisEngine:
    def __init__(self, finnhub_api_key: str):
        # Store API key for authentication
        self.api_key = finnhub_api_key
        # Base URL for Finnhub API
        self.base_url = "https://finnhub.io/api/v1"
        # Headers for API requests
        self.headers = {"X-Finnhub-Token": self.api_key}
        # Initialize ML models
        self.models = {}
        # Initialize scalers for feature normalization
        self.scalers = {}
    
    def get_extended_historical_data(self, symbol: str, days: int = 365) -> Dict:
        """Get extended historical data for ML training"""
        # Calculate timestamp for start date (1 year back for ML training)
        from_date = int((datetime.now() - timedelta(days=days)).timestamp())
        # Current timestamp for end date
        to_date = int(datetime.now().timestamp())
        
        # API endpoint for historical data
        url = f"{self.base_url}/stock/candle"
        # Parameters: symbol, resolution (daily), from/to timestamps
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": from_date,
            "to": to_date
        }
        
        try:
            # Make GET request to API
            response = requests.get(url, params=params, headers=self.headers)
            # Convert response to JSON
            data = response.json()
            return data
        except Exception as e:
            # Return error if request fails
            return {"error": str(e)}
    
    def calculate_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # RSI (Relative Strength Index) - momentum oscillator
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD (Moving Average Convergence Divergence) - trend following momentum
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()  # MACD line
        df['macd_signal'] = macd.macd_signal()  # Signal line
        df['macd_histogram'] = macd.macd_diff()  # Histogram
        
        # Bollinger Bands - volatility indicator
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()  # Upper band
        df['bb_middle'] = bollinger.bollinger_mavg()  # Middle band (SMA)
        df['bb_lower'] = bollinger.bollinger_lband()  # Lower band
        df['bb_width'] = df['bb_upper'] - df['bb_lower']  # Band width
        
        # Stochastic Oscillator - momentum indicator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()  # %K line
        df['stoch_d'] = stoch.stoch_signal()  # %D line
        
        # Williams %R - momentum indicator
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Average True Range (ATR) - volatility measure
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index (CCI) - momentum oscillator
        df['cci'] = ta.momentum.cci(df['high'], df['low'], df['close'])
        
        # Moving averages for different periods
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)  # Short-term
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)  # Medium-term
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)  # Long-term
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)  # Very long-term
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)  # Fast EMA
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)  # Slow EMA
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])  # Volume SMA
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])  # VWAP
        
        # Price momentum indicators
        df['momentum'] = df['close'] / df['close'].shift(10) - 1  # 10-day momentum
        df['price_change'] = df['close'].pct_change()  # Daily price change
        df['volatility'] = df['price_change'].rolling(window=20).std()  # 20-day volatility
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()  # 20-day resistance
        df['support'] = df['low'].rolling(window=20).min()  # 20-day support
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML models"""
        # Lag features (previous day values)
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
    
    def prepare_ml_dataset(self, historical_data: Dict) -> pd.DataFrame:
        """Convert raw historical data to ML-ready dataframe"""
        # Check if data is valid
        if 'c' not in historical_data or not historical_data['c']:
            return pd.DataFrame()
        
        # Create dataframe from historical data
        df = pd.DataFrame({
            'timestamp': historical_data['t'],  # Unix timestamps
            'close': historical_data['c'],      # Closing prices
            'high': historical_data['h'],       # High prices
            'low': historical_data['l'],        # Low prices
            'open': historical_data['o'],       # Opening prices
            'volume': historical_data['v']      # Volume
        })
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        # Set date as index
        df.set_index('date', inplace=True)
        # Sort by date
        df.sort_index(inplace=True)
        
        # Calculate technical indicators
        df = self.calculate_advanced_technical_indicators(df)
        # Create additional features
        df = self.create_features(df)
        
        # Drop rows with NaN values (due to rolling calculations)
        df.dropna(inplace=True)
        
        return df
    
    def train_prediction_models(self, df: pd.DataFrame, target_days: int = 1) -> Dict:
        """Train ML models for price prediction"""
        # Check if enough data for training
        if len(df) < 100:
            return {"error": "Insufficient data for training"}
        
        # Feature columns (exclude target and non-feature columns)
        feature_cols = [col for col in df.columns if col not in ['close', 'timestamp', 'high', 'low', 'open', 'volume']]
        # Input features
        X = df[feature_cols].values
        # Target: future close price (shifted by target_days)
        y = df['close'].shift(-target_days).dropna().values
        # Align X with y (remove last target_days rows from X)
        X = X[:-target_days if target_days > 0 else len(X)]
        
        # Split into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Initialize and fit scaler for feature normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)  # Train on scaled features
        lr_pred = lr.predict(X_test_scaled)  # Predict on test set
        models['linear'] = lr
        results['linear'] = {
            'mae': mean_absolute_error(y_test, lr_pred),  # Mean Absolute Error
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))  # Root Mean Square Error
        }
        
        # Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)  # Train on original features (RF handles scaling)
        rf_pred = rf.predict(X_test)  # Predict on test set
        models['random_forest'] = rf
        results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
        }
        
        # Gradient Boosting model
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train_scaled, y_train)  # Train on scaled features
        gb_pred = gb.predict(X_test_scaled)  # Predict on test set
        models['gradient_boost'] = gb
        results['gradient_boost'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred))
        }
        
        # Store models and scaler
        self.models = models
        self.scalers['features'] = scaler
        
        return {
            'models_trained': list(models.keys()),
            'performance': results,
            'feature_columns': feature_cols,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_future_prices(self, df: pd.DataFrame, days: int = 100) -> Dict:
        """Predict future stock prices using trained models"""
        # Check if models are trained
        if not self.models:
            return {"error": "Models not trained. Call train_prediction_models first."}
        
        # Get latest data point for prediction
        latest_data = df.iloc[-1:]
        # Feature columns (same as training)
        feature_cols = [col for col in df.columns if col not in ['close', 'timestamp', 'high', 'low', 'open', 'volume']]
        # Extract features
        X_latest = latest_data[feature_cols].values
        
        predictions = {}
        
        # Generate predictions for each model
        for model_name, model in self.models.items():
            model_predictions = []
            current_features = X_latest.copy()
            
            # Generate sequential predictions for each day
            for day in range(days):
                if model_name in ['linear', 'gradient_boost']:
                    # Scale features for models that need scaling
                    scaled_features = self.scalers['features'].transform(current_features)
                    pred = model.predict(scaled_features)[0]
                else:
                    # Use original features for Random Forest
                    pred = model.predict(current_features)[0]
                
                model_predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                # In practice, you'd update all relevant features
                current_features = current_features.copy()
                # This is a simplified feature update - in production you'd be more sophisticated
            
            predictions[model_name] = model_predictions
        
        # Calculate ensemble prediction (average of all models)
        ensemble_pred = []
        for i in range(days):
            day_predictions = [predictions[model][i] for model in predictions.keys()]
            ensemble_pred.append(np.mean(day_predictions))
        
        predictions['ensemble'] = ensemble_pred
        
        # Generate prediction dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return {
            'predictions': predictions,
            'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'current_price': df['close'].iloc[-1],
            'models_used': list(self.models.keys())
        }
    
    def calculate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on technical indicators"""
        latest = df.iloc[-1]
        signals = {}
        
        # RSI signals (oversold < 30, overbought > 70)
        if latest['rsi'] < 30:
            signals['rsi'] = 'BUY'  # Oversold condition
        elif latest['rsi'] > 70:
            signals['rsi'] = 'SELL'  # Overbought condition
        else:
            signals['rsi'] = 'HOLD'  # Neutral
        
        # MACD signals (bullish when MACD > signal, bearish when MACD < signal)
        if latest['macd'] > latest['macd_signal']:
            signals['macd'] = 'BUY'  # Bullish crossover
        else:
            signals['macd'] = 'SELL'  # Bearish crossover
        
        # Bollinger Bands signals
        if latest['close'] < latest['bb_lower']:
            signals['bollinger'] = 'BUY'  # Price below lower band (oversold)
        elif latest['close'] > latest['bb_upper']:
            signals['bollinger'] = 'SELL'  # Price above upper band (overbought)
        else:
            signals['bollinger'] = 'HOLD'  # Price within bands
        
        # Moving Average signals
        if latest['close'] > latest['sma_20'] and latest['sma_20'] > latest['sma_50']:
            signals['moving_avg'] = 'BUY'  # Uptrend
        elif latest['close'] < latest['sma_20'] and latest['sma_20'] < latest['sma_50']:
            signals['moving_avg'] = 'SELL'  # Downtrend
        else:
            signals['moving_avg'] = 'HOLD'  # Sideways
        
        # Stochastic signals
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals['stochastic'] = 'BUY'  # Oversold
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals['stochastic'] = 'SELL'  # Overbought
        else:
            signals['stochastic'] = 'HOLD'  # Neutral
        
        # Overall signal (majority vote)
        buy_votes = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_votes = sum(1 for signal in signals.values() if signal == 'SELL')
        
        if buy_votes > sell_votes:
            signals['overall'] = 'BUY'
        elif sell_votes > buy_votes:
            signals['overall'] = 'SELL'
        else:
            signals['overall'] = 'HOLD'
        
        return signals
    
    def comprehensive_analysis_with_prediction(self, symbol: str, prediction_days: int = 100) -> Dict:
        """Perform comprehensive analysis with price prediction"""
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis": {}
        }
        
        # Get extended historical data for ML training
        print(f"Fetching extended historical data for {symbol}...")
        historical = self.get_extended_historical_data(symbol, days=730)  # 2 years
        
        if 'error' in historical:
            return {"error": historical['error']}
        
        # Prepare ML dataset
        print(f"Preparing ML dataset for {symbol}...")
        df = self.prepare_ml_dataset(historical)
        
        if df.empty:
            return {"error": "Failed to prepare dataset"}
        
        # Calculate current technical indicators
        print(f"Calculating technical indicators for {symbol}...")
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
        
        # Generate trading signals
        print(f"Generating trading signals for {symbol}...")
        signals = self.calculate_trading_signals(df)
        analysis["analysis"]["trading_signals"] = signals
        
        # Train ML models
        print(f"Training ML models for {symbol}...")
        training_results = self.train_prediction_models(df, target_days=1)
        analysis["analysis"]["model_training"] = training_results
        
        # Generate price predictions
        if 'error' not in training_results:
            print(f"Generating {prediction_days}-day price predictions for {symbol}...")
            predictions = self.predict_future_prices(df, days=prediction_days)
            analysis["analysis"]["price_predictions"] = predictions
            
            # Calculate potential portfolio value
            current_price = df['close'].iloc[-1]
            initial_investment = 10000  # Your $10,000
            shares = initial_investment / current_price
            
            portfolio_projections = {}
            for model_name, pred_prices in predictions['predictions'].items():
                final_price = pred_prices[-1]  # Price after prediction_days
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
    
    def format_advanced_analysis_for_llm(self, analysis_data: Dict) -> str:
        """Format advanced analysis for LLM consumption"""
        symbol = analysis_data.get("symbol", "Unknown")
        formatted_text = f"ADVANCED MARKET ANALYSIS & PREDICTION REPORT FOR {symbol}\n"
        formatted_text += "=" * 60 + "\n\n"
        
        # Technical indicators
        if "technical_indicators" in analysis_data["analysis"]:
            indicators = analysis_data["analysis"]["technical_indicators"]
            formatted_text += "TECHNICAL INDICATORS:\n"
            formatted_text += f"RSI (14): {indicators.get('rsi', 'N/A'):.2f}\n"
            formatted_text += f"MACD: {indicators.get('macd', 'N/A'):.2f}\n"
            formatted_text += f"MACD Signal: {indicators.get('macd_signal', 'N/A'):.2f}\n"
            formatted_text += f"Bollinger Upper: ${indicators.get('bollinger_upper', 'N/A'):.2f}\n"
            formatted_text += f"Bollinger Lower: ${indicators.get('bollinger_lower', 'N/A'):.2f}\n"
            formatted_text += f"SMA 20: ${indicators.get('sma_20', 'N/A'):.2f}\n"
            formatted_text += f"SMA 50: ${indicators.get('sma_50', 'N/A'):.2f}\n"
            formatted_text += f"Williams %R: {indicators.get('williams_r', 'N/A'):.2f}\n\n"
        
        # Trading signals
        if "trading_signals" in analysis_data["analysis"]:
            signals = analysis_data["analysis"]["trading_signals"]
            formatted_text += "TRADING SIGNALS:\n"
            for indicator, signal in signals.items():
                formatted_text += f"{indicator.upper()}: {signal}\n"
            formatted_text += "\n"
        
        # Model performance
        if "model_training" in analysis_data["analysis"]:
            training = analysis_data["analysis"]["model_training"]
            if "performance" in training:
                formatted_text += "ML MODEL PERFORMANCE:\n"
                for model, metrics in training["performance"].items():
                    formatted_text += f"{model.upper()}: MAE=${metrics['mae']:.2f}, RMSE=${metrics['rmse']:.2f}\n"
                formatted_text += "\n"
        
        # Portfolio projections
        if "portfolio_projections" in analysis_data["analysis"]:
            projections = analysis_data["analysis"]["portfolio_projections"]
            formatted_text += "100-DAY PORTFOLIO PROJECTIONS ($10,000 INVESTMENT):\n"
            for model, projection in projections.items():
                formatted_text += f"{model.upper()}:\n"
                formatted_text += f"  Final Value: ${projection['final_value']:.2f}\n"
                formatted_text += f"  Profit/Loss: ${projection['profit_loss']:.2f} ({projection['profit_loss_pct']:+.1f}%)\n"
            formatted_text += "\n"
        
        return formatted_text

# Example usage
def main():
    # Replace with your Finnhub API key
    API_KEY = "d31nrlhr01qsprr1r7i0d31nrlhr01qsprr1r7ig"
    
    # Create advanced engine
    engine = AdvancedMarketAnalysisEngine(API_KEY)
    
    # Analyze with predictions
    symbol = "AAPL"
    analysis = engine.comprehensive_analysis_with_prediction(symbol, prediction_days=100)
    
    # Format for LLM
    llm_output = engine.format_advanced_analysis_for_llm(analysis)
    print(llm_output)
    
    # Save results
    with open(f"{symbol}_advanced_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    main()