"""
Simplified Historical Backtesting for demonstration
Uses 2024 data as ground truth
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

class SimpleBacktester:
    """Simplified backtester with real market data ground truth"""

    def __init__(self):
        self.predictions = []
        self.outcomes = []

    def get_historical_data(self, symbol: str, start_date: str, end_date: str):
        """Get historical data for backtesting"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                return None

            # Calculate technical indicators
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            # Calculate MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

            return data

        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def simulate_predictions(self, data, symbol):
        """Simulate what predictions our system would have made"""
        predictions = []

        for i in range(50, len(data) - 14):  # Need enough history for indicators and future for evaluation
            current_date = data.index[i]
            current_row = data.iloc[i]

            # Skip if we don't have indicator data
            if pd.isna(current_row['RSI']) or pd.isna(current_row['MACD']):
                continue

            # Simulate prediction logic
            rsi = current_row['RSI']
            macd = current_row['MACD']
            macd_signal = current_row['MACD_Signal']
            current_price = current_row['Close']

            # Simple prediction rules
            predicted_action = "HOLD"
            confidence = 0.5

            if rsi < 30:  # Oversold - BUY signal
                predicted_action = "BUY"
                confidence = 0.8
            elif rsi > 70:  # Overbought - SELL signal
                predicted_action = "SELL"
                confidence = 0.8
            elif macd > macd_signal and predicted_action == "HOLD":  # MACD crossover
                predicted_action = "BUY"
                confidence = 0.6

            # Calculate actual outcome 14 days later
            future_date = data.index[i + 14]
            future_price = data.iloc[i + 14]['Close']
            actual_return = (future_price - current_price) / current_price

            # Determine ground truth outcome
            if actual_return > 0.02:  # 2% gain
                actual_outcome = "PROFIT"
            elif actual_return < -0.02:  # 2% loss
                actual_outcome = "LOSS"
            else:
                actual_outcome = "NEUTRAL"

            predictions.append({
                'symbol': symbol,
                'date': current_date,
                'predicted_action': predicted_action,
                'confidence': confidence,
                'current_price': current_price,
                'future_price': future_price,
                'actual_return': actual_return,
                'actual_outcome': actual_outcome,
                'rsi': rsi,
                'macd_bullish': macd > macd_signal
            })

        return predictions

    def evaluate_predictions(self, predictions):
        """Calculate precision, recall, F1 with REAL market data ground truth"""

        if not predictions:
            return {"error": "No predictions to evaluate"}

        # Prepare data for sklearn metrics
        y_true = []  # What action should have been taken based on actual outcome
        y_pred = []  # What action was predicted

        correct_predictions = 0
        total_predictions = len(predictions)

        # Action-wise statistics
        action_stats = {
            'BUY': {'total': 0, 'correct': 0, 'profitable': 0},
            'SELL': {'total': 0, 'correct': 0, 'profitable': 0},
            'HOLD': {'total': 0, 'correct': 0, 'profitable': 0}
        }

        for pred in predictions:
            predicted = pred['predicted_action']
            actual_outcome = pred['actual_outcome']

            # Determine what the correct action should have been
            if actual_outcome == "PROFIT":
                correct_action = "BUY"
            elif actual_outcome == "LOSS":
                correct_action = "SELL"  # Should have sold to avoid loss
            else:
                correct_action = "HOLD"

            y_true.append(correct_action)
            y_pred.append(predicted)

            # Check if prediction was correct
            is_correct = (predicted == correct_action)
            if is_correct:
                correct_predictions += 1

            # Update action statistics
            action_stats[predicted]['total'] += 1
            if is_correct:
                action_stats[predicted]['correct'] += 1
            if pred['actual_return'] > 0:
                action_stats[predicted]['profitable'] += 1

        # Calculate sklearn metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Calculate profitability metrics
        buy_predictions = [p for p in predictions if p['predicted_action'] == 'BUY']
        sell_predictions = [p for p in predictions if p['predicted_action'] == 'SELL']

        buy_success_rate = sum(1 for p in buy_predictions if p['actual_return'] > 0) / len(buy_predictions) if buy_predictions else 0
        sell_success_rate = sum(1 for p in sell_predictions if p['actual_return'] < 0) / len(sell_predictions) if sell_predictions else 0

        return {
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "overall_accuracy": correct_predictions / total_predictions,

            # Standard ML Metrics
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,

            # Trading-specific metrics
            "buy_success_rate": buy_success_rate,
            "sell_success_rate": sell_success_rate,

            # Action-wise breakdown
            "by_action": {
                action: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "profitability": stats["profitable"] / stats["total"] if stats["total"] > 0 else 0
                }
                for action, stats in action_stats.items()
            },

            "classification_report": classification_report(y_true, y_pred, zero_division=0),

            # Sample predictions for inspection
            "sample_predictions": [
                {
                    "symbol": p["symbol"],
                    "date": str(p["date"].date()),
                    "predicted": p["predicted_action"],
                    "actual_outcome": p["actual_outcome"],
                    "return": f"{p['actual_return']:.3f}",
                    "rsi": f"{p['rsi']:.1f}",
                    "price": f"${p['current_price']:.2f} → ${p['future_price']:.2f}"
                }
                for p in predictions[:10]
            ]
        }

    def run_backtest(self, symbols, start_date="2024-01-01", end_date="2024-10-01"):
        """Run complete backtest with real market data"""
        all_predictions = []
        all_results = {}

        print(f"🚀 Running Historical Backtest with REAL Market Data")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Symbols: {symbols}")
        print("="*60)

        for symbol in symbols:
            print(f"\n📊 Processing {symbol}...")

            # Get historical data
            data = self.get_historical_data(symbol, start_date, end_date)
            if data is None or len(data) < 100:
                print(f"   ❌ Insufficient data for {symbol}")
                continue

            # Generate predictions
            predictions = self.simulate_predictions(data, symbol)
            print(f"   ✅ Generated {len(predictions)} predictions for {symbol}")

            all_predictions.extend(predictions)

        if not all_predictions:
            return {"error": "No predictions generated"}

        # Evaluate all predictions
        print(f"\n🔍 Evaluating {len(all_predictions)} total predictions...")
        results = self.evaluate_predictions(all_predictions)

        # Add configuration info
        results["config"] = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "total_predictions": len(all_predictions)
        }

        return results

    def print_results(self, results):
        """Print formatted backtest results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return

        config = results["config"]

        print(f"\n📊 HISTORICAL BACKTEST RESULTS")
        print("="*60)
        print(f"🎯 GROUND TRUTH: Real {config['start_date']} to {config['end_date']} market performance")
        print(f"📈 Symbols: {', '.join(config['symbols'])}")
        print(f"🔢 Total Predictions: {config['total_predictions']}")
        print("="*60)

        print(f"\n🎯 PRECISION, RECALL & F1-SCORE:")
        print(f"   Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"   Precision (Macro): {results['precision_macro']:.3f}")
        print(f"   Recall (Macro): {results['recall_macro']:.3f}")
        print(f"   F1-Score (Macro): {results['f1_macro']:.3f}")

        print(f"\n💰 TRADING PERFORMANCE:")
        print(f"   BUY Signal Success Rate: {results['buy_success_rate']:.3f}")
        print(f"   SELL Signal Success Rate: {results['sell_success_rate']:.3f}")

        print(f"\n📋 BY ACTION TYPE:")
        for action, stats in results["by_action"].items():
            print(f"   {action}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.3f}), {stats['profitability']:.3f} profitable")

        print(f"\n🔍 SAMPLE PREDICTIONS (REAL OUTCOMES):")
        for sample in results["sample_predictions"]:
            print(f"   {sample['symbol']} {sample['date']}: {sample['predicted']} → {sample['actual_outcome']} (Return: {sample['return']}, RSI: {sample['rsi']})")

        print("="*60)

def main():
    """Demo the simplified backtester"""
    backtester = SimpleBacktester()

    # Run backtest on 2024 data (definitely exists)
    symbols = ["AAPL", "TSLA", "MSFT"]
    results = backtester.run_backtest(symbols, "2024-01-01", "2024-09-01")

    # Print results
    backtester.print_results(results)

    # Save results
    import json
    with open("simple_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to simple_backtest_results.json")

if __name__ == "__main__":
    main()