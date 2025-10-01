"""
Historical Backtesting System for Stock RAG Evaluation
Uses real market data as ground truth for precision, recall, and F1-score calculation
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import sys
sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')
from data_sources import MultiMarketDataProvider

@dataclass
class PredictionRecord:
    symbol: str
    prediction_date: datetime
    predicted_action: str  # BUY, SELL, HOLD
    confidence: float
    price_at_prediction: float
    rsi_at_prediction: float
    macd_at_prediction: float
    reasoning: str

@dataclass
class OutcomeRecord:
    symbol: str
    prediction_date: datetime
    evaluation_date: datetime
    price_at_prediction: float
    price_at_evaluation: float
    actual_return: float
    actual_outcome: str  # PROFIT, LOSS, NEUTRAL
    days_held: int

class HistoricalBacktester:
    """Historical backtesting system using real market data"""

    def __init__(self, data_provider: MultiMarketDataProvider = None):
        self.data_provider = data_provider or MultiMarketDataProvider()
        self.predictions = []
        self.outcomes = []
        self.signal_evaluations = []

    def get_historical_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get historical price for a specific date"""
        try:
            # Get data for the date range
            start_date = date - timedelta(days=5)  # Buffer for weekends
            end_date = date + timedelta(days=5)

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                return None

            # Find closest trading day
            closest_date = min(hist.index, key=lambda x: abs((x.date() - date.date()).days))
            return float(hist.loc[closest_date]['Close'])

        except Exception as e:
            print(f"Error getting historical price for {symbol} on {date}: {e}")
            return None

    def get_historical_indicators(self, symbol: str, date: datetime) -> Dict:
        """Get historical technical indicators for a specific date"""
        try:
            # Get enough historical data to calculate indicators
            start_date = date - timedelta(days=100)
            end_date = date + timedelta(days=5)

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) < 50:  # Need enough data for indicators
                return {}

            # Calculate indicators using same methods as data_provider
            close_prices = hist['Close']

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()

            # Find closest date to target (fix timezone issue)
            target_date = pd.Timestamp(date.date()).tz_localize(hist.index.tz)
            closest_idx = hist.index.get_indexer([target_date], method='nearest')[0]

            if closest_idx >= 0 and closest_idx < len(hist):
                return {
                    'rsi': float(rsi.iloc[closest_idx]) if not pd.isna(rsi.iloc[closest_idx]) else None,
                    'macd': float(macd_line.iloc[closest_idx]) if not pd.isna(macd_line.iloc[closest_idx]) else None,
                    'macd_signal': float(signal_line.iloc[closest_idx]) if not pd.isna(signal_line.iloc[closest_idx]) else None,
                    'price': float(hist['Close'].iloc[closest_idx])
                }

        except Exception as e:
            print(f"Error getting historical indicators for {symbol} on {date}: {e}")

        return {}

    def simulate_historical_predictions(self, symbols: List[str],
                                      start_date: datetime,
                                      end_date: datetime,
                                      prediction_frequency_days: int = 7) -> List[PredictionRecord]:
        """
        Simulate what predictions our system would have made historically
        """
        predictions = []
        current_date = start_date

        print(f"🔄 Simulating historical predictions from {start_date.date()} to {end_date.date()}")

        while current_date <= end_date:
            for symbol in symbols:
                try:
                    # Get historical indicators for this date
                    indicators = self.get_historical_indicators(symbol, current_date)

                    if not indicators or indicators.get('rsi') is None:
                        continue

                    # Simulate prediction logic based on technical indicators
                    rsi = indicators['rsi']
                    macd = indicators.get('macd', 0)
                    macd_signal = indicators.get('macd_signal', 0)
                    price = indicators['price']

                    # Simple prediction rules (you can enhance these)
                    predicted_action = "HOLD"
                    confidence = 0.5
                    reasoning = "Default hold"

                    if rsi < 30:  # Oversold
                        predicted_action = "BUY"
                        confidence = 0.8
                        reasoning = f"RSI oversold at {rsi:.2f}"
                    elif rsi > 70:  # Overbought
                        predicted_action = "SELL"
                        confidence = 0.8
                        reasoning = f"RSI overbought at {rsi:.2f}"
                    elif macd > macd_signal:  # MACD bullish
                        if predicted_action == "HOLD":
                            predicted_action = "BUY"
                            confidence = 0.6
                            reasoning = f"MACD bullish crossover"
                    elif macd < macd_signal:  # MACD bearish
                        if predicted_action == "HOLD":
                            predicted_action = "SELL"
                            confidence = 0.6
                            reasoning = f"MACD bearish crossover"

                    # Only make prediction if confidence is reasonable
                    if confidence >= 0.6:
                        prediction = PredictionRecord(
                            symbol=symbol,
                            prediction_date=current_date,
                            predicted_action=predicted_action,
                            confidence=confidence,
                            price_at_prediction=price,
                            rsi_at_prediction=rsi,
                            macd_at_prediction=macd,
                            reasoning=reasoning
                        )
                        predictions.append(prediction)

                        print(f"   📊 {symbol} on {current_date.date()}: {predicted_action} (RSI: {rsi:.1f}, Price: ${price:.2f})")

                except Exception as e:
                    print(f"Error processing {symbol} on {current_date}: {e}")

            current_date += timedelta(days=prediction_frequency_days)

        print(f"✅ Generated {len(predictions)} historical predictions")
        return predictions

    def evaluate_predictions(self, predictions: List[PredictionRecord],
                           holding_period_days: int = 14) -> List[OutcomeRecord]:
        """
        Evaluate historical predictions against actual market performance
        """
        outcomes = []

        print(f"🔍 Evaluating {len(predictions)} predictions with {holding_period_days}-day holding period...")

        for i, prediction in enumerate(predictions, 1):
            try:
                evaluation_date = prediction.prediction_date + timedelta(days=holding_period_days)

                # Get actual price at evaluation date
                price_at_evaluation = self.get_historical_price(prediction.symbol, evaluation_date)

                if price_at_evaluation is None:
                    continue

                # Calculate actual return
                actual_return = (price_at_evaluation - prediction.price_at_prediction) / prediction.price_at_prediction

                # Determine actual outcome based on return thresholds
                if actual_return > 0.02:  # 2% gain threshold
                    actual_outcome = "PROFIT"
                elif actual_return < -0.02:  # 2% loss threshold
                    actual_outcome = "LOSS"
                else:
                    actual_outcome = "NEUTRAL"

                outcome = OutcomeRecord(
                    symbol=prediction.symbol,
                    prediction_date=prediction.prediction_date,
                    evaluation_date=evaluation_date,
                    price_at_prediction=prediction.price_at_prediction,
                    price_at_evaluation=price_at_evaluation,
                    actual_return=actual_return,
                    actual_outcome=actual_outcome,
                    days_held=holding_period_days
                )
                outcomes.append(outcome)

                if i % 10 == 0:
                    print(f"   Processed {i}/{len(predictions)} predictions...")

            except Exception as e:
                print(f"Error evaluating prediction {i}: {e}")

        print(f"✅ Evaluated {len(outcomes)} predictions successfully")
        return outcomes

    def calculate_prediction_accuracy(self, predictions: List[PredictionRecord],
                                    outcomes: List[OutcomeRecord]) -> Dict:
        """
        Calculate precision, recall, and F1-score for predictions using REAL market data
        """
        # Create prediction-outcome pairs
        prediction_map = {(p.symbol, p.prediction_date): p for p in predictions}
        outcome_map = {(o.symbol, o.prediction_date): o for o in outcomes}

        # Match predictions with outcomes
        matched_pairs = []
        for key in prediction_map.keys():
            if key in outcome_map:
                matched_pairs.append((prediction_map[key], outcome_map[key]))

        if not matched_pairs:
            return {"error": "No matched prediction-outcome pairs found"}

        print(f"📊 Calculating accuracy for {len(matched_pairs)} matched predictions...")

        # Calculate accuracy metrics
        correct_predictions = 0
        total_predictions = len(matched_pairs)

        # By action type
        action_stats = {"BUY": {"total": 0, "correct": 0},
                       "SELL": {"total": 0, "correct": 0},
                       "HOLD": {"total": 0, "correct": 0}}

        # For sklearn metrics
        y_true = []  # Expected action based on actual outcome
        y_pred = []  # Predicted action

        for prediction, outcome in matched_pairs:
            # Determine what action SHOULD have been taken based on actual outcome
            if outcome.actual_outcome == "PROFIT":
                expected_action = "BUY"
            elif outcome.actual_outcome == "LOSS":
                expected_action = "SELL"
            else:
                expected_action = "HOLD"

            y_true.append(expected_action)
            y_pred.append(prediction.predicted_action)

            # Check if prediction was correct
            is_correct = self._is_prediction_correct(prediction.predicted_action, outcome.actual_outcome)
            if is_correct:
                correct_predictions += 1

            # Update action stats
            action_stats[prediction.predicted_action]["total"] += 1
            if is_correct:
                action_stats[prediction.predicted_action]["correct"] += 1

        # Calculate overall metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate precision, recall, F1 by action
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "by_action": {
                action: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                }
                for action, stats in action_stats.items()
            },
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
            "sample_predictions": [
                {
                    "symbol": p.symbol,
                    "date": p.prediction_date.strftime("%Y-%m-%d"),
                    "predicted": p.predicted_action,
                    "actual_outcome": o.actual_outcome,
                    "return": f"{o.actual_return:.3f}",
                    "correct": self._is_prediction_correct(p.predicted_action, o.actual_outcome)
                }
                for p, o in matched_pairs[:10]  # Show first 10 examples
            ]
        }

    def _is_prediction_correct(self, predicted_action: str, actual_outcome: str) -> bool:
        """Determine if prediction was correct based on actual market outcome"""
        if predicted_action == "BUY" and actual_outcome == "PROFIT":
            return True
        elif predicted_action == "SELL" and actual_outcome == "LOSS":
            return True  # Avoided loss by selling
        elif predicted_action == "HOLD" and actual_outcome == "NEUTRAL":
            return True
        return False

    def evaluate_signal_accuracy(self, predictions: List[PredictionRecord],
                               outcomes: List[OutcomeRecord]) -> Dict:
        """Evaluate RSI and MACD signal accuracy"""
        signal_results = {
            "rsi_oversold": {"total": 0, "correct": 0},
            "rsi_overbought": {"total": 0, "correct": 0},
            "macd_bullish": {"total": 0, "correct": 0},
            "macd_bearish": {"total": 0, "correct": 0}
        }

        outcome_map = {(o.symbol, o.prediction_date): o for o in outcomes}

        for prediction in predictions:
            key = (prediction.symbol, prediction.prediction_date)
            if key not in outcome_map:
                continue

            outcome = outcome_map[key]

            # RSI signals
            if prediction.rsi_at_prediction < 30:  # Oversold
                signal_results["rsi_oversold"]["total"] += 1
                if outcome.actual_outcome == "PROFIT":
                    signal_results["rsi_oversold"]["correct"] += 1

            elif prediction.rsi_at_prediction > 70:  # Overbought
                signal_results["rsi_overbought"]["total"] += 1
                if outcome.actual_outcome == "LOSS":
                    signal_results["rsi_overbought"]["correct"] += 1

            # MACD signals (if we have MACD data)
            if "MACD bullish" in prediction.reasoning:
                signal_results["macd_bullish"]["total"] += 1
                if outcome.actual_outcome == "PROFIT":
                    signal_results["macd_bullish"]["correct"] += 1

            elif "MACD bearish" in prediction.reasoning:
                signal_results["macd_bearish"]["total"] += 1
                if outcome.actual_outcome == "LOSS":
                    signal_results["macd_bearish"]["correct"] += 1

        # Calculate precision/recall for each signal type
        for signal_type in signal_results:
            stats = signal_results[signal_type]
            if stats["total"] > 0:
                stats["precision"] = stats["correct"] / stats["total"]
                stats["recall"] = stats["correct"] / stats["total"]  # Same as precision for this case
                stats["f1_score"] = 2 * (stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"]) if (stats["precision"] + stats["recall"]) > 0 else 0

        return signal_results

    def run_backtest(self, symbols: List[str],
                    start_date: datetime,
                    end_date: datetime,
                    holding_period_days: int = 14) -> Dict:
        """
        Run complete historical backtest with real market data
        """
        print(f"🚀 Starting Historical Backtest")
        print(f"   Symbols: {symbols}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Holding Period: {holding_period_days} days")
        print("="*60)

        # Step 1: Generate historical predictions
        predictions = self.simulate_historical_predictions(symbols, start_date, end_date)

        if not predictions:
            return {"error": "No predictions generated"}

        # Step 2: Evaluate predictions against actual market performance
        outcomes = self.evaluate_predictions(predictions, holding_period_days)

        if not outcomes:
            return {"error": "No outcomes calculated"}

        # Step 3: Calculate accuracy metrics
        accuracy_results = self.calculate_prediction_accuracy(predictions, outcomes)

        # Step 4: Evaluate signal accuracy
        signal_results = self.evaluate_signal_accuracy(predictions, outcomes)

        # Step 5: Compile comprehensive results
        results = {
            "backtest_config": {
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "holding_period_days": holding_period_days,
                "total_predictions": len(predictions),
                "total_outcomes": len(outcomes)
            },
            "accuracy_metrics": accuracy_results,
            "signal_accuracy": signal_results,
            "summary": {
                "overall_accuracy": accuracy_results.get("accuracy", 0),
                "precision_macro": accuracy_results.get("precision_macro", 0),
                "recall_macro": accuracy_results.get("recall_macro", 0),
                "f1_macro": accuracy_results.get("f1_macro", 0)
            }
        }

        return results

    def print_backtest_report(self, results: Dict):
        """Print formatted backtest report"""
        if "error" in results:
            print(f"❌ Backtest Error: {results['error']}")
            return

        config = results["backtest_config"]
        accuracy = results["accuracy_metrics"]
        signals = results["signal_accuracy"]
        summary = results["summary"]

        print("\n" + "="*80)
        print("📊 HISTORICAL BACKTEST RESULTS (REAL MARKET DATA)")
        print("="*80)

        print(f"\n🔧 BACKTEST CONFIGURATION:")
        print(f"   Symbols: {', '.join(config['symbols'])}")
        print(f"   Period: {config['start_date'][:10]} to {config['end_date'][:10]}")
        print(f"   Holding Period: {config['holding_period_days']} days")
        print(f"   Total Predictions: {config['total_predictions']}")

        print(f"\n🎯 PREDICTION ACCURACY (GROUND TRUTH: ACTUAL MARKET PERFORMANCE):")
        print(f"   Overall Accuracy: {summary['overall_accuracy']:.3f}")
        print(f"   Precision (Macro): {summary['precision_macro']:.3f}")
        print(f"   Recall (Macro): {summary['recall_macro']:.3f}")
        print(f"   F1-Score (Macro): {summary['f1_macro']:.3f}")

        if "by_action" in accuracy:
            print(f"\n📈 ACCURACY BY PREDICTION TYPE:")
            for action, stats in accuracy["by_action"].items():
                print(f"   {action}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")

        print(f"\n🔍 TECHNICAL SIGNAL ACCURACY:")
        for signal_type, stats in signals.items():
            if stats["total"] > 0:
                print(f"   {signal_type.upper()}: {stats['correct']}/{stats['total']} = {stats.get('precision', 0):.3f}")

        if "sample_predictions" in accuracy:
            print(f"\n📋 SAMPLE PREDICTIONS:")
            for sample in accuracy["sample_predictions"][:5]:
                print(f"   {sample['symbol']} {sample['date']}: {sample['predicted']} → {sample['actual_outcome']} (Return: {sample['return']}) {'✅' if sample['correct'] else '❌'}")

        print("="*80)

# Demo function
def demo_historical_backtest():
    """Demonstrate historical backtesting"""
    backtester = HistoricalBacktester()

    # Test with major stocks over a more recent period
    symbols = ["AAPL", "TSLA", "MSFT"]
    end_date = datetime.now() - timedelta(days=30)  # End 30 days ago to have evaluation period
    start_date = end_date - timedelta(days=90)  # 3 months of data (more realistic)

    # Run backtest
    results = backtester.run_backtest(symbols, start_date, end_date, holding_period_days=14)

    # Print results
    backtester.print_backtest_report(results)

    # Save results
    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to backtest_results.json")

if __name__ == "__main__":
    demo_historical_backtest()