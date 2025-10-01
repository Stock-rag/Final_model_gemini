"""
ML Evaluation metrics for Stock RAG system: Precision, Recall, F1-Score
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json

class StockRAGEvaluator:
    """Evaluate Precision, Recall, F1-Score for stock predictions and recommendations"""

    def __init__(self):
        self.predictions = []
        self.signals = []
        self.retrieval_results = []

    def add_prediction(self, symbol: str, predicted_action: str, actual_outcome: str,
                      confidence: float, prediction_date: datetime, evaluation_date: datetime):
        """
        Add a stock prediction for evaluation

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            predicted_action: 'BUY', 'SELL', 'HOLD'
            actual_outcome: 'PROFIT', 'LOSS', 'NEUTRAL' (based on actual price movement)
            confidence: Model confidence (0-1)
            prediction_date: When prediction was made
            evaluation_date: When outcome was evaluated
        """
        self.predictions.append({
            'symbol': symbol,
            'predicted_action': predicted_action,
            'actual_outcome': actual_outcome,
            'confidence': confidence,
            'prediction_date': prediction_date,
            'evaluation_date': evaluation_date,
            'correct': self._is_prediction_correct(predicted_action, actual_outcome)
        })

    def add_signal(self, symbol: str, signal_type: str, signal_strength: str,
                  actual_movement: str, rsi: float, macd_signal: str):
        """
        Add technical analysis signal for evaluation

        Args:
            symbol: Stock symbol
            signal_type: 'RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'MACD_BULLISH', 'MACD_BEARISH'
            signal_strength: 'STRONG', 'MODERATE', 'WEAK'
            actual_movement: 'UP', 'DOWN', 'SIDEWAYS' (actual price movement after signal)
            rsi: RSI value when signal was generated
            macd_signal: 'BULLISH' or 'BEARISH'
        """
        self.signals.append({
            'symbol': symbol,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'actual_movement': actual_movement,
            'rsi': rsi,
            'macd_signal': macd_signal,
            'correct': self._is_signal_correct(signal_type, actual_movement)
        })

    def add_retrieval_result(self, query: str, retrieved_data: List[str],
                           relevant_data: List[str], response_quality: str):
        """
        Add RAG retrieval result for evaluation

        Args:
            query: User query
            retrieved_data: List of data sources retrieved (e.g., ['AAPL_price', 'AAPL_rsi'])
            relevant_data: List of actually relevant data for the query
            response_quality: 'EXCELLENT', 'GOOD', 'POOR'
        """
        self.retrieval_results.append({
            'query': query,
            'retrieved_data': retrieved_data,
            'relevant_data': relevant_data,
            'response_quality': response_quality,
            'precision': self._calculate_retrieval_precision(retrieved_data, relevant_data),
            'recall': self._calculate_retrieval_recall(retrieved_data, relevant_data)
        })

    def _is_prediction_correct(self, predicted: str, actual: str) -> bool:
        """Determine if prediction was correct"""
        correct_mappings = {
            'BUY': 'PROFIT',
            'SELL': 'PROFIT',  # Selling before a drop is profitable
            'HOLD': 'NEUTRAL'
        }
        return correct_mappings.get(predicted) == actual

    def _is_signal_correct(self, signal_type: str, actual_movement: str) -> bool:
        """Determine if technical signal was correct"""
        signal_expectations = {
            'RSI_OVERSOLD': 'UP',      # Oversold should lead to price increase
            'RSI_OVERBOUGHT': 'DOWN',   # Overbought should lead to price decrease
            'MACD_BULLISH': 'UP',      # Bullish MACD should lead to price increase
            'MACD_BEARISH': 'DOWN'     # Bearish MACD should lead to price decrease
        }
        return signal_expectations.get(signal_type) == actual_movement

    def _calculate_retrieval_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision for retrieval"""
        if not retrieved:
            return 0.0
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(retrieved)

    def _calculate_retrieval_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall for retrieval"""
        if not relevant:
            return 1.0
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(relevant)

    def evaluate_predictions(self) -> Dict:
        """Calculate Precision, Recall, F1 for stock predictions"""
        if not self.predictions:
            return {"error": "No predictions to evaluate"}

        # Convert to binary classification (correct vs incorrect)
        y_true = [1 if p['correct'] else 0 for p in self.predictions]
        y_pred = [1 for _ in self.predictions]  # All predictions made (vs not making predictions)

        # For multi-class evaluation (BUY/SELL/HOLD)
        actions = [p['predicted_action'] for p in self.predictions]
        outcomes = [p['actual_outcome'] for p in self.predictions]

        # Map outcomes to expected actions for evaluation
        outcome_to_action = {'PROFIT': 'BUY', 'LOSS': 'SELL', 'NEUTRAL': 'HOLD'}
        expected_actions = [outcome_to_action.get(outcome, 'HOLD') for outcome in outcomes]

        return {
            "binary_classification": {
                "total_predictions": len(self.predictions),
                "correct_predictions": sum(y_true),
                "accuracy": sum(y_true) / len(y_true),
                "precision": sum(y_true) / len(y_true),  # Since we're evaluating all predictions made
                "recall": sum(y_true) / len(y_true),     # Proportion of correct predictions
                "f1_score": f1_score(y_true, y_pred, average='binary', zero_division=0)
            },
            "multiclass_classification": {
                "precision_macro": precision_score(expected_actions, actions, average='macro', zero_division=0),
                "recall_macro": recall_score(expected_actions, actions, average='macro', zero_division=0),
                "f1_macro": f1_score(expected_actions, actions, average='macro', zero_division=0),
                "precision_weighted": precision_score(expected_actions, actions, average='weighted', zero_division=0),
                "recall_weighted": recall_score(expected_actions, actions, average='weighted', zero_division=0),
                "f1_weighted": f1_score(expected_actions, actions, average='weighted', zero_division=0),
                "classification_report": classification_report(expected_actions, actions, zero_division=0)
            },
            "by_action": self._evaluate_by_action(),
            "by_confidence": self._evaluate_by_confidence()
        }

    def evaluate_signals(self) -> Dict:
        """Calculate Precision, Recall, F1 for technical signals"""
        if not self.signals:
            return {"error": "No signals to evaluate"}

        # Overall signal accuracy
        y_true = [1 if s['correct'] else 0 for s in self.signals]
        total_signals = len(self.signals)
        correct_signals = sum(y_true)

        # By signal type
        signal_types = list(set(s['signal_type'] for s in self.signals))
        by_signal_type = {}

        for signal_type in signal_types:
            type_signals = [s for s in self.signals if s['signal_type'] == signal_type]
            type_correct = sum(1 for s in type_signals if s['correct'])

            by_signal_type[signal_type] = {
                "total": len(type_signals),
                "correct": type_correct,
                "precision": type_correct / len(type_signals) if type_signals else 0,
                "recall": type_correct / len(type_signals) if type_signals else 0  # Assuming all opportunities detected
            }

        return {
            "overall": {
                "total_signals": total_signals,
                "correct_signals": correct_signals,
                "accuracy": correct_signals / total_signals,
                "precision": correct_signals / total_signals,
                "recall": correct_signals / total_signals
            },
            "by_signal_type": by_signal_type,
            "confusion_matrix": self._create_signal_confusion_matrix()
        }

    def evaluate_retrieval(self) -> Dict:
        """Calculate Precision, Recall, F1 for RAG retrieval"""
        if not self.retrieval_results:
            return {"error": "No retrieval results to evaluate"}

        precisions = [r['precision'] for r in self.retrieval_results]
        recalls = [r['recall'] for r in self.retrieval_results]

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        return {
            "average_metrics": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1
            },
            "total_queries": len(self.retrieval_results),
            "perfect_retrievals": sum(1 for r in self.retrieval_results if r['precision'] == 1.0 and r['recall'] == 1.0),
            "by_quality": self._evaluate_retrieval_by_quality()
        }

    def _evaluate_by_action(self) -> Dict:
        """Evaluate metrics by prediction action"""
        actions = ['BUY', 'SELL', 'HOLD']
        results = {}

        for action in actions:
            action_predictions = [p for p in self.predictions if p['predicted_action'] == action]
            if action_predictions:
                correct = sum(1 for p in action_predictions if p['correct'])
                results[action] = {
                    "total": len(action_predictions),
                    "correct": correct,
                    "precision": correct / len(action_predictions),
                    "recall": correct / len(action_predictions)  # Within this action type
                }

        return results

    def _evaluate_by_confidence(self) -> Dict:
        """Evaluate metrics by confidence levels"""
        high_conf = [p for p in self.predictions if p['confidence'] >= 0.8]
        med_conf = [p for p in self.predictions if 0.5 <= p['confidence'] < 0.8]
        low_conf = [p for p in self.predictions if p['confidence'] < 0.5]

        results = {}
        for conf_level, preds, name in [(high_conf, 'high'), (med_conf, 'medium'), (low_conf, 'low')]:
            if preds:
                correct = sum(1 for p in preds if p['correct'])
                results[name] = {
                    "total": len(preds),
                    "correct": correct,
                    "accuracy": correct / len(preds)
                }

        return results

    def _create_signal_confusion_matrix(self) -> Dict:
        """Create confusion matrix for signal predictions"""
        # Simplified: correct vs incorrect signals
        y_true = ['Correct' if s['correct'] else 'Incorrect' for s in self.signals]
        y_pred = ['Predicted' for _ in self.signals]  # All signals were predicted

        unique_labels = list(set(y_true))
        cm = confusion_matrix(y_true, ['Correct' if s['correct'] else 'Incorrect' for s in self.signals], labels=unique_labels)

        return {
            "matrix": cm.tolist(),
            "labels": unique_labels
        }

    def _evaluate_retrieval_by_quality(self) -> Dict:
        """Evaluate retrieval by response quality"""
        quality_levels = ['EXCELLENT', 'GOOD', 'POOR']
        results = {}

        for quality in quality_levels:
            quality_results = [r for r in self.retrieval_results if r['response_quality'] == quality]
            if quality_results:
                avg_precision = np.mean([r['precision'] for r in quality_results])
                avg_recall = np.mean([r['recall'] for r in quality_results])
                results[quality] = {
                    "count": len(quality_results),
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall
                }

        return results

    def print_comprehensive_report(self):
        """Print detailed evaluation report"""
        print("\n" + "="*80)
        print("📊 STOCK RAG SYSTEM - ML EVALUATION REPORT")
        print("="*80)

        # Predictions evaluation
        pred_eval = self.evaluate_predictions()
        if "error" not in pred_eval:
            print("\n🎯 PREDICTION PERFORMANCE:")
            binary = pred_eval["binary_classification"]
            multiclass = pred_eval["multiclass_classification"]

            print(f"   Total Predictions: {binary['total_predictions']}")
            print(f"   Accuracy: {binary['accuracy']:.3f}")
            print(f"   Precision (Macro): {multiclass['precision_macro']:.3f}")
            print(f"   Recall (Macro): {multiclass['recall_macro']:.3f}")
            print(f"   F1-Score (Macro): {multiclass['f1_macro']:.3f}")

            print(f"\n   📈 BY ACTION TYPE:")
            for action, metrics in pred_eval["by_action"].items():
                print(f"   {action}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

        # Signals evaluation
        signal_eval = self.evaluate_signals()
        if "error" not in signal_eval:
            print(f"\n🔍 TECHNICAL SIGNALS PERFORMANCE:")
            overall = signal_eval["overall"]
            print(f"   Total Signals: {overall['total_signals']}")
            print(f"   Accuracy: {overall['accuracy']:.3f}")
            print(f"   Precision: {overall['precision']:.3f}")
            print(f"   Recall: {overall['recall']:.3f}")

        # Retrieval evaluation
        retrieval_eval = self.evaluate_retrieval()
        if "error" not in retrieval_eval:
            print(f"\n🔎 RAG RETRIEVAL PERFORMANCE:")
            avg_metrics = retrieval_eval["average_metrics"]
            print(f"   Total Queries: {retrieval_eval['total_queries']}")
            print(f"   Precision: {avg_metrics['precision']:.3f}")
            print(f"   Recall: {avg_metrics['recall']:.3f}")
            print(f"   F1-Score: {avg_metrics['f1_score']:.3f}")
            print(f"   Perfect Retrievals: {retrieval_eval['perfect_retrievals']}")

        print("="*80)

    def export_results(self, filename: str = "evaluation_results.json"):
        """Export all evaluation results to JSON"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "predictions_evaluation": self.evaluate_predictions(),
            "signals_evaluation": self.evaluate_signals(),
            "retrieval_evaluation": self.evaluate_retrieval(),
            "raw_data": {
                "predictions": self.predictions,
                "signals": self.signals,
                "retrieval_results": self.retrieval_results
            }
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📁 Evaluation results exported to {filename}")

# Demo and testing functions
def demo_evaluation():
    """Demonstrate the evaluation system with sample data"""
    evaluator = StockRAGEvaluator()

    print("🧪 Adding sample predictions...")

    # Add sample predictions
    base_date = datetime.now() - timedelta(days=30)

    # Sample stock predictions
    sample_predictions = [
        ("AAPL", "BUY", "PROFIT", 0.85),
        ("AAPL", "BUY", "LOSS", 0.75),
        ("TSLA", "SELL", "PROFIT", 0.90),
        ("MSFT", "HOLD", "NEUTRAL", 0.60),
        ("GOOGL", "BUY", "PROFIT", 0.80),
        ("NVDA", "SELL", "LOSS", 0.70),
    ]

    for i, (symbol, action, outcome, confidence) in enumerate(sample_predictions):
        evaluator.add_prediction(
            symbol, action, outcome, confidence,
            base_date + timedelta(days=i),
            base_date + timedelta(days=i+7)
        )

    # Add sample technical signals
    sample_signals = [
        ("AAPL", "RSI_OVERSOLD", "STRONG", "UP", 25.5, "BULLISH"),
        ("TSLA", "RSI_OVERBOUGHT", "MODERATE", "DOWN", 75.2, "BEARISH"),
        ("MSFT", "MACD_BULLISH", "STRONG", "UP", 45.0, "BULLISH"),
        ("GOOGL", "MACD_BEARISH", "WEAK", "UP", 55.0, "BEARISH"),  # Wrong signal
    ]

    for signal_data in sample_signals:
        evaluator.add_signal(*signal_data)

    # Add sample retrieval results
    sample_retrievals = [
        ("What is AAPL price?", ["AAPL_price", "AAPL_volume"], ["AAPL_price"], "EXCELLENT"),
        ("RSI for Tesla", ["TSLA_rsi", "TSLA_price", "TSLA_news"], ["TSLA_rsi"], "GOOD"),
        ("Market overview", ["SPY_price"], ["SPY_price", "market_news", "sector_performance"], "POOR"),
    ]

    for retrieval_data in sample_retrievals:
        evaluator.add_retrieval_result(*retrieval_data)

    # Generate and display report
    evaluator.print_comprehensive_report()
    evaluator.export_results("demo_evaluation.json")

if __name__ == "__main__":
    demo_evaluation()