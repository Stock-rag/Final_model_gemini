"""
RAG System Historical Performance Evaluation
Measures retrieval quality, factual accuracy, and response relevance using historical data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import re
import sys
sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')
from data_sources import MultiMarketDataProvider

class RAGHistoricalEvaluator:
    """Evaluate RAG system performance using historical market data as ground truth"""

    def __init__(self):
        self.data_provider = MultiMarketDataProvider()
        self.evaluations = []

    def get_historical_ground_truth(self, symbol: str, date: datetime) -> Dict:
        """Get historical ground truth data for a specific date"""
        try:
            # Get data around the target date
            start_date = date - timedelta(days=5)
            end_date = date + timedelta(days=5)

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                return {}

            # Find closest trading day
            target_date = min(hist.index, key=lambda x: abs((x.date() - date.date()).days))
            day_data = hist.loc[target_date]

            # Calculate technical indicators for ground truth
            if len(hist) >= 50:  # Need enough data for indicators
                close_prices = hist['Close']

                # RSI calculation
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))

                # MACD calculation
                ema_12 = close_prices.ewm(span=12).mean()
                ema_26 = close_prices.ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()

                # Get values for target date
                target_idx = hist.index.get_loc(target_date)

                ground_truth = {
                    'symbol': symbol,
                    'date': target_date,
                    'price': float(day_data['Close']),
                    'volume': int(day_data['Volume']),
                    'high': float(day_data['High']),
                    'low': float(day_data['Low']),
                    'change': float(day_data['Close'] - day_data['Open']),
                    'change_percent': float((day_data['Close'] - day_data['Open']) / day_data['Open'] * 100),
                    'rsi': float(rsi_series.iloc[target_idx]) if target_idx < len(rsi_series) and not pd.isna(rsi_series.iloc[target_idx]) else None,
                    'macd': float(macd_line.iloc[target_idx]) if target_idx < len(macd_line) and not pd.isna(macd_line.iloc[target_idx]) else None,
                    'macd_signal': float(signal_line.iloc[target_idx]) if target_idx < len(signal_line) and not pd.isna(signal_line.iloc[target_idx]) else None
                }

                return ground_truth

        except Exception as e:
            print(f"Error getting ground truth for {symbol} on {date}: {e}")

        return {}

    def simulate_rag_queries(self, symbols: List[str], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Simulate RAG queries that would have been made historically"""
        queries = []
        current_date = start_date

        query_templates = [
            "What is {symbol} stock price on {date}?",
            "What is the RSI for {symbol} on {date}?",
            "Give me technical analysis for {symbol} on {date}",
            "What is {symbol} trading at on {date}?",
            "Show me {symbol} indicators on {date}",
        ]

        # Generate queries for each week
        while current_date <= end_date:
            for symbol in symbols:
                for template in query_templates[:2]:  # Use first 2 templates to keep manageable
                    query = template.format(symbol=symbol, date=current_date.strftime("%Y-%m-%d"))
                    queries.append({
                        'query': query,
                        'symbol': symbol,
                        'date': current_date,
                        'query_type': self._classify_query_type(query)
                    })

            current_date += timedelta(days=7)  # Weekly queries

        return queries

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for targeted evaluation"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['price', 'trading', 'cost']):
            return 'price_query'
        elif any(word in query_lower for word in ['rsi', 'indicator', 'technical']):
            return 'technical_query'
        elif any(word in query_lower for word in ['analysis', 'recommendation']):
            return 'analysis_query'
        else:
            return 'general_query'

    def evaluate_data_retrieval(self, query: str, symbol: str, query_date: datetime,
                              retrieved_data: Dict, ground_truth: Dict) -> Dict:
        """Evaluate how well the RAG system retrieved relevant data"""

        # Determine what data SHOULD have been retrieved based on query
        required_fields = set()
        query_lower = query.lower()

        if any(word in query_lower for word in ['price', 'trading', 'cost']):
            required_fields.update(['price', 'change', 'change_percent'])
        if any(word in query_lower for word in ['rsi']):
            required_fields.add('rsi')
        if any(word in query_lower for word in ['macd']):
            required_fields.update(['macd', 'macd_signal'])
        if any(word in query_lower for word in ['technical', 'indicator', 'analysis']):
            required_fields.update(['rsi', 'macd', 'price'])
        if any(word in query_lower for word in ['volume']):
            required_fields.add('volume')

        # Check what was actually retrieved
        retrieved_fields = set()

        # Check if price data was retrieved
        if 'current_data' in retrieved_data and 'error' not in retrieved_data.get('current_data', {}):
            current_data = retrieved_data['current_data']
            if 'current_price' in current_data:
                retrieved_fields.add('price')
            if 'change' in current_data:
                retrieved_fields.add('change')
            if 'change_percent' in current_data:
                retrieved_fields.add('change_percent')
            if 'volume' in current_data:
                retrieved_fields.add('volume')

        # Check if technical indicators were retrieved
        if 'historical_data' in retrieved_data and 'technical_indicators' in retrieved_data.get('historical_data', {}):
            indicators = retrieved_data['historical_data']['technical_indicators']
            if indicators.get('rsi') is not None:
                retrieved_fields.add('rsi')
            if indicators.get('macd') is not None:
                retrieved_fields.add('macd')
            if indicators.get('macd_signal') is not None:
                retrieved_fields.add('macd_signal')

        # Calculate retrieval metrics
        if required_fields:
            precision = len(retrieved_fields & required_fields) / len(retrieved_fields) if retrieved_fields else 0
            recall = len(retrieved_fields & required_fields) / len(required_fields)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 1.0  # Perfect if no specific requirements

        return {
            'required_fields': list(required_fields),
            'retrieved_fields': list(retrieved_fields),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'completeness': len(retrieved_fields & required_fields) / max(len(required_fields), 1)
        }

    def evaluate_factual_accuracy(self, retrieved_data: Dict, ground_truth: Dict) -> Dict:
        """Evaluate factual accuracy of retrieved data against ground truth"""
        accuracy_scores = {}
        total_checks = 0
        correct_checks = 0

        # Check price accuracy
        if ('current_data' in retrieved_data and
            'current_price' in retrieved_data.get('current_data', {}) and
            'price' in ground_truth):

            retrieved_price = retrieved_data['current_data']['current_price']
            actual_price = ground_truth['price']

            # Allow small tolerance for timing differences
            price_error = abs(retrieved_price - actual_price) / actual_price
            price_accurate = price_error < 0.05  # 5% tolerance

            accuracy_scores['price_accuracy'] = 1.0 if price_accurate else 0.0
            total_checks += 1
            if price_accurate:
                correct_checks += 1

        # Check RSI accuracy
        if ('historical_data' in retrieved_data and
            'technical_indicators' in retrieved_data.get('historical_data', {}) and
            retrieved_data['historical_data']['technical_indicators'].get('rsi') is not None and
            ground_truth.get('rsi') is not None):

            retrieved_rsi = retrieved_data['historical_data']['technical_indicators']['rsi']
            actual_rsi = ground_truth['rsi']

            rsi_error = abs(retrieved_rsi - actual_rsi)
            rsi_accurate = rsi_error < 5.0  # 5 point tolerance for RSI

            accuracy_scores['rsi_accuracy'] = 1.0 if rsi_accurate else 0.0
            total_checks += 1
            if rsi_accurate:
                correct_checks += 1

        # Check MACD accuracy
        if ('historical_data' in retrieved_data and
            'technical_indicators' in retrieved_data.get('historical_data', {}) and
            retrieved_data['historical_data']['technical_indicators'].get('macd') is not None and
            ground_truth.get('macd') is not None):

            retrieved_macd = retrieved_data['historical_data']['technical_indicators']['macd']
            actual_macd = ground_truth['macd']

            # MACD is relative, so use percentage error
            if actual_macd != 0:
                macd_error = abs(retrieved_macd - actual_macd) / abs(actual_macd)
                macd_accurate = macd_error < 0.1  # 10% tolerance
            else:
                macd_accurate = abs(retrieved_macd) < 0.001

            accuracy_scores['macd_accuracy'] = 1.0 if macd_accurate else 0.0
            total_checks += 1
            if macd_accurate:
                correct_checks += 1

        overall_accuracy = correct_checks / total_checks if total_checks > 0 else 0.0

        return {
            'overall_accuracy': overall_accuracy,
            'individual_scores': accuracy_scores,
            'total_checks': total_checks,
            'correct_checks': correct_checks
        }

    def run_historical_rag_evaluation(self, symbols: List[str],
                                    start_date: datetime,
                                    end_date: datetime) -> Dict:
        """Run comprehensive RAG evaluation using historical data"""

        print(f"🔍 Running RAG Historical Evaluation")
        print(f"   Symbols: {symbols}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print("="*60)

        # Generate historical queries
        queries = self.simulate_rag_queries(symbols, start_date, end_date)
        print(f"📝 Generated {len(queries)} historical queries")

        # Process each query
        evaluations = []

        for i, query_info in enumerate(queries[:50]):  # Limit for demo
            try:
                query = query_info['query']
                symbol = query_info['symbol']
                query_date = query_info['date']

                print(f"[{i+1}/50] Processing: {query}")

                # Get ground truth for this date
                ground_truth = self.get_historical_ground_truth(symbol, query_date)
                if not ground_truth:
                    continue

                # Simulate RAG retrieval (what our system would have retrieved)
                retrieved_data = self.data_provider.get_comprehensive_analysis(symbol)

                # Evaluate retrieval quality
                retrieval_eval = self.evaluate_data_retrieval(query, symbol, query_date,
                                                            retrieved_data, ground_truth)

                # Evaluate factual accuracy
                accuracy_eval = self.evaluate_factual_accuracy(retrieved_data, ground_truth)

                evaluation = {
                    'query': query,
                    'symbol': symbol,
                    'date': query_date.isoformat(),
                    'query_type': query_info['query_type'],
                    'retrieval_evaluation': retrieval_eval,
                    'accuracy_evaluation': accuracy_eval,
                    'ground_truth': ground_truth
                }

                evaluations.append(evaluation)

            except Exception as e:
                print(f"Error processing query {i}: {e}")

        return self._compile_rag_results(evaluations)

    def _compile_rag_results(self, evaluations: List[Dict]) -> Dict:
        """Compile RAG evaluation results"""
        if not evaluations:
            return {"error": "No evaluations completed"}

        # Aggregate retrieval metrics
        retrieval_precisions = [e['retrieval_evaluation']['precision'] for e in evaluations]
        retrieval_recalls = [e['retrieval_evaluation']['recall'] for e in evaluations]
        retrieval_f1s = [e['retrieval_evaluation']['f1_score'] for e in evaluations]

        # Aggregate accuracy metrics
        accuracy_scores = [e['accuracy_evaluation']['overall_accuracy'] for e in evaluations]

        # By query type
        query_types = {}
        for eval_data in evaluations:
            query_type = eval_data['query_type']
            if query_type not in query_types:
                query_types[query_type] = {
                    'count': 0,
                    'avg_precision': 0,
                    'avg_recall': 0,
                    'avg_f1': 0,
                    'avg_accuracy': 0
                }

            query_types[query_type]['count'] += 1
            query_types[query_type]['avg_precision'] += eval_data['retrieval_evaluation']['precision']
            query_types[query_type]['avg_recall'] += eval_data['retrieval_evaluation']['recall']
            query_types[query_type]['avg_f1'] += eval_data['retrieval_evaluation']['f1_score']
            query_types[query_type]['avg_accuracy'] += eval_data['accuracy_evaluation']['overall_accuracy']

        # Calculate averages for query types
        for query_type in query_types:
            count = query_types[query_type]['count']
            query_types[query_type]['avg_precision'] /= count
            query_types[query_type]['avg_recall'] /= count
            query_types[query_type]['avg_f1'] /= count
            query_types[query_type]['avg_accuracy'] /= count

        return {
            'total_evaluations': len(evaluations),
            'rag_performance': {
                'data_retrieval': {
                    'precision': np.mean(retrieval_precisions),
                    'recall': np.mean(retrieval_recalls),
                    'f1_score': np.mean(retrieval_f1s)
                },
                'factual_accuracy': {
                    'overall_accuracy': np.mean(accuracy_scores)
                }
            },
            'by_query_type': query_types,
            'sample_evaluations': evaluations[:5]  # Show first 5 examples
        }

    def print_rag_evaluation_report(self, results: Dict):
        """Print formatted RAG evaluation report"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return

        rag_perf = results['rag_performance']
        retrieval = rag_perf['data_retrieval']
        accuracy = rag_perf['factual_accuracy']

        print(f"\n📊 RAG SYSTEM HISTORICAL EVALUATION")
        print("="*70)
        print(f"🎯 GROUND TRUTH: Historical Yahoo Finance data")
        print(f"🔢 Total Evaluations: {results['total_evaluations']}")
        print("="*70)

        print(f"\n🔍 DATA RETRIEVAL PERFORMANCE:")
        print(f"   Precision: {retrieval['precision']:.3f} (How relevant was retrieved data?)")
        print(f"   Recall: {retrieval['recall']:.3f} (Did we get all needed data?)")
        print(f"   F1-Score: {retrieval['f1_score']:.3f} (Overall retrieval quality)")

        print(f"\n✅ FACTUAL ACCURACY:")
        print(f"   Overall Accuracy: {accuracy['overall_accuracy']:.3f} (Were facts correct?)")

        print(f"\n📋 BY QUERY TYPE:")
        for query_type, stats in results['by_query_type'].items():
            print(f"   {query_type.upper()}: P={stats['avg_precision']:.3f}, R={stats['avg_recall']:.3f}, F1={stats['avg_f1']:.3f}")

        print(f"\n🔍 SAMPLE EVALUATIONS:")
        for sample in results['sample_evaluations']:
            ret_eval = sample['retrieval_evaluation']
            acc_eval = sample['accuracy_evaluation']
            print(f"   Query: {sample['query'][:50]}...")
            print(f"   → Retrieval F1: {ret_eval['f1_score']:.3f}, Accuracy: {acc_eval['overall_accuracy']:.3f}")

        print("="*70)

def demo_rag_evaluation():
    """Demonstrate RAG historical evaluation"""
    evaluator = RAGHistoricalEvaluator()

    # Evaluate RAG performance on historical data
    symbols = ["AAPL", "TSLA"]
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 1)

    results = evaluator.run_historical_rag_evaluation(symbols, start_date, end_date)
    evaluator.print_rag_evaluation_report(results)

    # Save results
    with open("rag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 RAG evaluation results saved to rag_evaluation_results.json")

if __name__ == "__main__":
    demo_rag_evaluation()