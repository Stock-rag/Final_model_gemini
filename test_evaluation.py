"""
Test the ML evaluation system with the Gemini Stock RAG chatbot
"""
import sys
import asyncio
sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')

from gemini_integration import GeminiStockRAGChatbot

async def test_evaluation_system():
    """Test the evaluation system with various queries"""
    try:
        print("🚀 Initializing Gemini Stock RAG Chatbot with Evaluation...")
        chatbot = GeminiStockRAGChatbot()

        print("\n✅ Chatbot with ML Evaluation initialized!")

        # Test queries that will generate evaluation data
        test_queries = [
            "What is AAPL stock price today?",
            "What are the RSI and MACD indicators for Apple?",
            "Should I buy Tesla stock based on technical analysis?",
            "Give me news about Microsoft stock",
            "What's the current price of NVDA?"
        ]

        print(f"\n🧪 Running {len(test_queries)} test queries for evaluation...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] 👤 User: {query}")
            try:
                response = await chatbot.chat(query)
                print(f"🤖 Response length: {len(response)} characters")
                print(f"🤖 First 150 chars: {response[:150]}...")

                # Simulate user feedback (in real app, this would come from user)
                import random
                rating = random.randint(3, 5)  # Simulate mostly positive feedback
                print(f"📊 Simulated user rating: {rating}/5")

                print("-" * 60)

            except Exception as e:
                print(f"❌ Error processing query: {e}")

        # Generate and display evaluation report
        print(f"\n📊 GENERATING ML EVALUATION REPORT...")
        print("="*80)

        chatbot.print_evaluation_report()

        # Export results
        filename = chatbot.export_evaluation_results()
        print(f"\n✅ Evaluation results exported to: {filename}")

        # Show specific metrics
        report = chatbot.get_evaluation_report()

        if "error" not in report["retrieval"]:
            retrieval = report["retrieval"]["average_metrics"]
            print(f"\n🎯 KEY METRICS SUMMARY:")
            print(f"   Retrieval Precision: {retrieval['precision']:.3f}")
            print(f"   Retrieval Recall: {retrieval['recall']:.3f}")
            print(f"   Retrieval F1-Score: {retrieval['f1_score']:.3f}")

        return chatbot

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def demo_manual_evaluation():
    """Demo adding manual evaluation data"""
    print("\n🧪 DEMO: Manual Evaluation Data Entry")
    print("="*50)

    # Import the evaluator directly
    from ml_evaluation import StockRAGEvaluator

    evaluator = StockRAGEvaluator()

    # Add sample data
    from datetime import datetime, timedelta

    print("📊 Adding sample predictions...")

    # Sample predictions with outcomes
    predictions = [
        ("AAPL", "BUY", "PROFIT", 0.85),
        ("TSLA", "SELL", "PROFIT", 0.90),
        ("MSFT", "HOLD", "NEUTRAL", 0.60),
        ("NVDA", "BUY", "LOSS", 0.75),    # Wrong prediction
        ("GOOGL", "BUY", "PROFIT", 0.80),
    ]

    base_date = datetime.now() - timedelta(days=7)

    for i, (symbol, action, outcome, confidence) in enumerate(predictions):
        evaluator.add_prediction(
            symbol, action, outcome, confidence,
            base_date + timedelta(days=i),
            base_date + timedelta(days=i+3)
        )
        print(f"   {symbol}: {action} → {outcome} (confidence: {confidence})")

    print(f"\n📊 Adding sample technical signals...")

    # Sample technical signals
    signals = [
        ("AAPL", "RSI_OVERSOLD", "STRONG", "UP", 28.5, "BULLISH"),
        ("TSLA", "RSI_OVERBOUGHT", "MODERATE", "DOWN", 72.1, "BEARISH"),
        ("MSFT", "MACD_BULLISH", "STRONG", "UP", 45.0, "BULLISH"),
        ("NVDA", "MACD_BEARISH", "WEAK", "UP", 55.0, "BEARISH"),  # Wrong signal
    ]

    for signal_data in signals:
        evaluator.add_signal(*signal_data)
        symbol, signal_type, strength, actual, rsi, macd = signal_data
        print(f"   {symbol}: {signal_type} → {actual} (RSI: {rsi})")

    print(f"\n📊 Computing evaluation metrics...")

    # Generate report
    evaluator.print_comprehensive_report()

    return evaluator

if __name__ == "__main__":
    print("🎯 Stock RAG ML Evaluation System Test")
    print("="*60)

    # Choose which test to run
    print("\nSelect test mode:")
    print("1. Full chatbot evaluation test")
    print("2. Manual evaluation demo")
    print("3. Both")

    choice = input("Enter choice (1-3): ").strip()

    if choice in ["1", "3"]:
        print(f"\n🚀 Running chatbot evaluation test...")
        asyncio.run(test_evaluation_system())

    if choice in ["2", "3"]:
        print(f"\n🧪 Running manual evaluation demo...")
        demo_manual_evaluation()

    print(f"\n✅ Evaluation tests completed!")