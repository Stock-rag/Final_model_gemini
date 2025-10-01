"""
Test script for Gemini-powered Stock RAG Chatbot
"""
import sys
import os
sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')

async def test_gemini():
    """Test the Gemini integration"""
    try:
        from gemini_integration import GeminiStockRAGChatbot

        print("🚀 Initializing Gemini Stock RAG Chatbot...")
        chatbot = GeminiStockRAGChatbot()

        print("\n✅ Chatbot initialized successfully!")

        # Test queries including technical analysis and ML predictions
        test_queries = [
            "Hello, how are you?",
            "What is TSLA stock price today?",
            "What are the RSI and MACD indicators showing for NVDA stock?",
            "Should I buy NVDA based on technical analysis?",
            "Predict the future price of AAPL for the next 30 days using ML models"
        ]

        for query in test_queries:
            print(f"\n👤 User: {query}")
            response = await chatbot.chat(query)
            print(f"🤖 Gemini: {response}")
            print("-" * 80)

        print("\n🎉 Test completed successfully!")

        # Print evaluation report showing Precision, Recall, F1-Score
        print("\n" + "="*80)
        print("📊 RETRIEVAL PERFORMANCE METRICS")
        print("="*80)

        # Get retrieval evaluation
        retrieval_eval = chatbot.evaluator.evaluate_retrieval()

        if "error" not in retrieval_eval:
            metrics = retrieval_eval["average_metrics"]
            print(f"\n✅ RAG RETRIEVAL PERFORMANCE:")
            print(f"   Total Queries Processed: {retrieval_eval['total_queries']}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall: {metrics['recall']:.3f}")
            print(f"   F1-Score: {metrics['f1_score']:.3f}")
            print(f"   Perfect Retrievals: {retrieval_eval['perfect_retrievals']}/{retrieval_eval['total_queries']}")

            # Show breakdown by quality
            if retrieval_eval.get('by_quality'):
                print(f"\n📈 BREAKDOWN BY RESPONSE QUALITY:")
                for quality, stats in retrieval_eval['by_quality'].items():
                    print(f"   {quality}: {stats['count']} queries | "
                          f"Precision={stats['avg_precision']:.3f}, "
                          f"Recall={stats['avg_recall']:.3f}")
        else:
            print(f"\n⚠️ {retrieval_eval['error']}")

        print("="*80)

        # Export full evaluation results
        export_file = chatbot.export_evaluation_results()
        print(f"\n📁 Full evaluation report exported to: {export_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        if "GEMINI_API_KEY" in str(e):
            print("\n📝 Please:")
            print("1. Go to https://aistudio.google.com/")
            print("2. Get your free API key")
            print("3. Add it to .env file: GEMINI_API_KEY=your_actual_key")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gemini())