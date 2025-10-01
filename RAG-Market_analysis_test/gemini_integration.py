"""
Gemini-powered Stock RAG Chatbot
"""
import os
import google.generativeai as genai
from typing import Dict, List
from datetime import datetime
from data_sources import MultiMarketDataProvider
from config import Config
import re
import sys
sys.path.append('/Volumes/D/RAGZ-C')
sys.path.append('/Volumes/D/RAGZ-C/market')
from ml_evaluation import StockRAGEvaluator
from analysis_engine import MarketAnalysisEngine

class GeminiStockRAGChatbot:
    """Stock analysis chatbot powered by Google Gemini"""

    def __init__(self, finnhub_api_key: str = None, gemini_api_key: str = None):
        # Load API keys from config
        if not gemini_api_key:
            gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
            raise ValueError("Please set your GEMINI_API_KEY in the .env file")

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Try different model names until one works - using latest available models
        model_names = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash',
            'models/gemini-flash-latest',
            'models/gemini-pro-latest',
            'models/gemini-2.5-pro'
        ]

        self.model = None
        for model_name in model_names:
            try:
                print(f"[GEMINI] Trying model: {model_name}...", flush=True)
                self.model = genai.GenerativeModel(model_name)
                # Test with a simple generation (with timeout handling)
                print(f"[GEMINI] Testing connection...", flush=True)
                test_response = self.model.generate_content("Hello", request_options={"timeout": 10})
                if test_response and test_response.text:
                    print(f"[GEMINI] ✓ Successfully using model: {model_name}")
                    break
                else:
                    print(f"[GEMINI] Model {model_name} returned empty response")
            except Exception as e:
                print(f"[GEMINI] ✗ Model {model_name} failed: {str(e)[:100]}")
                continue

        if not self.model:
            raise ValueError("Could not initialize any Gemini model. Please check your API key and internet connection.")

        # Initialize multi-source data provider
        self.data_provider = MultiMarketDataProvider(finnhub_api_key)

        # Initialize ML-powered market analysis engine
        self.ml_engine = MarketAnalysisEngine(finnhub_api_key)

        # Chat history
        self.chat_history = []

        # Initialize evaluation system
        self.evaluator = StockRAGEvaluator()

        print("✅ Gemini-powered Stock RAG Chatbot initialized with ML evaluation and prediction models!")

    def extract_stock_symbols(self, user_input: str) -> List[str]:
        """Extract stock symbols from user input - improved accuracy"""
        symbols = []

        # Known stock symbols (much more precise)
        known_symbols = {
            'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA',
            'NFLX', 'SPOT', 'UBER', 'ABNB', 'COIN', 'SQ', 'PYPL', 'V', 'MA',
            'JPM', 'BAC', 'WMT', 'TGT', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'DIS'
        }

        # Look for known symbols in uppercase input
        words = user_input.upper().split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in known_symbols:
                symbols.append(clean_word)

        # Look for $ prefixed symbols
        dollar_matches = re.findall(r'\$([A-Z]{1,5})', user_input.upper())
        for match in dollar_matches:
            if match in known_symbols:
                symbols.append(match)

        # Company name to symbol mapping
        company_mapping = {
            'APPLE': 'AAPL', 'TESLA': 'TSLA', 'MICROSOFT': 'MSFT',
            'GOOGLE': 'GOOGL', 'AMAZON': 'AMZN', 'META': 'META',
            'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'SPOTIFY': 'SPOT'
        }

        # Check for company names
        for company, symbol in company_mapping.items():
            if company in user_input.upper():
                symbols.append(symbol)

        # Remove duplicates and return
        return list(set(symbols))

    def is_market_related_query(self, user_input: str) -> bool:
        """Check if user query is market/stock related"""
        market_keywords = [
            'stock', 'price', 'market', 'trading', 'buy', 'sell', 'invest',
            'portfolio', 'prediction', 'forecast', 'analysis', 'profit',
            'loss', 'rsi', 'macd', 'technical', 'bullish', 'bearish',
            'earnings', 'dividend', 'shares', 'volume', 'volatility'
        ]

        return any(keyword in user_input.lower() for keyword in market_keywords)

    def format_analysis_for_context(self, analysis: Dict) -> str:
        """Format analysis data as context for Gemini"""
        if "error" in analysis:
            return f"Error in analysis: {analysis['error']}"

        # Handle multi-source data format
        if "current_data" in analysis:
            formatted = f"REAL-TIME MARKET DATA FOR {analysis['symbol']}:\n"
            formatted += f"Data Sources: {', '.join(analysis.get('data_sources', []))}\n"
            formatted += f"Timestamp: {analysis['timestamp']}\n\n"

            # Current price data
            if "error" not in analysis.get("current_data", {}):
                current = analysis["current_data"]
                formatted += f"CURRENT PRICE: ${current['current_price']:.2f}\n"
                formatted += f"Change: ${current['change']:.2f} ({current['change_percent']:.2f}%)\n"
                formatted += f"Day Range: ${current['low']:.2f} - ${current['high']:.2f}\n"
                formatted += f"Volume: {current['volume']:,}\n\n"

            # Technical indicators from historical data
            if "error" not in analysis.get("historical_data", {}):
                historical = analysis["historical_data"]
                if "technical_indicators" in historical:
                    indicators = historical["technical_indicators"]
                    formatted += f"TECHNICAL INDICATORS:\n"

                    if indicators.get('rsi') is not None:
                        rsi_val = indicators['rsi']
                        rsi_signal = "OVERSOLD" if rsi_val < 30 else "OVERBOUGHT" if rsi_val > 70 else "NEUTRAL"
                        formatted += f"RSI (14): {rsi_val:.2f} ({rsi_signal})\n"

                    if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
                        macd_val = indicators['macd']
                        signal_val = indicators['macd_signal']
                        macd_signal = "BULLISH" if macd_val > signal_val else "BEARISH"
                        formatted += f"MACD: {macd_val:.4f}, Signal: {signal_val:.4f} ({macd_signal})\n"

                    if indicators.get('sma_20') is not None:
                        sma20 = indicators['sma_20']
                        current_price = analysis.get("current_data", {}).get("current_price", 0)
                        trend = "ABOVE" if current_price > sma20 else "BELOW"
                        formatted += f"SMA 20: ${sma20:.2f} (Price is {trend} SMA)\n"

                    if indicators.get('sma_50') is not None:
                        sma50 = indicators['sma_50']
                        formatted += f"SMA 50: ${sma50:.2f}\n"

                    if indicators.get('bb_upper') is not None and indicators.get('bb_lower') is not None:
                        bb_upper = indicators['bb_upper']
                        bb_lower = indicators['bb_lower']
                        current_price = analysis.get("current_data", {}).get("current_price", 0)
                        bb_position = "NEAR UPPER" if current_price > (bb_upper * 0.98) else "NEAR LOWER" if current_price < (bb_lower * 1.02) else "MIDDLE"
                        formatted += f"Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f} ({bb_position})\n"

                    formatted += "\n"

            # Company info
            if "error" not in analysis.get("company_info", {}):
                company = analysis["company_info"]
                formatted += f"COMPANY: {company['company_name']}\n"
                formatted += f"Sector: {company['sector']}\n"
                formatted += f"Industry: {company['industry']}\n"
                if company['market_cap'] > 0:
                    formatted += f"Market Cap: ${company['market_cap']:,}\n"

            return formatted

        # Handle ML engine format (from MarketAnalysisEngine)
        elif "analysis" in analysis:
            return self.ml_engine.format_analysis_for_llm(analysis)

        return "No market data available"

    def extract_structured_data(self, analysis: Dict, symbol: str) -> Dict:
        """Extract structured data for RAG evaluation"""
        structured_data = {
            'symbol': symbol,
            'extraction_success': True,
            'retrieved_fields': [],
            'data': {}
        }

        try:
            # Extract current price data
            if "current_data" in analysis and "error" not in analysis.get("current_data", {}):
                current_data = analysis["current_data"]
                if "current_price" in current_data:
                    structured_data['data']['price'] = float(current_data["current_price"])
                    structured_data['retrieved_fields'].append('price')

                if "change" in current_data:
                    structured_data['data']['change'] = float(current_data["change"])
                    structured_data['retrieved_fields'].append('change')

                if "change_percent" in current_data:
                    structured_data['data']['change_percent'] = float(current_data["change_percent"])
                    structured_data['retrieved_fields'].append('change_percent')

                if "volume" in current_data:
                    structured_data['data']['volume'] = int(current_data["volume"])
                    structured_data['retrieved_fields'].append('volume')

            # Extract technical indicators
            if "historical_data" in analysis and "technical_indicators" in analysis.get("historical_data", {}):
                indicators = analysis["historical_data"]["technical_indicators"]

                if "rsi" in indicators and indicators["rsi"] is not None:
                    structured_data['data']['rsi'] = float(indicators["rsi"])
                    structured_data['retrieved_fields'].append('rsi')

                if "macd" in indicators and indicators["macd"] is not None:
                    structured_data['data']['macd'] = float(indicators["macd"])
                    structured_data['retrieved_fields'].append('macd')

                if "macd_signal" in indicators and indicators["macd_signal"] is not None:
                    structured_data['data']['macd_signal'] = float(indicators["macd_signal"])
                    structured_data['retrieved_fields'].append('macd_signal')

        except Exception as e:
            structured_data['extraction_success'] = False
            structured_data['error'] = str(e)

        return structured_data

    async def get_structured_response(self, user_input: str) -> Dict:
        """Get structured data response instead of text (for RAG evaluation)"""
        symbols = self.extract_stock_symbols(user_input)

        if not symbols:
            return {'error': 'No stock symbols found in query'}

        results = {}
        for symbol in symbols[:1]:  # Limit to 1 symbol for testing
            analysis = self.data_provider.get_comprehensive_analysis(symbol)
            structured_data = self.extract_structured_data(analysis, symbol)
            results[symbol] = structured_data

        return results

    async def chat(self, user_input: str) -> str:
        """Main chat function using Gemini API"""
        print(f"[GEMINI] Processing: {user_input}")

        # Store user input in chat history
        self.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })

        try:
            # Check if query is market-related
            if self.is_market_related_query(user_input):
                # Extract stock symbols
                symbols = self.extract_stock_symbols(user_input)

                if symbols:
                    print(f"[GEMINI] Analyzing stocks: {symbols}")

                    # Check if prediction is requested
                    prediction_keywords = ['predict', 'forecast', 'future', 'projection', 'will', 'gonna', 'going to']
                    needs_prediction = any(keyword in user_input.lower() for keyword in prediction_keywords)

                    # Get market analysis for each symbol
                    market_contexts = []
                    analyses = {}
                    for symbol in symbols[:3]:  # Limit to 3 symbols
                        print(f"[GEMINI] Getting data for {symbol}...")
                        analysis = self.data_provider.get_comprehensive_analysis(symbol)
                        analyses[symbol] = analysis
                        context = self.format_analysis_for_context(analysis)
                        market_contexts.append(f"=== {symbol} ANALYSIS ===\n{context}")

                        # Evaluate technical signals
                        if 'technical_indicators' in analysis.get('historical_data', {}):
                            self.evaluate_technical_signals(symbol, analysis['historical_data']['technical_indicators'])

                        # Add ML predictions if requested
                        if needs_prediction:
                            print(f"[ML] Training prediction models for {symbol}...")
                            ml_analysis = self.ml_engine.comprehensive_analysis(symbol, prediction_days=30)
                            if "error" not in ml_analysis:
                                ml_context = self.format_analysis_for_context(ml_analysis)
                                market_contexts.append(f"=== {symbol} ML PREDICTIONS (30 DAYS) ===\n{ml_context}")
                                print(f"[ML] ✓ Predictions generated for {symbol}")

                    # Evaluate retrieval quality
                    self.evaluate_retrieval_quality(user_input, symbols, analyses)

                    # Combine contexts
                    combined_context = "\n\n".join(market_contexts)

                    # Create enhanced prompt for Gemini with technical analysis focus
                    ml_instruction = ""
                    if needs_prediction:
                        ml_instruction = """
- ML PREDICTIONS: I've trained 3 machine learning models (Linear Regression, Random Forest, Gradient Boosting)
- Analyze the portfolio projections showing potential profit/loss over 30 days
- Compare the ensemble prediction with individual model predictions
- Discuss model performance metrics (MAE, RMSE) to assess prediction reliability"""

                    prompt = f"""You are a professional financial advisor AI with access to real-time market data, technical indicators, ML predictions, and web search.

CURRENT MARKET DATA WITH TECHNICAL ANALYSIS:
{combined_context}

USER QUESTION: {user_input}

Instructions:
- Use the market data and technical indicators provided above as your primary source
- Analyze RSI levels: <30 = oversold (potential buy), >70 = overbought (potential sell)
- Interpret MACD signals: MACD above signal line = bullish momentum, below = bearish momentum
- Consider price position relative to moving averages and Bollinger Bands{ml_instruction}
- You can search the web for latest news, earnings, or analyst reports for additional context
- Include specific current prices, technical indicator values, and their interpretations
- Provide buy/sell/hold recommendations based on technical analysis
- Mention any recent news or developments if relevant
- Be professional and accurate in your technical analysis

Please provide your comprehensive technical analysis:"""

                    print(f"[GEMINI] Sending request to Gemini API...")

                else:
                    # Market-related but no specific symbols - use web search
                    prompt = f"""You are a financial advisor AI with web search access. The user asked: "{user_input}"

This appears to be a general market question. Please:
1. Search the web for current market information related to this question
2. Provide helpful, up-to-date financial advice
3. Include relevant recent market developments or news
4. Suggest specific stock symbols if appropriate for detailed analysis

Please provide your analysis:"""

            else:
                # Non-market related query
                prompt = f"User question: {user_input}\n\nPlease provide a helpful response."

            # Send to Gemini
            response = self.model.generate_content(prompt)

            # Extract text from response
            if response and response.text:
                gemini_response = response.text.strip()
                print(f"[GEMINI] Response received: {len(gemini_response)} characters")
            else:
                gemini_response = "I apologize, but I couldn't generate a response. Please try again."

            # Store response in chat history
            self.chat_history.append({
                "role": "assistant",
                "content": gemini_response,
                "timestamp": datetime.now()
            })

            return gemini_response

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"[GEMINI] Error: {e}")
            return error_msg

    def get_chat_history(self):
        """Get formatted chat history"""
        return self.chat_history

    def evaluate_retrieval_quality(self, query: str, symbols: List[str], analysis_data: Dict) -> str:
        """Evaluate and log retrieval quality"""
        # Determine what data should have been retrieved based on query
        query_lower = query.lower()

        required_data = []
        if any(word in query_lower for word in ['price', 'cost', 'trading']):
            required_data.extend([f"{s}_price" for s in symbols])
        if any(word in query_lower for word in ['rsi', 'technical', 'indicator']):
            required_data.extend([f"{s}_rsi" for s in symbols])
        if any(word in query_lower for word in ['macd']):
            required_data.extend([f"{s}_macd" for s in symbols])
        if any(word in query_lower for word in ['news', 'latest']):
            required_data.extend([f"{s}_news" for s in symbols])

        # What we actually retrieved
        retrieved_data = []
        for symbol in symbols:
            if 'current_data' in analysis_data and 'error' not in analysis_data['current_data']:
                retrieved_data.append(f"{symbol}_price")
            if 'technical_indicators' in analysis_data.get('historical_data', {}):
                retrieved_data.extend([f"{symbol}_rsi", f"{symbol}_macd"])

        # Determine response quality based on completeness
        if len(retrieved_data) >= len(required_data):
            quality = "EXCELLENT"
        elif len(retrieved_data) >= len(required_data) * 0.7:
            quality = "GOOD"
        else:
            quality = "POOR"

        # Log for evaluation
        self.evaluator.add_retrieval_result(query, retrieved_data, required_data, quality)

        return quality

    def evaluate_technical_signals(self, symbol: str, indicators: Dict):
        """Evaluate and log technical signals"""
        if not indicators:
            return

        # RSI signals
        if 'rsi' in indicators and indicators['rsi'] is not None:
            rsi_val = indicators['rsi']
            if rsi_val < 30:
                # We'll need to track actual movement later for evaluation
                self.evaluator.add_signal(symbol, "RSI_OVERSOLD", "STRONG", "PENDING", rsi_val, "BULLISH")
            elif rsi_val > 70:
                self.evaluator.add_signal(symbol, "RSI_OVERBOUGHT", "STRONG", "PENDING", rsi_val, "BEARISH")

        # MACD signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            if indicators['macd'] > indicators['macd_signal']:
                self.evaluator.add_signal(symbol, "MACD_BULLISH", "MODERATE", "PENDING",
                                        indicators.get('rsi', 50), "BULLISH")
            else:
                self.evaluator.add_signal(symbol, "MACD_BEARISH", "MODERATE", "PENDING",
                                        indicators.get('rsi', 50), "BEARISH")

    def get_evaluation_report(self):
        """Get current evaluation metrics"""
        return {
            "predictions": self.evaluator.evaluate_predictions(),
            "signals": self.evaluator.evaluate_signals(),
            "retrieval": self.evaluator.evaluate_retrieval()
        }

    def print_evaluation_report(self):
        """Print comprehensive evaluation report"""
        self.evaluator.print_comprehensive_report()

    def export_evaluation_results(self, filename: str = None):
        """Export evaluation results"""
        if filename is None:
            filename = f"chatbot_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.evaluator.export_results(filename)
        return filename

# Test function
async def test_gemini_chatbot():
    """Test the Gemini chatbot"""
    try:
        # Initialize chatbot
        chatbot = GeminiStockRAGChatbot()

        # Test queries
        queries = [
            "Hello, how are you?",
            "What is AAPL stock price?",
            "Should I buy Tesla stock?"
        ]

        for query in queries:
            print(f"\n🤖 User: {query}")
            response = await chatbot.chat(query)
            print(f"🎯 Gemini: {response[:200]}...")

    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gemini_chatbot())