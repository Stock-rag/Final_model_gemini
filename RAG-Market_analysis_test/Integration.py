import asyncio
from typing import Dict, List, Optional
import json
from datetime import datetime
import re
import sys
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import your market analysis engine and new data sources
from ML_e import AdvancedMarketAnalysisEngine
from data_sources import MultiMarketDataProvider

class StockRAGChatbot:
    def __init__(self, finnhub_api_key: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        # Initialize market analysis engine
        self.market_engine = AdvancedMarketAnalysisEngine(finnhub_api_key)
        # Initialize multi-source data provider
        self.data_provider = MultiMarketDataProvider(finnhub_api_key)

        # Initialize Llama model
        self.model_name = model_name
        print(f"Loading Llama model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if we're on Mac (where bitsandbytes may not work properly)
        is_mac = platform.system() == "Darwin"

        if is_mac:
            print("Mac detected - checking for MPS (GPU) support")
            if torch.backends.mps.is_available():
                print("🚀 MPS (Mac GPU) available! Loading model on GPU...")
                # Load model on MPS device for M4 Pro GPU acceleration
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to("mps")
                self.device = "mps"
            else:
                print("MPS not available, loading on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.device = "cpu"
        else:
            # Use quantization on Linux/Windows
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception as e:
                print(f"Quantization failed, loading without: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )

        print("Llama model loaded successfully")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Knowledge base for storing market analysis
        self.knowledge_base = {}
        
        # Chat history
        self.chat_history = []
    
    def extract_stock_symbols(self, user_input: str) -> List[str]:
        """Extract stock symbols from user input"""
        # Common stock symbol patterns
        patterns = [
            r'\b([A-Z]{1,5})\b',  # 1-5 uppercase letters
            r'\$([A-Z]{1,5})',    # Dollar sign prefix
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, user_input.upper())
            symbols.extend(matches)
        
        # Common stock names to symbols mapping
        stock_mapping = {
            'APPLE': 'AAPL', 'TESLA': 'TSLA', 'MICROSOFT': 'MSFT',
            'GOOGLE': 'GOOGL', 'AMAZON': 'AMZN', 'META': 'META',
            'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'SPOTIFY': 'SPOT'
        }
        
        # Check for company names in input
        for name, symbol in stock_mapping.items():
            if name in user_input.upper():
                symbols.append(symbol)
        
        # Remove duplicates and filter valid symbols
        valid_symbols = []
        for symbol in set(symbols):
            if len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalpha():
                valid_symbols.append(symbol)
        
        return valid_symbols
    
    def is_market_related_query(self, user_input: str) -> bool:
        """Check if user query is market/stock related"""
        market_keywords = [
            'stock', 'price', 'market', 'trading', 'buy', 'sell', 'invest',
            'portfolio', 'prediction', 'forecast', 'analysis', 'profit',
            'loss', 'rsi', 'macd', 'technical', 'bullish', 'bearish',
            'earnings', 'dividend', 'shares', 'volume', 'volatility'
        ]
        
        # Check if any market keyword exists in input
        return any(keyword in user_input.lower() for keyword in market_keywords)
    
    async def get_market_analysis(self, symbol: str, use_cache: bool = True) -> Dict:
        """Get market analysis for a symbol with caching - now with multiple data sources"""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"

        # Check cache first (daily cache)
        if use_cache and cache_key in self.knowledge_base:
            return self.knowledge_base[cache_key]

        # Get fresh analysis using new multi-source provider
        try:
            print(f"[ANALYSIS] Getting comprehensive analysis for {symbol}...")
            analysis = self.data_provider.get_comprehensive_analysis(symbol)

            # Cache the result
            self.knowledge_base[cache_key] = analysis

            return analysis
        except Exception as e:
            # Fallback to old engine if new one fails
            try:
                print(f"[ANALYSIS] Falling back to original engine for {symbol}...")
                analysis = self.market_engine.comprehensive_analysis_with_prediction(symbol, prediction_days=30)
                self.knowledge_base[cache_key] = analysis
                return analysis
            except Exception as e2:
                return {"error": f"Failed to get analysis for {symbol}: {str(e2)}"}
    
    def format_analysis_for_context(self, analysis: Dict) -> str:
        """Format analysis data as context for LLM"""
        if "error" in analysis:
            return f"Error in analysis: {analysis['error']}"

        # Handle new multi-source data format
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

            # Company info
            if "error" not in analysis.get("company_info", {}):
                company = analysis["company_info"]
                formatted += f"COMPANY: {company['company_name']}\n"
                formatted += f"Sector: {company['sector']}\n"
                formatted += f"Industry: {company['industry']}\n"
                if company['market_cap'] > 0:
                    formatted += f"Market Cap: ${company['market_cap']:,}\n"

            return formatted
        else:
            # Fallback to old format
            return self.market_engine.format_advanced_analysis_for_llm(analysis)
    
    def create_rag_prompt(self, user_query: str, market_context: str) -> str:
        """Create RAG prompt combining user query with market context - optimized for Llama-3.2"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional financial advisor AI with access to real-time market data. You must ONLY use the current market data provided below. DO NOT make up prices, dates, or company information. If specific data is not provided, say so explicitly.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CURRENT MARKET DATA:
{market_context}

QUESTION: {user_query}

Instructions:
- Use ONLY the market data provided above
- Include specific current prices and percentages from the data
- Be professional and accurate
- If data is missing, explicitly state "data not available"
- Do not reference outdated information (no iPhone 6s, old stock prices, etc.)
- Provide current, accurate financial analysis based solely on the data provided

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response using Llama model"""
        try:
            print(f"[DEBUG] Starting response generation...")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")

            # Tokenize input with proper truncation
            print(f"[DEBUG] Tokenizing input...")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            print(f"[DEBUG] Input tokens shape: {inputs['input_ids'].shape}")

            # Move to same device as model
            if hasattr(self, 'device'):
                device = self.device
            elif torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            print(f"[DEBUG] Using device: {device}")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response with device-optimized parameters
            print(f"[DEBUG] Starting model generation...")
            with torch.no_grad():
                if device == "mps":
                    # GPU parameters - can handle more tokens
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    # CPU parameters - reduced for speed
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_new_tokens, 30),
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            print(f"[DEBUG] Model generation completed!")

            # Decode only the new tokens (exclude input tokens)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"[DEBUG] Response generated: {len(response)} characters")

            return response

        except Exception as e:
            print(f"[DEBUG] Error in generate_response: {str(e)}")
            return f"Sorry, I encountered an error generating the response: {str(e)}"
    
    async def chat(self, user_input: str) -> str:
        """Main chat function that handles user input"""
        # Store user input in chat history
        self.chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
        
        # Check if query is market-related
        if self.is_market_related_query(user_input):
            # Extract stock symbols from user input
            symbols = self.extract_stock_symbols(user_input)
            
            if symbols:
                # Get market analysis for extracted symbols
                market_contexts = []
                
                for symbol in symbols[:3]:  # Limit to 3 symbols to avoid token overflow
                    print(f"Analyzing {symbol}...")
                    analysis = await self.get_market_analysis(symbol)
                    context = self.format_analysis_for_context(analysis)
                    market_contexts.append(f"=== {symbol} ANALYSIS ===\n{context}")
                
                # Combine all market contexts
                combined_context = "\n\n".join(market_contexts)
                
                # Create RAG prompt
                rag_prompt = self.create_rag_prompt(user_input, combined_context)

                # Debug: Show what market context is being provided
                print(f"[DEBUG] Market context length: {len(combined_context)} characters")
                print(f"[DEBUG] First 200 chars of context: {combined_context[:200]}...")

                # Generate response with market context
                response = self.generate_response(rag_prompt)
                
            else:
                # Market-related but no specific symbols detected
                general_prompt = f"""You are a financial advisor AI. The user asked: "{user_input}"

This appears to be a general market question. Provide helpful financial advice and suggest they specify stock symbols for detailed analysis.

RESPONSE:"""
                response = self.generate_response(general_prompt)
        
        else:
            # Non-market related query - use general conversation
            general_prompt = f"User: {user_input}\nAssistant:"
            response = self.generate_response(general_prompt, max_new_tokens=150)
        
        # Store assistant response in chat history
        self.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
        
        return response
    
    def get_portfolio_analysis(self, portfolio: List[str], investment_amount: float = 10000) -> str:
        """Analyze entire portfolio"""
        portfolio_context = f"PORTFOLIO ANALYSIS (${investment_amount:,.2f} investment):\n\n"
        
        for symbol in portfolio:
            try:
                # Use synchronous call for portfolio analysis
                import asyncio
                analysis = asyncio.run(self.get_market_analysis(symbol))
                context = self.format_analysis_for_context(analysis)
                portfolio_context += f"{symbol}:\n{context}\n\n"
            except Exception as e:
                portfolio_context += f"{symbol}: Analysis failed - {str(e)}\n\n"
        
        return portfolio_context
    
    def save_chat_history(self, filename: str = "chat_history.json"):
        """Save chat history to file"""
        # Convert datetime objects to strings for JSON serialization
        serializable_history = []
        for entry in self.chat_history:
            serializable_entry = entry.copy()
            serializable_entry["timestamp"] = entry["timestamp"].isoformat()
            serializable_history.append(serializable_entry)
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(serializable_history, f, indent=2)
    
    def clear_cache(self):
        """Clear the knowledge base cache"""
        self.knowledge_base.clear()
        print("Market analysis cache cleared.")

# Example usage and testing
async def main():
    # Your Finnhub API key
    API_KEY = "d31nrlhr01qsprr1r7i0d31nrlhr01qsprr1r7ig"
    
    # Initialize RAG chatbot
    chatbot = StockRAGChatbot(API_KEY)
    
    print("Stock RAG Chatbot initialized! Type 'quit' to exit.\n")
    
    # Example conversations
    example_queries = [
        "What's the current price and analysis for AAPL?",
        "Should I buy Tesla stock right now?",
        "Give me a technical analysis of MSFT with RSI and MACD",
        "What will happen to my $10,000 NVDA investment in 30 days?",
        "Compare GOOGL and META for investment",
        "What are the trading signals for Apple stock?"
    ]
    
    print("Example queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "="*50)
    
    # Interactive chat loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Assistant: Analyzing...")
        
        try:
            # Get chatbot response
            response = await chatbot.chat(user_input)
            print(f"Assistant: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Save chat history
    chatbot.save_chat_history()
    print("Chat history saved to chat_history.json")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())