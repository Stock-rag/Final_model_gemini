"""
Updated LLM implementations using the unified factory
"""
from utils.llm_factory import LLMFactory
from config import Config

def simple_concept_explanation():
    """Simple LLM example using factory"""
    print("=== Simple Concept Explanation ===")

    # Get LLM instance from factory
    llm = LLMFactory.get_llm()

    # Create chain with template
    chain = llm.create_chain(
        template="Explain the concept of {concept} in one simple sentence.",
        input_variables=["concept"]
    )

    # Run the chain
    concept = "Large Language Model"
    result = chain.invoke({"concept": concept})
    print(f"Concept: {concept}")
    print(f"Explanation: {result}")

    return result

def lightweight_model_example():
    """Example using a smaller model for faster inference"""
    print("\n=== Lightweight Model Example ===")

    # Use a smaller model for this example
    llm = LLMFactory.get_llm(model_name="google/flan-t5-base", use_quantization=False)

    # Simple generation
    prompt = "Explain machine learning in simple terms:"
    response = llm.generate(prompt, max_length=100)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    return response

def conversational_example():
    """Example of conversational AI"""
    print("\n=== Conversational Example ===")

    llm = LLMFactory.get_llm()

    # Create a conversational chain
    chain = llm.create_chain(
        template="""You are a helpful AI assistant. Have a natural conversation with the user.

User: {user_input}
Assistant:""",
        input_variables=["user_input"]
    )

    # Example conversation
    user_inputs = [
        "Hello, how can you help me?",
        "What's the weather like?",
        "Tell me about artificial intelligence"
    ]

    for user_input in user_inputs:
        response = chain.invoke({"user_input": user_input})
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print()

def financial_analysis_example():
    """Example using LLM for financial analysis context"""
    print("\n=== Financial Analysis Example ===")

    llm = LLMFactory.get_llm()

    # Financial advisor template
    template = """You are a financial advisor AI. Based on the following market data, provide investment advice.

Market Data:
{market_data}

User Question: {question}

Provide a professional response with specific recommendations:"""

    chain = llm.create_chain(
        template=template,
        input_variables=["market_data", "question"]
    )

    # Example market data
    market_data = """
AAPL: Price $150.25, RSI: 65, MACD: Bullish, Moving Average: Above 20-day SMA
TSLA: Price $200.50, RSI: 75, MACD: Bearish, Moving Average: Below 20-day SMA
"""

    question = "Should I buy Apple or Tesla stock right now?"

    response = chain.invoke({
        "market_data": market_data,
        "question": question
    })

    print(f"Market Data: {market_data}")
    print(f"Question: {question}")
    print(f"AI Advisor: {response}")

    return response

if __name__ == "__main__":
    print("Running LLM Implementation Examples")
    print("=" * 50)

    try:
        # Run examples
        simple_concept_explanation()

        # Note: Uncomment these for full testing
        # lightweight_model_example()  # Requires different model
        # conversational_example()
        # financial_analysis_example()

        print("\n=== All Examples Completed Successfully ===")

    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("This might be due to model loading requirements or missing dependencies.")