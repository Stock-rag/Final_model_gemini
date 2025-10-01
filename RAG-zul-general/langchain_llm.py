from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langsmith import Client

# Initialize LangSmith client
client = Client()

# Load LLaMA 2 model (7B parameter version)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "NousResearch/Llama-2-7b-hf"  # Use open model that doesn't require auth
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
# Create a Hugging Face pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Define prompt
template = "Explain the following concept in one sentence: {concept}"
prompt = PromptTemplate(template=template, input_variables=["concept"])

# Create a chain using new syntax
chain = prompt | llm

# Run the chain
concept = "Large Language Model"
result = chain.invoke({"concept": concept})

print("LLM output:", result)

# Optional: log to LangSmith
client.record_run(chain, {"concept": concept}, {"output": result})