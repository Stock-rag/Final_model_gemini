# langchain_llm_open.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# ---------------------------
# 1️⃣ Load an open LLaMA 2 model
# ---------------------------
model_name = "NousResearch/Llama-2-7b-hf"  # fully open variant

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # automatically uses GPU if available
    torch_dtype="auto",
    load_in_8bit=True      # automatically chooses FP16 or BF16 if supported
)

# ---------------------------
# 2️⃣ Create a HuggingFace pipeline
# ---------------------------
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# 3️⃣ Define a LangChain prompt
# ---------------------------
prompt = PromptTemplate(
    input_variables=["concept"],
    template="Explain the concept of {concept} in one simple sentence."
)

# ---------------------------
# 4️⃣ Create a chain
# ---------------------------
chain = prompt | llm

# ---------------------------
# 5️⃣ Run the chain
# ---------------------------
concept = "Large Language Model"
result = chain.invoke({"concept": concept})
print("Result:\n", result)
