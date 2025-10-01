"""
Unified LLM factory for creating consistent model instances
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from config import Config

class BaseLLM(ABC):
    """Base class for all LLM implementations"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.DEFAULT_LLM_MODEL
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.llm = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

class HuggingFaceLLM(BaseLLM):
    """Standardized HuggingFace LLM implementation"""

    def __init__(self, model_name: str = None, use_quantization: bool = True):
        super().__init__(model_name)
        self.use_quantization = use_quantization
        self.load_model()

    def load_model(self) -> None:
        """Load model with consistent configuration"""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if requested
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }

        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(**Config.QUANTIZATION_CONFIG)
            model_kwargs["quantization_config"] = quantization_config

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            **Config.GENERATION_PARAMS
        )

        # Create LangChain wrapper
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        print("Model loaded successfully")

    def generate(self, prompt: str, max_length: int = None, **kwargs) -> str:
        """Generate text using the pipeline"""
        try:
            # Use pipeline directly for generation
            generation_params = Config.GENERATION_PARAMS.copy()
            if max_length:
                generation_params["max_new_tokens"] = max_length
            generation_params.update(kwargs)

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + generation_params["max_new_tokens"],
                    num_return_sequences=1,
                    temperature=generation_params["temperature"],
                    do_sample=generation_params["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only new generated part
            response = response[len(prompt):].strip()

            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def create_chain(self, template: str, input_variables: list):
        """Create a LangChain chain with the given template"""
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        return prompt | self.llm

class LLMFactory:
    """Factory for creating LLM instances"""

    _instances = {}

    @classmethod
    def get_llm(cls, model_name: str = None, use_quantization: bool = True, force_reload: bool = False) -> HuggingFaceLLM:
        """Get or create LLM instance (singleton pattern)"""
        key = f"{model_name or Config.DEFAULT_LLM_MODEL}_{use_quantization}"

        if force_reload or key not in cls._instances:
            cls._instances[key] = HuggingFaceLLM(model_name, use_quantization)

        return cls._instances[key]

    @classmethod
    def clear_cache(cls):
        """Clear cached instances"""
        cls._instances.clear()