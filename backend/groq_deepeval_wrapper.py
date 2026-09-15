"""Groq DeepEval LLM wrapper - uses Groq API instead of OpenAI.

This implements DeepEvalBaseLLM to allow the existing DeepEval metrics
(AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric)
to use Groq instead of OpenAI.

Uses LangChain ChatGroq under the hood, configured with GROQ_API_KEY
from the environment.
"""

import os
import asyncio

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """A DeepEvalBaseLLM wrapper around LangChain ChatGroq using Groq API."""

    def __init__(self, model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it via environment variable or pass explicitly."
            )

        # Initialize the LangChain ChatGroq
        self._llm = ChatGroq(
            model_name=model_name,
            temperature=self.temperature,
            groq_api_key=self.groq_api_key,
        )

    def get_model_name(self):
        """Required by DeepEvalBaseLLM interface."""
        return self.model_name

    def load_model(self):
        """Required by DeepEvalBaseLLM interface - no-op for LangChain model."""
        pass

    async def a_generate(self, prompt: str, **kwargs):
        """Async generation required by DeepEvalBaseLLM."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm.invoke(prompt)
        )
        # Return string content
        if hasattr(result, 'content'):
            return result.content
        elif hasattr(result, 'text'):
            return result.text
        return str(result)

    def generate(self, prompt: str, **kwargs):
        """Sync generation required by DeepEvalBaseLLM."""
        result = self._llm.invoke(prompt)
        if hasattr(result, 'content'):
            return result.content
        elif hasattr(result, 'text'):
            return result.text
        return str(result)


def get_groq_llm(model_name="llama-3.1-8b-instant", temperature=0.0):
    """Factory function to create a GroqDeepEvalLLM instance."""
    return GroqDeepEvalLLM(
        model_name=model_name,
        temperature=temperature,
    )