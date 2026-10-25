"""Groq DeepEval LLM wrapper - uses Groq API instead of OpenAI.

This implements DeepEvalBaseLLM to allow the existing DeepEval metrics
(AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric)
to use Groq instead of OpenAI.

Model default is sourced from config.Settings.groq_eval_model so that
a single environment variable (GROQ_EVAL_MODEL) controls the model
used for both routing and evaluation.

Uses LangChain ChatGroq under the hood, configured with GROQ_API_KEY
from the environment.
"""

import os
import asyncio
from typing import Any

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

from config import settings


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """A DeepEvalBaseLLM wrapper around LangChain ChatGroq using Groq API.

    The default model is ``settings.groq_eval_model`` (currently
    ``openai/gpt-oss-120b``).  Override via the GROQ_EVAL_MODEL env
    variable or by passing ``model_name`` explicitly.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        groq_api_key: str | None = None,
    ):
        # Fall back to the centralized registry if not given explicitly
        self.model_name = model_name or settings.groq_eval_model
        self.temperature = temperature
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it via environment variable or pass explicitly."
            )

        # Initialize the LangChain ChatGroq
        self._llm = ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            groq_api_key=self.groq_api_key,
        )

    def get_model_name(self) -> str:
        """Required by DeepEvalBaseLLM interface."""
        return self.model_name

    def load_model(self):
        """Required by DeepEvalBaseLLM interface - no-op for LangChain model."""

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> Any:
        """Async generation required by DeepEvalBaseLLM."""
        loop = asyncio.get_event_loop()
        if schema is not None:
            try:
                structured_llm = self._llm.with_structured_output(schema)
                return await loop.run_in_executor(
                    None,
                    lambda: structured_llm.invoke(prompt),
                )
            except Exception:
                pass
        result = await loop.run_in_executor(
            None,
            lambda: self._llm.invoke(prompt),
        )
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def generate(self, prompt: str, schema=None, **kwargs) -> Any:
        """Sync generation required by DeepEvalBaseLLM."""
        if schema is not None:
            try:
                structured_llm = self._llm.with_structured_output(schema)
                return structured_llm.invoke(prompt)
            except Exception:
                pass
        result = self._llm.invoke(prompt)
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "text"):
            return result.text
        return str(result)


def get_groq_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
) -> GroqDeepEvalLLM:
    """Factory function to create a GroqDeepEvalLLM instance.

    If ``model_name`` is omitted the value from
    ``settings.groq_eval_model`` is used (default: ``openai/gpt-oss-120b``).
    """
    return GroqDeepEvalLLM(
        model_name=model_name,
        temperature=temperature,
    )