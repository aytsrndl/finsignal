"""Lazy, configurable OpenAI structured-output client."""
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class AnalysisError(RuntimeError):
    """The provider did not return a complete structured analysis."""


class StructuredLLM:
    def __init__(self, *, model=None, client=None):
        self.model = model or os.environ.get('FINSIGNAL_LLM_MODEL') or 'gpt-4.1-mini'
        self._client = client

    def validate_configuration(self):
        """Fail early before fetching SEC data when credentials are missing."""
        if self._client is None and not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('Set OPENAI_API_KEY before requesting LLM analysis.')

    def analyze(self, *, instructions: str, payload: str, schema: type[T]) -> T:
        self.validate_configuration()
        if self._client is None:
            self._client = OpenAI(timeout=120.0, max_retries=2)
        response = self._client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=payload,
            text_format=schema,
            store=False,
        )
        if response.status != 'completed' or response.output_parsed is None:
            raise AnalysisError('LLM analysis was refused or incomplete; no report was produced.')
        return schema.model_validate(response.output_parsed)
