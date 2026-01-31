from typing import Optional
from google import genai
from google.genai import types
from ..api.llm_engine import LLMEngine, LLMOptions, LLMResponse


class GoogleGenAIEngine(LLMEngine):
    """
    Implementation of LLMEngine using the Google GenAI SDK (Gemini).
    """

    def __init__(self, api_key):
        self._api_key = api_key
        self._client: Optional[genai.Client] = None
        self._generation_config: Optional[types.GenerateContentConfig] = None
        self._model_name: Optional[str] = None

    def load(self, options: LLMOptions) -> None:
        if not self._api_key:
            raise ValueError("API Key is missing.")

        self._client = genai.Client(api_key=self._api_key)
        self._model_name = options.model_name

        thinking_config = None
        if not options.thinking_mode:
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        else:
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1, include_thoughts=False
            )

        self._generation_config = types.GenerateContentConfig(
            temperature=options.temperature,
            thinking_config=thinking_config,
        )

    def unload(self) -> None:
        """
        Resets the client and configuration.
        """
        self._client = None
        self._generation_config = None
        self._model_name = None

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generates content using the loaded configuration.
        """
        if self._client is None or self._generation_config is None:
            raise RuntimeError("Engine is not loaded. Call load() first.")

        try:
            # We use generate_content for a stateless request
            response = self._client.models.generate_content(
                model=self._model_name, contents=prompt, config=self._generation_config
            )

            response_text = response.text if response.text else ""
            return LLMResponse(text=response_text)

        except Exception as e:
            raise RuntimeError(f"Google GenAI generation failed: {e}") from e
