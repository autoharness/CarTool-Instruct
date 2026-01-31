from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str


@dataclass
class LLMOptions:
    """
    Configuration for the LLM generation.
    """

    model_name: str
    temperature: float = 1.0
    thinking_mode: bool = False

    def __post_init__(self):
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")


class LLMEngine(ABC):
    """
    Abstract base class for LLM inference engines.
    """

    @abstractmethod
    def load(self, options: LLMOptions) -> None:
        """
        Initializes the engine with the provided options.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Releases resources and resets the engine state.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, options: LLMOptions) -> LLMResponse:
        """
        Stateless generation: Sends a prompt, returns a response.
        Does not maintain conversation history internally.
        """
        pass
