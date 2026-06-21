"""OpenAI-compatible LLM clients."""

from openai import OpenAI


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMClientError(ValueError):
    """Raised when LLM configuration is incomplete."""


class OpenAICompatibleClient:
    """Small wrapper around the official OpenAI SDK client."""

    base_url = OPENAI_BASE_URL

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMClientError("api_key is required to create an LLM client")

        self.api_key = api_key
        self.base_url = base_url or self.base_url
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def chat(self):
        return self.client.chat


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI client."""


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek client using its OpenAI-compatible API."""

    base_url = DEEPSEEK_BASE_URL
