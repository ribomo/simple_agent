import sys
import types
import unittest


class FakeOpenAI:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeOpenAI.last_kwargs = kwargs
        self.chat = object()


fake_openai_module = types.ModuleType("openai")
fake_openai_module.OpenAI = FakeOpenAI
sys.modules["openai"] = fake_openai_module

from simple_agent.llm_client import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    DeepSeekClient,
    LLMClientError,
    OpenAIClient,
)


class LLMClientTest(unittest.TestCase):
    def test_openai_client_creates_sdk_client_with_openai_settings(self) -> None:
        client = OpenAIClient(api_key="test-key", timeout=12)

        self.assertEqual(client.base_url, OPENAI_BASE_URL)
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.timeout, 12)
        self.assertIsInstance(client.client, FakeOpenAI)
        self.assertEqual(
            FakeOpenAI.last_kwargs,
            {
                "base_url": OPENAI_BASE_URL,
                "timeout": 12,
                "api_key": "test-key",
            },
        )

    def test_client_accepts_custom_base_url(self) -> None:
        client = OpenAIClient(
            base_url="https://example.test/v1",
            api_key="test-key",
            timeout=12,
        )

        self.assertEqual(client.base_url, "https://example.test/v1")
        self.assertEqual(
            FakeOpenAI.last_kwargs,
            {
                "base_url": "https://example.test/v1",
                "timeout": 12,
                "api_key": "test-key",
            },
        )

    def test_client_requires_api_key(self) -> None:
        with self.assertRaisesRegex(LLMClientError, "api_key is required"):
            OpenAIClient(api_key=None)

    def test_client_delegates_chat_to_sdk_client(self) -> None:
        client = OpenAIClient(api_key="test-key")

        self.assertIs(client.chat, client.client.chat)

    def test_deepseek_client_creates_sdk_client_with_deepseek_base_url(self) -> None:
        client = DeepSeekClient(api_key="test-key")

        self.assertEqual(client.base_url, DEEPSEEK_BASE_URL)
        self.assertEqual(FakeOpenAI.last_kwargs["base_url"], DEEPSEEK_BASE_URL)


if __name__ == "__main__":
    unittest.main()
