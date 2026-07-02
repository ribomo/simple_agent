import unittest

from plain_agent.config import AppConfig
from plain_agent.environment import ENV_VARIABLE_NAMES
from plain_agent.providers import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_MODEL,
)


class AppConfigTest(unittest.TestCase):
    def test_lists_supported_environment_variable_names(self) -> None:
        self.assertEqual(
            set(ENV_VARIABLE_NAMES),
            {
                "LLM_PROVIDER",
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_BASE_URL",
                "LLM_MODEL",
                "LLM_TIMEOUT",
                "LLM_COMPACTION_KEEP_RECENT_EXCHANGES",
                "LLM_COMPACTION_MODEL",
                "LLM_COMPACTION_AUTO_MAX_TOKENS",
                "PLAIN_AGENT_ENABLE_NETWORK",
            },
        )

    def test_uses_deepseek_defaults(self) -> None:
        config = AppConfig.from_env({})

        self.assertEqual(config.llm.provider, "deepseek")
        self.assertIsNone(config.llm.api_key)
        self.assertEqual(config.llm.base_url, DEEPSEEK_BASE_URL)
        self.assertEqual(config.llm.model, DEEPSEEK_DEFAULT_MODEL)
        self.assertEqual(config.llm.timeout, 60.0)
        self.assertEqual(config.compaction.keep_recent_exchanges, 2)
        self.assertIsNone(config.compaction.model)
        self.assertEqual(config.compaction.auto_max_tokens, 200_000)
        self.assertTrue(config.network.enabled)

    def test_loads_openai_and_optional_settings(self) -> None:
        config = AppConfig.from_env(
            {
                "LLM_PROVIDER": "OPENAI",
                "OPENAI_API_KEY": "openai-secret",
                "LLM_BASE_URL": "https://llm.example/v1",
                "LLM_MODEL": "custom-model",
                "LLM_TIMEOUT": "12.5",
                "LLM_COMPACTION_KEEP_RECENT_EXCHANGES": "4",
                "LLM_COMPACTION_MODEL": "summary-model",
                "LLM_COMPACTION_AUTO_MAX_TOKENS": "0",
                "PLAIN_AGENT_ENABLE_NETWORK": "yes",
            }
        )

        self.assertEqual(config.llm.provider, "openai")
        self.assertEqual(config.llm.api_key, "openai-secret")
        self.assertEqual(config.llm.base_url, "https://llm.example/v1")
        self.assertEqual(config.llm.model, "custom-model")
        self.assertEqual(config.llm.timeout, 12.5)
        self.assertEqual(config.compaction.keep_recent_exchanges, 4)
        self.assertEqual(config.compaction.model, "summary-model")
        self.assertIsNone(config.compaction.auto_max_tokens)
        self.assertTrue(config.network.enabled)

    def test_uses_generic_llm_api_key_as_fallback(self) -> None:
        config = AppConfig.from_env(
            {
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": "generic-secret",
            }
        )

        self.assertEqual(config.llm.api_key, "generic-secret")
        self.assertEqual(config.llm.base_url, OPENAI_BASE_URL)
        self.assertEqual(config.llm.model, OPENAI_DEFAULT_MODEL)

    def test_secret_values_are_hidden_from_repr(self) -> None:
        config = AppConfig.from_env(
            {
                "DEEPSEEK_API_KEY": "llm-secret",
            }
        )

        rendered = repr(config)

        self.assertNotIn("llm-secret", rendered)

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            AppConfig.from_env({"LLM_PROVIDER": "unknown"})

    def test_rejects_invalid_network_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean environment settings"):
            AppConfig.from_env({"PLAIN_AGENT_ENABLE_NETWORK": "sometimes"})

    def test_network_can_be_disabled(self) -> None:
        config = AppConfig.from_env({"PLAIN_AGENT_ENABLE_NETWORK": "false"})

        self.assertFalse(config.network.enabled)

    def test_rejects_negative_integer_setting(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            AppConfig.from_env({"LLM_COMPACTION_AUTO_MAX_TOKENS": "-1"})

    def test_rejects_zero_recent_exchange_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            AppConfig.from_env({"LLM_COMPACTION_KEEP_RECENT_EXCHANGES": "0"})

    def test_rejects_non_positive_or_non_finite_timeout(self) -> None:
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    AppConfig.from_env({"LLM_TIMEOUT": value})


if __name__ == "__main__":
    unittest.main()
