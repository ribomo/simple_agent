# Plain Agent

A small agent that calls OpenAI compatible LLM APIs.

This project starts with a streaming agent loop:

```text
Interactive terminal
  -> reads your prompt
  -> streams assistant text as it arrives
  -> detects tool calls from the model
  -> runs workspace tools when requested
  -> sends tool results back to the model
  -> repeats until the assistant gives a final answer
```

## Install

Plain Agent is available on PyPI and can be installed with `pipx`:

```bash
pipx install plain-agent
```

If `pipx` is not installed, follow the [official installation guide](https://pipx.pypa.io/stable/how-to/install-pipx/).

For local development, this project uses `uv` to track the Python environment.
If `uv` is not installed, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Configuration

Create a local `.env` file or export environment variables in your shell. See `.env.example` for more examples.

For DeepSeek:

```bash
export DEEPSEEK_API_KEY="your-api-key"
export LLM_PROVIDER="deepseek"
export LLM_MODEL="deepseek-v4-flash"
```

For OpenAI:

```bash
export OPENAI_API_KEY="your-api-key"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-5.4-mini"
```

You can still set `LLM_BASE_URL` when you want to override the provider default, such as pointing at a local OpenAI compatible server like Ollama.

## Run

```bash
uv run plain-agent
```
