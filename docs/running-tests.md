# Running Tests

## Set up the development environment

Install the project and its dependencies from the repository root:

```bash
uv sync
```

## Run the full test suite

Use Python's standard library test runner:

```bash
uv run python -m unittest discover
```

Add `-v` for more detailed output:

```bash
uv run python -m unittest discover -v
```

## Run specific tests

Run one test module:

```bash
uv run python -m unittest tests.test_llm_client
```

Run one test class or method by providing its fully qualified name:

```bash
uv run python -m unittest tests.test_llm_client.LLMClientTest
uv run python -m unittest tests.test_llm_client.LLMClientTest.test_client_requires_api_key
```

Replace the class and method names in these examples with the test you want to run.

## Linux integration tests

Some integration tests require Bubblewrap. They run when Bubblewrap passes the project's
availability checks; otherwise, the affected test class is reported as skipped.
