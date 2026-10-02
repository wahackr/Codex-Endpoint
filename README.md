# Codex Endpoint

A FastAPI service that passes a prompt to `codex exec` through standard input.
Codex runs with its filesystem sandbox in read-only mode.

## Run locally

Python 3.10+ and an installed, authenticated `codex` CLI are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Send an OpenAI-compatible chat completion request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.6-luna",
    "messages": [
      {"role": "developer", "content": "Use the available MCP tools."},
      {"role": "user", "content": "Retrieve today\u0027s sales."}
    ],
    "reasoning_effort": "medium"
  }'
```

The response uses the Chat Completions envelope:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1790928000,
  "model": "gpt-5.6-luna",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...Codex output...",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

Streaming requests are supported with `"stream": true`. They return OpenAI-style
`chat.completion.chunk` objects as server-sent events and finish with
`data: [DONE]`. The original `POST /prompt` endpoint remains available for simple
clients.

List the models currently advertised by the installed Codex CLI. The service
reads `models_cache.json` from `CODEX_HOME` (or `~/.codex`) and returns only
visible models supported by the API, including the metadata actually present in
that cache:

```bash
curl http://localhost:8000/v1/models
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Test

```bash
pytest
```
