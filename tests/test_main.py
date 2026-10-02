import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, chat_completion_stream, load_codex_models

client = TestClient(app)


def completed(stdout: str = "Hello") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["codex"], returncode=0, stdout=f"{stdout}\n", stderr=""
    )


def test_chat_completion_has_openai_compatible_shape() -> None:
    with patch("app.main.subprocess.run", return_value=completed("Sales: 42")) as run:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-luna",
                "messages": [
                    {"role": "developer", "content": "Use MCP tools."},
                    {"role": "user", "content": "Retrieve today's sales."},
                ],
                "reasoning_effort": "high",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "gpt-5.6-luna"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Sales: 42",
                "refusal": None,
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]

    call = run.call_args
    assert call.args[0] == [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--model",
        "gpt-5.6-luna",
        "--config",
        "model_reasoning_effort=high",
        "-",
    ]
    prompt = call.kwargs["input"]
    assert json.dumps(
        [
            {"role": "developer", "content": "Use MCP tools."},
            {"role": "user", "content": "Retrieve today's sales."},
        ]
    ) in prompt
    assert call.kwargs["timeout"] == 300


class FakeStreamingProcess:
    def __init__(self, lines: list[dict[str, object]], returncode: int = 0) -> None:
        self.stdout = io.StringIO("".join(f"{json.dumps(line)}\n" for line in lines))
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: int | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_chat_completion_stream_translates_codex_events() -> None:
    process = FakeStreamingProcess(
        [
            {
                "type": "item.updated",
                "item": {"type": "agent_message", "text": "Hello"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Hello world"},
            },
        ]
    )

    events = list(
        chat_completion_stream(
            process,  # type: ignore[arg-type]
            "chatcmpl-test",
            123,
            "gpt-test",
        )
    )

    assert json.loads(events[0].removeprefix("data: "))["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    assert json.loads(events[1].removeprefix("data: "))["choices"][0]["delta"] == {
        "content": "Hello"
    }
    assert json.loads(events[2].removeprefix("data: "))["choices"][0]["delta"] == {
        "content": " world"
    }
    assert json.loads(events[3].removeprefix("data: "))["choices"][0][
        "finish_reason"
    ] == "stop"
    assert events[4] == "data: [DONE]\n\n"


def test_chat_completion_stream_endpoint() -> None:
    process = FakeStreamingProcess(
        [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Hello"},
            }
        ]
    )
    with patch("app.main.start_codex_stream", return_value=process):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-luna",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert '"content":"Hello"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_chat_completion_requires_messages() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.6-luna", "messages": []},
    )

    assert response.status_code == 422


def test_list_models_uses_codex_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "models_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-test",
                        "display_name": "GPT Test",
                        "description": "Test model",
                        "visibility": "list",
                        "supported_in_api": True,
                        "input_modalities": ["text", "image"],
                        "context_window": 128000,
                        "max_context_window": 256000,
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "high"},
                        ],
                        "additional_speed_tiers": ["fast"],
                    },
                    {
                        "slug": "hidden-model",
                        "visibility": "hide",
                        "supported_in_api": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    body = load_codex_models(cache_path).model_dump()

    assert body["object"] == "list"
    assert len(body["models"]) == 1
    assert body["models"][0] == {
        "name": "gpt-test",
        "model": "gpt-test",
        "type": "model",
        "description": "Test model",
        "tags": ["fast"],
        "capabilities": ["completion", "multimodal"],
        "details": {
            "display_name": "GPT Test",
            "input_modalities": ["text", "image"],
            "reasoning_efforts": ["low", "high"],
            "context_window": 128000,
            "max_context_window": 256000,
        },
    }
    assert body["data"][0]["id"] == "gpt-test"
    assert body["data"][0]["owned_by"] == "openai"


def test_models_endpoint_returns_catalog() -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"]
    assert all(model["object"] == "model" for model in body["data"])


def test_prompt_endpoint_remains_backward_compatible() -> None:
    with patch("app.main.subprocess.run", return_value=completed("Hi")):
        response = client.post("/prompt", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json() == {"output": "Hi"}


def test_cli_failure_is_reported() -> None:
    failure = subprocess.CompletedProcess(
        args=["codex"], returncode=1, stdout="", stderr="authentication failed\n"
    )
    with patch("app.main.subprocess.run", return_value=failure):
        response = client.post("/prompt", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Codex failed: authentication failed"}


def test_timeout_is_reported() -> None:
    with patch(
        "app.main.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=300),
    ):
        response = client.post("/prompt", json={"prompt": "Hello"})

    assert response.status_code == 504


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
