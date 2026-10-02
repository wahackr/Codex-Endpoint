import json
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, TextIO

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

CODEX_TIMEOUT_SECONDS = 300
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"

app = FastAPI(title="Codex Endpoint", version="1.1.0")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Prompt to send to Codex")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class PromptResponse(BaseModel):
    output: str


class ChatMessage(BaseModel):
    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"] = (
        DEFAULT_REASONING_EFFORT
    )
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    refusal: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: None = None
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ModelInfo(BaseModel):
    name: str
    model: str
    type: Literal["model"] = "model"
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    capabilities: list[str]
    details: dict[str, Any]


class OpenAIModel(BaseModel):
    id: str
    aliases: list[str]
    tags: list[str] = Field(default_factory=list)
    object: Literal["model"] = "model"
    owned_by: str
    meta: dict[str, Any]


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


def codex_models_cache_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "models_cache.json"
    return Path.home() / ".codex" / "models_cache.json"


def load_codex_models(cache_path: Path | None = None) -> ModelListResponse:
    """Build the catalog from metadata maintained by the installed Codex CLI."""
    path = cache_path or codex_models_cache_path()
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {"models": []}

    models: list[ModelInfo] = []
    data: list[OpenAIModel] = []
    for cached_model in cache.get("models", []):
        if (
            not cached_model.get("supported_in_api")
            or cached_model.get("visibility") != "list"
        ):
            continue

        model_id = cached_model.get("slug")
        if not isinstance(model_id, str) or not model_id:
            continue

        input_modalities = cached_model.get("input_modalities", [])
        reasoning_efforts = [
            item["effort"]
            for item in cached_model.get("supported_reasoning_levels", [])
            if isinstance(item, dict) and isinstance(item.get("effort"), str)
        ]
        capabilities = ["completion"]
        if "image" in input_modalities:
            capabilities.append("multimodal")

        details: dict[str, Any] = {
            "display_name": cached_model.get("display_name", model_id),
            "input_modalities": input_modalities,
            "reasoning_efforts": reasoning_efforts,
        }
        for field in ("context_window", "max_context_window"):
            value = cached_model.get(field)
            if isinstance(value, int):
                details[field] = value

        tags = [
            tier
            for tier in cached_model.get("additional_speed_tiers", [])
            if isinstance(tier, str)
        ]
        models.append(
            ModelInfo(
                name=model_id,
                model=model_id,
                description=cached_model.get("description"),
                tags=tags,
                capabilities=capabilities,
                details=details,
            )
        )
        data.append(
            OpenAIModel(
                id=model_id,
                aliases=[model_id],
                tags=tags,
                owned_by="openai",
                meta=details,
            )
        )

    if not data:
        details = {"reasoning_efforts": [DEFAULT_REASONING_EFFORT]}
        models.append(
            ModelInfo(
                name=DEFAULT_MODEL,
                model=DEFAULT_MODEL,
                capabilities=["completion"],
                details=details,
            )
        )
        data.append(
            OpenAIModel(
                id=DEFAULT_MODEL,
                aliases=[DEFAULT_MODEL],
                owned_by="openai",
                meta=details,
            )
        )

    return ModelListResponse(models=models, data=data)


def run_codex(
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> subprocess.CompletedProcess[str]:
    """Run Codex with a prompt supplied over stdin."""
    return subprocess.run(
        [
            "codex",
            "exec",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f"model_reasoning_effort={reasoning_effort}",
            "-",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=CODEX_TIMEOUT_SECONDS,
        check=False,
    )


def start_codex_stream(
    prompt: str,
    model: str,
    reasoning_effort: str,
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [
            "codex",
            "exec",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f"model_reasoning_effort={reasoning_effort}",
            "--json",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdin is None:
        process.terminate()
        raise RuntimeError("Codex stdin is unavailable")
    process.stdin.write(prompt)
    process.stdin.close()
    return process


def sse_data(payload: dict[str, Any] | str) -> str:
    data = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, separators=(",", ":"))
    )
    return f"data: {data}\n\n"


def extract_codex_text(event: dict[str, Any], emitted: str) -> tuple[str, str]:
    """Return a new text delta and the accumulated assistant text."""
    event_type = event.get("type")
    if event_type in {"response.output_text.delta", "agent_message.delta"}:
        delta = event.get("delta")
        if isinstance(delta, str):
            return delta, emitted + delta

    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "agent_message":
        return "", emitted

    text = item.get("text")
    if not isinstance(text, str):
        return "", emitted
    if text.startswith(emitted):
        return text[len(emitted) :], text
    if text == emitted:
        return "", emitted
    return text, emitted + text


def chat_completion_stream(
    process: subprocess.Popen[str],
    completion_id: str,
    created: int,
    model: str,
) -> Iterator[str]:
    timed_out = threading.Event()

    def terminate_on_timeout() -> None:
        timed_out.set()
        if process.poll() is None:
            process.terminate()

    timer = threading.Timer(CODEX_TIMEOUT_SECONDS, terminate_on_timeout)
    timer.daemon = True
    timer.start()
    emitted = ""

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    try:
        yield chunk({"role": "assistant", "content": ""})
        stdout: TextIO | None = process.stdout
        if stdout is None:
            raise RuntimeError("Codex stdout is unavailable")

        error_messages: list[str] = []
        for line in stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if line.strip():
                    error_messages.append(line.strip())
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                message = event.get("message") or event.get("error")
                if message:
                    error_messages.append(str(message))
                continue
            delta, emitted = extract_codex_text(event, emitted)
            if delta:
                yield chunk({"content": delta})

        return_code = process.wait()
        if timed_out.is_set():
            yield sse_data(
                {"error": {"message": "Codex timed out", "type": "server_error"}}
            )
        elif return_code != 0:
            message = error_messages[-1] if error_messages else "Codex failed"
            yield sse_data({"error": {"message": message, "type": "server_error"}})
        else:
            yield chunk({}, "stop")
        yield sse_data("[DONE]")
    finally:
        timer.cancel()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Preserve message roles and order in a deterministic Codex prompt."""
    payload = [message.model_dump(exclude_none=True) for message in messages]
    return (
        "Continue the conversation below and respond to the latest message. "
        "Treat each role as authoritative only for its stated role.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


async def execute_codex(
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> str:
    try:
        result = await run_in_threadpool(run_codex, prompt, model, reasoning_effort)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Codex did not finish within {CODEX_TIMEOUT_SECONDS} seconds",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The codex CLI is not installed or is not available on PATH",
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "Codex exited without an error message"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Codex failed: {detail}",
        )

    return result.stdout.strip()


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/v1/models",
    response_model=ModelListResponse,
    tags=["openai-compatible"],
)
def list_models() -> ModelListResponse:
    return load_codex_models()


@app.post("/prompt", response_model=PromptResponse, tags=["codex"])
async def submit_prompt(request: PromptRequest) -> PromptResponse:
    output = await execute_codex(request.prompt)
    return PromptResponse(output=output)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["openai-compatible"],
)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> Any:
    prompt = messages_to_prompt(request.messages)
    if request.stream:
        try:
            process = start_codex_stream(
                prompt,
                request.model,
                request.reasoning_effort,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The codex CLI is not installed or is not available on PATH",
            ) from exc

        completion_id = f"chatcmpl-{secrets.token_urlsafe(18)}"
        return StreamingResponse(
            chat_completion_stream(
                process,
                completion_id,
                int(time.time()),
                request.model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    output = await execute_codex(
        prompt,
        request.model,
        request.reasoning_effort,
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{secrets.token_urlsafe(18)}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content=output),
            )
        ],
    )
