import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

_FAKE_DEFAULT_RESPONSE = json.dumps({
    "agentNextActionType": "Defend",
    "remarks": "Holding position",
    "SubAgentsRecall": [],
    "agentMoral": "Medium",
    "speed": 5,
    "agentNextPosition": [15, -10],
    "deploySubUnit": False,
    "targetedAgentId": "",
    "actions": [],
    "casualties_result": [],
})

_fake_model = None


def _get_fake_model():
    global _fake_model
    if _fake_model is None:
        _fake_model = FakeListChatModel(responses=[_FAKE_DEFAULT_RESPONSE])
    return _fake_model


def _make_claude_model():
    return ChatAnthropic(
        model="claude-3-opus-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=3000,
        temperature=1,
        max_retries=3,
    )


def _make_gpt_model(temperature=0, model="gpt-4-1106-preview", seed=440):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        model_kwargs={"seed": seed},
        max_retries=3,
    )


def _make_openrouter_model(temperature=0, seed=440):
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        model_kwargs={"seed": seed},
        max_retries=3,
    )


def _make_ollama_model(temperature=0):
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return ChatOpenAI(
        model=model,
        api_key="ollama",
        base_url=base_url,
        temperature=temperature,
        max_retries=3,
    )


def _get_model(model_type):
    if model_type == "gpt":
        return _make_gpt_model()
    elif model_type == "claude":
        return _make_claude_model()
    elif model_type == "openrouter":
        return _make_openrouter_model()
    elif model_type == "ollama":
        return _make_ollama_model()
    elif model_type == "fake":
        return _get_fake_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_message(model_type, prompt, cache_prefix):
    """Build the HumanMessage for a call, optionally with an Anthropic cache_control prefix block."""
    if cache_prefix is None:
        return HumanMessage(content=prompt)
    if model_type == "claude":
        return HumanMessage(content=[
            {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ])
    # Other providers: caching not wired here — fold the prefix back into a plain string.
    return HumanMessage(content=cache_prefix + "\n\n" + prompt)


def _record_usage(accumulator, role, response):
    if accumulator is not None and role is not None:
        meta = response.usage_metadata
        if meta is not None:
            accumulator.add(role, meta["input_tokens"], meta["output_tokens"])


def _clean_content(model_type, content):
    if model_type in ("gpt", "openrouter", "ollama"):
        content = content.replace("```json", "").replace("```", "")
    return content


def run_LLM(model_type, prompt, accumulator=None, role=None, cache_prefix=None):
    """String-in/string-out LLM call. Optionally captures token usage into accumulator.

    When ``cache_prefix`` is given and ``model_type == "claude"``, the prefix is sent as a
    separate ``cache_control`` text block so Anthropic prompt caching can reuse it across calls.
    """
    model = _get_model(model_type)
    response = model.invoke([_build_message(model_type, prompt, cache_prefix)])
    _record_usage(accumulator, role, response)
    return _clean_content(model_type, response.content)


def run_LLM_batch(model_type, prompts, accumulator=None, role=None, max_concurrency=8, cache_prefix=None):
    """Batched, order-preserving variant of ``run_LLM``.

    Issues all ``prompts`` as a single concurrent fan-out via LangChain ``.batch(...)`` and
    returns a list of cleaned string responses in the same order. Per-response token usage is
    accumulated under ``role``.

    ``cache_prefix`` may be ``None``, a single string applied to every prompt, or a per-prompt
    list (parallel to ``prompts``) — the latter is needed when agents have different static
    prefixes (e.g. opposing armies) but still want per-agent prompt caching.
    """
    if not prompts:
        return []
    if cache_prefix is None or isinstance(cache_prefix, str):
        prefixes = [cache_prefix] * len(prompts)
    else:
        prefixes = list(cache_prefix)
    model = _get_model(model_type)
    messages = [[_build_message(model_type, p, pre)] for p, pre in zip(prompts, prefixes)]
    responses = model.batch(messages, config={"max_concurrency": max_concurrency})
    results = []
    for response in responses:
        _record_usage(accumulator, role, response)
        results.append(_clean_content(model_type, response.content))
    return results
