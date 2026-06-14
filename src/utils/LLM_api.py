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


def run_LLM(model_type, prompt, accumulator=None, role=None):
    """String-in/string-out LLM call. Optionally captures token usage into accumulator."""
    model = _get_model(model_type)
    response = model.invoke([HumanMessage(content=prompt)])

    if accumulator is not None and role is not None:
        meta = response.usage_metadata
        if meta is not None:
            accumulator.add(role, meta["input_tokens"], meta["output_tokens"])

    content = response.content
    if model_type in ("gpt", "openrouter", "ollama"):
        content = content.replace("```json", "").replace("```", "")
    return content
