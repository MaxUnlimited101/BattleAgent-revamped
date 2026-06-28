"""Phase 4: batched transport (run_LLM_batch) and the cache_control prefix block in run_LLM."""
from langchain_core.messages import HumanMessage


class _StubResponse:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class _StubModel:
    """Captures the messages it is handed and returns responses carrying usage metadata."""

    def __init__(self):
        self.captured = []
        self.batch_config = None

    def invoke(self, messages):
        self.captured.append(messages)
        return _StubResponse("solo", {"input_tokens": 10, "output_tokens": 5})

    def batch(self, messages_list, config=None):
        self.batch_config = config
        self.captured.extend(messages_list)
        return [
            _StubResponse(f"resp{i}", {"input_tokens": 1, "output_tokens": 2})
            for i, _ in enumerate(messages_list)
        ]


def test_run_llm_batch_order_and_token_accounting(monkeypatch):
    from utils import LLM_api
    from utils.token_accounting import TokenAccumulator

    stub = _StubModel()
    monkeypatch.setattr(LLM_api, "_get_model", lambda mt: stub)

    acc = TokenAccumulator()
    out = LLM_api.run_LLM_batch("claude", ["a", "b", "c"], acc, "diary", max_concurrency=4)

    assert out == ["resp0", "resp1", "resp2"]
    assert stub.batch_config == {"max_concurrency": 4}
    assert acc.diary.calls == 3
    assert acc.diary.input_tokens == 3
    assert acc.diary.output_tokens == 6


def test_run_llm_batch_empty_is_noop(monkeypatch):
    from utils import LLM_api

    called = {"hit": False}

    def _boom(mt):
        called["hit"] = True
        raise AssertionError("model should not be built for empty batch")

    monkeypatch.setattr(LLM_api, "_get_model", _boom)
    assert LLM_api.run_LLM_batch("claude", []) == []
    assert called["hit"] is False


def test_cache_prefix_builds_two_blocks_for_claude(monkeypatch):
    from utils import LLM_api

    stub = _StubModel()
    monkeypatch.setattr(LLM_api, "_get_model", lambda mt: stub)

    LLM_api.run_LLM("claude", "the-suffix", cache_prefix="the-prefix")

    message = stub.captured[0][0]
    assert isinstance(message, HumanMessage)
    assert isinstance(message.content, list) and len(message.content) == 2
    assert message.content[0]["text"] == "the-prefix"
    assert message.content[0]["cache_control"] == {"type": "ephemeral"}
    assert message.content[1]["text"] == "the-suffix"
    assert "cache_control" not in message.content[1]


def test_cache_prefix_concatenates_for_non_claude(monkeypatch):
    from utils import LLM_api

    stub = _StubModel()
    monkeypatch.setattr(LLM_api, "_get_model", lambda mt: stub)

    LLM_api.run_LLM("gpt", "the-suffix", cache_prefix="the-prefix")

    message = stub.captured[0][0]
    assert message.content == "the-prefix\n\nthe-suffix"
