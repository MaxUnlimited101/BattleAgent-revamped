"""Tests for TokenAccumulator and run_LLM token capture integration."""
import pytest

from utils.token_accounting import RoleTokens, TokenAccumulator


class TestRoleTokens:
    def test_initial_state_zero(self):
        rt = RoleTokens()
        assert rt.input_tokens == 0
        assert rt.output_tokens == 0
        assert rt.calls == 0

    def test_add_increments_all_fields(self):
        rt = RoleTokens()
        rt.add(100, 50)
        assert rt.input_tokens == 100
        assert rt.output_tokens == 50
        assert rt.calls == 1

    def test_add_accumulates(self):
        rt = RoleTokens()
        rt.add(100, 50)
        rt.add(200, 30)
        assert rt.input_tokens == 300
        assert rt.output_tokens == 80
        assert rt.calls == 2


class TestTokenAccumulator:
    def test_initial_totals_zero(self):
        acc = TokenAccumulator()
        assert acc.total_input == 0
        assert acc.total_output == 0

    def test_add_commander_increments_total(self):
        acc = TokenAccumulator()
        acc.add("commander", 100, 50)
        assert acc.commander.input_tokens == 100
        assert acc.total_input == 100
        assert acc.total_output == 50

    def test_add_multiple_roles_accumulate_total(self):
        acc = TokenAccumulator()
        acc.add("commander", 100, 50)
        acc.add("referee", 200, 30)
        acc.add("diary", 50, 20)
        assert acc.total_input == 350
        assert acc.total_output == 100

    def test_unknown_role_silently_ignored(self):
        acc = TokenAccumulator()
        acc.add("nonexistent_role", 100, 50)
        assert acc.total_input == 0
        assert acc.total_output == 0

    def test_summary_dict_has_all_keys(self):
        acc = TokenAccumulator()
        d = acc.summary_dict()
        assert "commander" in d
        assert "referee" in d
        assert "diary" in d
        assert "total" in d
        assert "in" in d["total"]
        assert "out" in d["total"]

    def test_summary_dict_values_correct(self):
        acc = TokenAccumulator()
        acc.add("commander", 100, 50)
        d = acc.summary_dict()
        assert d["commander"]["in"] == 100
        assert d["commander"]["out"] == 50
        assert d["commander"]["calls"] == 1
        assert d["total"]["in"] == 100

    def test_cost_estimate_known_model(self):
        acc = TokenAccumulator()
        acc.add("commander", 1_000_000, 1_000_000)
        cost = acc.cost_estimate("gpt")
        assert cost is not None
        assert cost > 0

    def test_cost_estimate_openrouter_returns_none(self):
        acc = TokenAccumulator()
        acc.add("commander", 1000, 500)
        cost = acc.cost_estimate("openrouter")
        assert cost is None

    def test_cost_estimate_ollama_returns_none(self):
        acc = TokenAccumulator()
        cost = acc.cost_estimate("ollama")
        assert cost is None

    def test_cost_estimate_fake_is_zero(self):
        acc = TokenAccumulator()
        acc.add("commander", 1_000_000, 1_000_000)
        cost = acc.cost_estimate("fake")
        assert cost == 0.0


class TestRunLLMTokenCapture:
    def test_run_llm_fake_without_accumulator_unchanged(self):
        from utils.LLM_api import run_LLM
        result = run_LLM("fake", "test prompt")
        assert isinstance(result, str)

    def test_run_llm_fake_with_accumulator_stays_zero(self):
        """FakeListChatModel returns None for usage_metadata — counts stay zero."""
        from utils.LLM_api import run_LLM
        acc = TokenAccumulator()
        run_LLM("fake", "test prompt", accumulator=acc, role="commander")
        # Fake model has no usage_metadata — no counts recorded
        assert acc.commander.input_tokens == 0
        assert acc.commander.calls == 0

    def test_run_llm_fake_with_accumulator_returns_string(self):
        from utils.LLM_api import run_LLM
        acc = TokenAccumulator()
        result = run_LLM("fake", "test", accumulator=acc, role="referee")
        assert isinstance(result, str)
