"""Tests for the 'fake' LLM provider in run_LLM."""
import json

import pytest


class TestFakeProvider:
    def test_run_llm_fake_returns_string(self):
        from utils.LLM_api import run_LLM
        result = run_LLM("fake", "any prompt")
        assert isinstance(result, str)

    def test_fake_response_is_valid_json(self):
        from utils.LLM_api import run_LLM
        result = run_LLM("fake", "any prompt")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_fake_response_has_commander_fields(self):
        from utils.LLM_api import run_LLM
        result = run_LLM("fake", "any prompt")
        parsed = json.loads(result)
        assert "agentNextActionType" in parsed
        assert "agentMoral" in parsed
        assert parsed["agentMoral"] in ("High", "Medium", "Low")
        assert "agentNextPosition" in parsed
        assert len(parsed["agentNextPosition"]) == 2

    def test_fake_response_passes_validate_parsed_output(self, make_agent):
        from utils.LLM_api import run_LLM
        import json as _json
        result = run_LLM("fake", "any prompt")
        parsed = _json.loads(result)
        agent = make_agent()
        errors = agent.validate_parsed_output(parsed)
        assert errors == [], f"Fake response failed validation: {errors}"

    def test_unknown_model_type_raises(self):
        from utils.LLM_api import run_LLM
        with pytest.raises(ValueError, match="Unknown model type"):
            run_LLM("invalid_model", "test")

    def test_fake_singleton_is_reused(self):
        from utils.LLM_api import _get_fake_model
        m1 = _get_fake_model()
        m2 = _get_fake_model()
        assert m1 is m2

    def test_fake_multiple_calls_all_return_same_valid_json(self):
        from utils.LLM_api import run_LLM
        for _ in range(5):
            result = run_LLM("fake", "prompt")
            parsed = json.loads(result)
            assert "agentNextActionType" in parsed
