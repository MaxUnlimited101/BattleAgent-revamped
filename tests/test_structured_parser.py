"""Tests for --parser structured mode in agent.execute()."""
import json

import pytest

from conftest import VALID_COMMANDER_JSON


class TestStructuredParser:
    def test_default_parser_mode_is_legacy(self, make_agent):
        agent = make_agent()
        assert agent.parser_mode == "legacy"

    def test_legacy_mode_still_works(self, make_agent, monkeypatch):
        agent = make_agent()
        assert agent.parser_mode == "legacy"
        monkeypatch.setattr(agent, "run_model", lambda p: json.dumps(VALID_COMMANDER_JSON))
        result = agent.execute()
        assert result["success"] is True

    def test_structured_valid_json_succeeds(self, make_agent, monkeypatch):
        agent = make_agent()
        agent.parser_mode = "structured"
        monkeypatch.setattr(agent, "run_model", lambda p: json.dumps(VALID_COMMANDER_JSON))
        result = agent.execute()
        assert result["success"] is True

    def test_structured_invalid_moral_triggers_retry(self, make_agent, monkeypatch):
        agent = make_agent()
        agent.parser_mode = "structured"
        bad = json.dumps({**VALID_COMMANDER_JSON, "agentMoral": "Scared"})
        good = json.dumps(VALID_COMMANDER_JSON)
        responses = iter([bad, good])
        monkeypatch.setattr(agent, "run_model", lambda p: next(responses))
        result = agent.execute()
        assert result["success"] is True
        assert result["attempts"] == 1

    def test_structured_all_retries_exhausted_raises(self, make_agent, monkeypatch):
        from agent import AgentExecutionError
        agent = make_agent()
        agent.parser_mode = "structured"
        bad = json.dumps({**VALID_COMMANDER_JSON, "agentMoral": "Frightened"})
        monkeypatch.setattr(agent, "run_model", lambda p: bad)
        with pytest.raises(AgentExecutionError):
            agent.execute()

    def test_structured_parse_failure_triggers_retry(self, make_agent, monkeypatch):
        """Non-JSON response should be treated as a parse+validation failure."""
        agent = make_agent()
        agent.parser_mode = "structured"
        responses = iter(["not json at all", json.dumps(VALID_COMMANDER_JSON)])
        monkeypatch.setattr(agent, "run_model", lambda p: next(responses))
        result = agent.execute()
        assert result["success"] is True

    def test_parser_mode_propagates_to_sub_agent(self, make_agent, monkeypatch):
        agent = make_agent(original=1000)
        agent.parser_mode = "structured"
        action = {
            "subAgent_NextActionType": "Charge",
            "troopType": "cavalry",
            "speed": 5,
            "deployedNum": 200,
            "ownPotentialLostNum": 0,
            "enemyPotentialLostNum": 0,
            "position": [5, 5],
            "agent_id": "ARMY-enemy",
            "remarks": "",
        }
        data = {**VALID_COMMANDER_JSON, "deploySubUnit": True, "actions": [action]}
        monkeypatch.setattr(agent, "run_model", lambda p: json.dumps(data))
        agent.execute()
        if agent.hierarchy.sub_agents:
            assert agent.hierarchy.sub_agents[0].parser_mode == "structured"
