"""Tests for Detachment_Agent JSON parsing, validation, and parsed_data_sync."""

import copy
import pytest
from conftest import VALID_COMMANDER_JSON

from agent import Detachment_Agent, BranchStreamlining


# ---------------------------------------------------------------------------
# parse_llm_output
# ---------------------------------------------------------------------------

class TestParseLLMOutput:
    def test_extracts_json_embedded_in_prose(self, make_agent):
        agent = make_agent()
        text = 'Sure! Here is my decision: {"agentMoral": "High", "speed": 3}'
        result = agent.parse_llm_output(text)
        assert isinstance(result, dict)
        assert result["agentMoral"] == "High"

    def test_returns_dict_for_bare_json(self, make_agent):
        agent = make_agent()
        text = '{"agentMoral": "Low", "speed": 1}'
        result = agent.parse_llm_output(text)
        assert isinstance(result, dict)
        assert result["speed"] == 1

    def test_no_json_returns_error_string(self, make_agent):
        agent = make_agent()
        result = agent.parse_llm_output("No JSON here at all.")
        # parse_llm_output returns a string starting with "Error" when extraction fails
        assert isinstance(result, str)
        assert "Error" in result

    def test_malformed_json_returns_error_string(self, make_agent):
        agent = make_agent()
        text = '{"unclosed": '
        result = agent.parse_llm_output(text)
        assert isinstance(result, str)
        assert "Error" in result

    def test_nested_json_picks_outermost_braces(self, make_agent):
        agent = make_agent()
        text = 'Result: {"outer": {"inner": 1}}'
        result = agent.parse_llm_output(text)
        assert isinstance(result, dict)
        assert "outer" in result

    def test_json_with_trailing_text(self, make_agent):
        agent = make_agent()
        text = '{"key": "val"} some trailing text'
        result = agent.parse_llm_output(text)
        # rindex picks the last }, so trailing text after last } is discarded
        assert isinstance(result, dict)

    def test_empty_string_returns_error(self, make_agent):
        agent = make_agent()
        result = agent.parse_llm_output("")
        assert isinstance(result, str)
        assert "Error" in result

    def test_json5_accepts_trailing_comma(self, make_agent):
        agent = make_agent()
        text = '{"agentMoral": "High",}'
        result = agent.parse_llm_output(text)
        assert isinstance(result, dict)
        assert result["agentMoral"] == "High"


# ---------------------------------------------------------------------------
# validate_parsed_output
# ---------------------------------------------------------------------------

class TestValidateParsedOutput:
    def test_valid_json_returns_no_errors(self, make_agent):
        agent = make_agent()
        errors = agent.validate_parsed_output(copy.deepcopy(VALID_COMMANDER_JSON))
        assert errors == []

    def test_missing_agent_moral_flagged(self, make_agent):
        agent = make_agent()
        data = copy.deepcopy(VALID_COMMANDER_JSON)
        del data["agentMoral"]
        errors = agent.validate_parsed_output(data)
        assert any("agentMoral" in e for e in errors)

    def test_invalid_moral_value_flagged(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "agentMoral": "Scared"}
        errors = agent.validate_parsed_output(data)
        assert any("agentMoral" in e for e in errors)

    def test_all_valid_moral_values_accepted(self, make_agent):
        agent = make_agent()
        for val in ("High", "Medium", "Low"):
            data = {**VALID_COMMANDER_JSON, "agentMoral": val}
            assert agent.validate_parsed_output(data) == []

    def test_speed_must_be_int(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "speed": "fast"}
        errors = agent.validate_parsed_output(data)
        assert any("speed" in e for e in errors)

    def test_agent_next_position_must_be_two_int_list(self, make_agent):
        agent = make_agent()
        for bad_pos in ([], [1], [1, 2, 3], [1.0, 2.0], "10,20"):
            data = {**VALID_COMMANDER_JSON, "agentNextPosition": bad_pos}
            errors = agent.validate_parsed_output(data)
            assert any("agentNextPosition" in e for e in errors), f"Expected error for pos={bad_pos}"

    def test_deploy_sub_unit_must_be_bool(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "deploySubUnit": 1}
        errors = agent.validate_parsed_output(data)
        assert any("deploySubUnit" in e for e in errors)

    def test_sub_agents_recall_must_be_list_of_strings(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "SubAgentsRecall": [1, 2]}
        errors = agent.validate_parsed_output(data)
        assert any("SubAgentsRecall" in e for e in errors)

    def test_action_fields_validated(self, make_agent):
        agent = make_agent()
        valid_action = {
            "subAgent_NextActionType": "Charge",
            "troopType": "cavalry",
            "speed": 8,
            "deployedNum": 200,
            "ownPotentialLostNum": 10,
            "enemyPotentialLostNum": 50,
            "position": [5, 10],
            "agent_id": "ARMY-target",
            "remarks": "Flanking move",
        }
        data = {**VALID_COMMANDER_JSON, "actions": [valid_action]}
        assert agent.validate_parsed_output(data) == []

    def test_action_with_bad_position_flagged(self, make_agent):
        agent = make_agent()
        bad_action = {
            "subAgent_NextActionType": "Charge",
            "troopType": "cavalry",
            "speed": 8,
            "deployedNum": 200,
            "ownPotentialLostNum": 10,
            "enemyPotentialLostNum": 50,
            "position": "10,20",  # should be list
            "agent_id": "ARMY-target",
            "remarks": "Flanking move",
        }
        data = {**VALID_COMMANDER_JSON, "actions": [bad_action]}
        errors = agent.validate_parsed_output(data)
        assert any("position" in e for e in errors)

    def test_empty_json_produces_multiple_errors(self, make_agent):
        agent = make_agent()
        errors = agent.validate_parsed_output({})
        assert len(errors) >= 5


# ---------------------------------------------------------------------------
# parsed_data_sync
# ---------------------------------------------------------------------------

class TestParsedDataSync:
    def test_updates_moral(self, make_agent):
        agent = make_agent(moral="Low")
        data = {**VALID_COMMANDER_JSON, "agentMoral": "High"}
        agent.parsed_data_sync(data)
        assert agent.profile.moral == "High"

    def test_updates_position(self, make_agent):
        agent = make_agent(position=[0, 0])
        data = {**VALID_COMMANDER_JSON, "agentNextPosition": [50, 75]}
        agent.parsed_data_sync(data)
        assert agent.profile.position == [50, 75]

    def test_position_unchanged_when_same(self, make_agent):
        agent = make_agent(position=[10, 10])
        data = {**VALID_COMMANDER_JSON, "agentNextPosition": [10, 10]}
        agent.parsed_data_sync(data)
        assert agent.profile.position == [10, 10]

    def test_updates_current_action(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "agentNextActionType": "Retreat", "remarks": "falling back"}
        agent.parsed_data_sync(data)
        assert "Retreat" in agent.profile.current_action
        assert "falling back" in agent.profile.current_action

    def test_crushing_defeat_when_remaining_below_10pct(self, make_agent):
        # original=1000, lost=920 → remaining=80 < 100 (10%)
        agent = make_agent(original=1000, lost=920)
        data = {**VALID_COMMANDER_JSON}
        agent.parsed_data_sync(data)
        assert agent.profile.current_stage == "Crushing Defeat"

    def test_crushing_defeat_when_lost_over_50pct(self, make_agent):
        # original=1000, lost=600 → lost > 500 (50%)
        agent = make_agent(original=1000, lost=600)
        data = {**VALID_COMMANDER_JSON}
        agent.parsed_data_sync(data)
        assert agent.profile.current_stage == "Crushing Defeat"

    def test_fleeing_off_map_when_x_over_1000(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "agentNextPosition": [1001, 0]}
        agent.parsed_data_sync(data)
        assert agent.profile.current_stage == "fleeing Off the Map"

    def test_fleeing_off_map_when_y_over_1000(self, make_agent):
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "agentNextPosition": [0, 1001]}
        agent.parsed_data_sync(data)
        assert agent.profile.current_stage == "fleeing Off the Map"

    def test_in_battle_stage_preserved_when_healthy(self, make_agent):
        agent = make_agent(original=1000, lost=0, stage="In Battle")
        data = {**VALID_COMMANDER_JSON}
        agent.parsed_data_sync(data)
        assert agent.profile.current_stage == "In Battle"

    def test_creates_sub_agent_from_action(self, make_agent):
        agent = make_agent(original=1000)
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
        data = {**VALID_COMMANDER_JSON, "actions": [action]}
        agent.parsed_data_sync(data)
        assert len(agent.hierarchy.sub_agents) == 1
        sub = agent.hierarchy.sub_agents[0]
        assert sub.profile.original_num_of_troops == 200

    def test_target_agent_id_set_on_hierarchy(self, make_agent):
        # Bug #3 from the plan is already fixed: sets self.hierarchy.target_agent_id
        agent = make_agent()
        data = {**VALID_COMMANDER_JSON, "targetedAgentId": "ARMY-TARGET-1"}
        agent.parsed_data_sync(data)
        assert agent.hierarchy.target_agent_id == "ARMY-TARGET-1"

    @pytest.mark.xfail(
        reason="Bug #4: execute() silently returns None after max retries instead of raising"
    )
    def test_execute_raises_after_max_attempts(self, make_agent, monkeypatch):
        agent = make_agent()
        # Always return unparseable text so parse fails every time
        monkeypatch.setattr(agent, "run_model", lambda p: "no json here")
        with pytest.raises(Exception, match="maximum attempts"):
            agent.execute()

    def test_execute_crashes_with_attribute_error_on_unparseable_response(self, make_agent, monkeypatch):
        """Characterization: when LLM returns unparseable text, parse_llm_output returns an
        error string, then validate_parsed_output calls .get() on it → AttributeError crash.
        Bug #4 manifest: the execute() retry loop does not guard against non-dict parse results."""
        agent = make_agent()
        monkeypatch.setattr(agent, "run_model", lambda p: "no json here")
        with pytest.raises(AttributeError):
            agent.execute()
