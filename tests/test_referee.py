"""Tests for the referee module (action interaction evaluation and casualty parsing).

Known bugs tested here:
- Bug #1: external_construct_judgment_prompt uses global map_info_json instead of map_info param
- Bug #2: retry loop never reassigns parsed_json so successful retries are discarded
"""

import pytest
from support_agents.referee import (
    Action_Interact_Evaluation,
    external_action_binding,
    external_construct_judgment_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(make_agent):
    """Create a minimal Action_Interact_Evaluation with no agents."""
    return Action_Interact_Evaluation("gpt", [], [], [], {})


# ---------------------------------------------------------------------------
# parse_llm_output (referee version — returns {} on error, not a string)
# ---------------------------------------------------------------------------

class TestRefereeParseLLMOutput:
    def test_extracts_valid_json(self, make_agent):
        ev = _make_evaluator(make_agent)
        text = 'Assessment: {"casualties_result": [{"agent_id": "A", "casualties": 10}]}'
        result = ev.parse_llm_output(text)
        assert isinstance(result, dict)
        assert "casualties_result" in result

    def test_pure_json_string(self, make_agent):
        ev = _make_evaluator(make_agent)
        text = '{"casualties_result": []}'
        result = ev.parse_llm_output(text)
        assert result == {"casualties_result": []}

    def test_no_json_returns_empty_dict(self, make_agent):
        ev = _make_evaluator(make_agent)
        result = ev.parse_llm_output("No JSON here.")
        assert result == {}

    def test_malformed_json_returns_empty_dict(self, make_agent):
        ev = _make_evaluator(make_agent)
        result = ev.parse_llm_output('{"unclosed":')
        assert result == {}

    def test_empty_string_returns_empty_dict(self, make_agent):
        ev = _make_evaluator(make_agent)
        result = ev.parse_llm_output("")
        assert result == {}

    def test_json_with_trailing_comma(self, make_agent):
        ev = _make_evaluator(make_agent)
        text = '{"casualties_result": [],}'
        result = ev.parse_llm_output(text)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# parsed_data_sync — casualty application
# ---------------------------------------------------------------------------

class TestRefereeParsedDataSync:
    def _evaluator_with_agents(self, make_agent, e_agents, f_agents):
        all_agents = e_agents + f_agents
        ev = Action_Interact_Evaluation("gpt", all_agents, e_agents, f_agents, {})
        return ev

    def test_applies_casualties_to_matching_agent(self, make_agent):
        victim = make_agent(identity="country_F", original=1000)
        ev = self._evaluator_with_agents(make_agent, [], [victim])
        parsed = {"casualties_result": [{"agent_id": victim.hierarchy.id, "casualties": 150}]}
        ev.parsed_data_sync(victim, parsed)
        assert victim.profile.lost_num_of_troops == 150

    def test_string_casualties_converted_to_int(self, make_agent):
        victim = make_agent(identity="country_F", original=1000)
        ev = self._evaluator_with_agents(make_agent, [], [victim])
        parsed = {"casualties_result": [{"agent_id": victim.hierarchy.id, "casualties": "75"}]}
        ev.parsed_data_sync(victim, parsed)
        assert victim.profile.lost_num_of_troops == 75

    def test_non_numeric_casualties_skipped(self, make_agent):
        victim = make_agent(identity="country_F", original=1000)
        ev = self._evaluator_with_agents(make_agent, [], [victim])
        parsed = {"casualties_result": [{"agent_id": victim.hierarchy.id, "casualties": "many"}]}
        msg = ev.parsed_data_sync(victim, parsed)
        assert victim.profile.lost_num_of_troops == 0
        assert "skipped" in msg

    def test_empty_casualties_result(self, make_agent):
        victim = make_agent()
        ev = self._evaluator_with_agents(make_agent, [victim], [])
        msg = ev.parsed_data_sync(victim, {"casualties_result": []})
        assert msg == "No casualties data to process."

    def test_unknown_agent_id_not_applied(self, make_agent):
        victim = make_agent(identity="country_E", original=1000)
        ev = self._evaluator_with_agents(make_agent, [victim], [])
        parsed = {"casualties_result": [{"agent_id": "ARMY-ghost", "casualties": 999}]}
        ev.parsed_data_sync(victim, parsed)
        assert victim.profile.lost_num_of_troops == 0

    def test_multiple_casualty_entries_applied(self, make_agent):
        a = make_agent(identity="country_E", original=500)
        b = make_agent(identity="country_E", original=500)
        ev = self._evaluator_with_agents(make_agent, [a, b], [])
        parsed = {
            "casualties_result": [
                {"agent_id": a.hierarchy.id, "casualties": 50},
                {"agent_id": b.hierarchy.id, "casualties": 30},
            ]
        }
        ev.parsed_data_sync(a, parsed)
        assert a.profile.lost_num_of_troops == 50
        assert b.profile.lost_num_of_troops == 30


# ---------------------------------------------------------------------------
# external_construct_judgment_prompt — map bug (Bug #1)
# ---------------------------------------------------------------------------

class TestRefereePromptMapParam:
    def test_prompt_uses_passed_map_info(self, make_agent):
        """Current code correctly uses the map_info parameter (not the global).
        The improvement plan listed this as Bug #1 referencing an older code version;
        the current referee.py already reads map_info["Geography"] (line 51)."""
        agent = make_agent()
        falkirk_map = {
            "Geography": {
                "Falkirk Marsh": {
                    "coordinates": [0, 50],
                    "description": "A marshy bog unique to Falkirk.",
                }
            }
        }
        prompt = external_construct_judgment_prompt(agent, {}, [], falkirk_map)
        assert "Falkirk Marsh" in prompt

    def test_prompt_reflects_custom_map_geography(self, make_agent):
        agent = make_agent()
        custom_map = {
            "Geography": {
                "Unique Test Location XYZ": {
                    "coordinates": [5, 10],
                    "description": "A unique test landmark.",
                }
            }
        }
        prompt = external_construct_judgment_prompt(agent, {}, [], custom_map)
        assert "Unique Test Location XYZ" in prompt

    def test_weapon_lore_is_hardcoded_for_all_battles(self, make_agent):
        """Bug #1 remnant still present: weapon lore (longbow/mud) is hardcoded in the
        prompt template regardless of the battle — see referee.py line ~90."""
        agent = make_agent()
        map_with_no_weapons = {"Geography": {}}
        prompt = external_construct_judgment_prompt(agent, {}, [], map_with_no_weapons)
        assert "longbow" in prompt


# ---------------------------------------------------------------------------
# Retry loop bug (Bug #2)
# ---------------------------------------------------------------------------

class TestRefereeRetryLoop:
    def test_retry_loop_never_saves_successful_result(self, make_agent, monkeypatch):
        """Characterization of Bug #2: the while loop calls parse_llm_output
        but discards the result, so parsed_json stays {} even after a valid retry."""
        ev = _make_evaluator(make_agent)

        call_count = [0]
        valid_result = {"casualties_result": [{"agent_id": "X", "casualties": 10}]}

        def mock_parse(response):
            call_count[0] += 1
            return {} if call_count[0] == 1 else valid_result

        monkeypatch.setattr(ev, "parse_llm_output", mock_parse)
        monkeypatch.setattr(ev, "run_model", lambda p: "retry response")

        # Replicate the actual retry loop in single_agent_evaluate
        LLM_response = "initial bad response"
        prompt = "test prompt"
        parsed_json = ev.parse_llm_output(LLM_response)  # returns {}

        if parsed_json == {}:
            attempts = 0
            while attempts < 3:
                LLM_response = ev.run_model(prompt)
                if ev.parse_llm_output(LLM_response):  # result discarded — the bug
                    break
                attempts += 1

        # Despite a successful retry, parsed_json is still {} — this is the bug
        assert parsed_json == {}

    @pytest.mark.xfail(
        reason="Bug #2: retry loop should assign parsed_json from successful retry result"
    )
    def test_retry_loop_should_save_successful_result(self, make_agent, monkeypatch):
        """Expected behavior: after successful retry, parsed_json contains the valid data."""
        ev = _make_evaluator(make_agent)

        call_count = [0]
        valid_result = {"casualties_result": [{"agent_id": "X", "casualties": 10}]}

        def mock_parse(response):
            call_count[0] += 1
            return {} if call_count[0] == 1 else valid_result

        monkeypatch.setattr(ev, "parse_llm_output", mock_parse)
        monkeypatch.setattr(ev, "run_model", lambda p: "retry response")

        LLM_response = "initial bad response"
        prompt = "test prompt"
        parsed_json = ev.parse_llm_output(LLM_response)

        if parsed_json == {}:
            attempts = 0
            while attempts < 3:
                LLM_response = ev.run_model(prompt)
                if ev.parse_llm_output(LLM_response):
                    break
                attempts += 1

        # This assertion currently fails because parsed_json is never updated
        assert parsed_json == valid_result


# ---------------------------------------------------------------------------
# external_action_binding
# ---------------------------------------------------------------------------

class TestActionBinding:
    def test_no_target_returns_empty_attacked_list(self, make_agent):
        attacker = make_agent(identity="country_E")
        attacker.hierarchy.target_agent_id = ""
        enemy = make_agent(identity="country_F")
        _, attacked = external_action_binding(attacker, [enemy], [], [attacker, enemy])
        assert attacked == []

    def test_matching_target_id_returns_attacked_agent(self, make_agent):
        attacker = make_agent(identity="country_E")
        enemy = make_agent(identity="country_F")
        attacker.hierarchy.target_agent_id = enemy.hierarchy.id
        _, attacked = external_action_binding(attacker, [enemy], [], [attacker, enemy])
        assert enemy in attacked

    def test_nonexistent_target_id_returns_empty(self, make_agent):
        attacker = make_agent(identity="country_E")
        attacker.hierarchy.target_agent_id = "ARMY-ghost"
        enemy = make_agent(identity="country_F")
        _, attacked = external_action_binding(attacker, [enemy], [], [attacker, enemy])
        assert attacked == []
