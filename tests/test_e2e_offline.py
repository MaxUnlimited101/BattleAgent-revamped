"""Offline end-to-end gate: 2-agent Poitiers sandbox, 2 steps, zero API tokens.

Run with: pytest tests/test_e2e_offline.py -q
This test must stay green after every phase.
"""
import os

import pytest


def _make_sandbox():
    from prompt.map_setting_of_other_battles import map_info_json_Poitiers
    from prompt.agent_profile_Poitiers import (
        country_E_Army_Poitiers,
        country_F_Army_Poitiers,
        System_Setting_Poitiers,
        History_Setting_Poitiers,
    )
    from agent import (
        ConstantPromptConfig,
        Detachment_Agent,
        Detachment_AgentHierarchy,
        Detachment_AgentProfile,
    )
    from prompt.Detachment_Agent_prompt import (
        RoleSetting,
        TroopInformation,
        action_instruction_block,
        json_constraint_variable,
        json_example_text,
    )
    from prompt.action_space_setting import action_list, action_property_definition
    from sandbox import Sandbox

    def _config(army):
        return ConstantPromptConfig(
            System_Setting=System_Setting_Poitiers,
            History_Setting=History_Setting_Poitiers,
            army_setting=army,
            role_setting=RoleSetting,
            troop_information=TroopInformation,
            json_constraint_variable=json_constraint_variable,
            json_example_text=json_example_text,
            action_list=action_list,
            action_property_definition=action_property_definition,
            action_instruction_block=action_instruction_block,
            map_info_json=map_info_json_Poitiers,
            additional_settings={},
        )

    e_profile = Detachment_AgentProfile(
        identity="country_E",
        position=[15, -10],
        original_num_of_troops=100,
        constant_prompt_config=_config(country_E_Army_Poitiers),
    )
    f_profile = Detachment_AgentProfile(
        identity="country_F",
        position=[-10, 5],
        original_num_of_troops=150,
        constant_prompt_config=_config(country_F_Army_Poitiers),
    )

    e_hier = Detachment_AgentHierarchy(level=1, parent_agent=None)
    f_hier = Detachment_AgentHierarchy(level=1, parent_agent=None)

    e_root = Detachment_Agent("fake", e_profile, e_hier)
    f_root = Detachment_Agent("fake", f_profile, f_hier)
    e_hier.parent_agent = e_root
    f_hier.parent_agent = f_root

    sb = Sandbox(
        "fake",
        map_info_json_Poitiers,
        "test_poitiers_offline",
        "1356-09-19 08:00",
        e_root,
        f_root,
    )
    sb.have_diaries = False
    sb.continue_run = True
    sb.GPT4V = False
    sb.LLM_MODEL = "fake"
    sb.on_agent_error = "abort"
    return sb


def _check_troop_conservation(root):
    def visit(agent):
        remaining = agent.profile.remaining_num_of_troops
        assert remaining >= 0, (
            f"Agent {agent.hierarchy.id} has negative remaining troops: {remaining}"
        )
        assert agent.profile.original_num_of_troops == (
            remaining
            + agent.profile.deployed_num_of_troops
            + agent.profile.lost_num_of_troops
        ), f"Conservation violated for {agent.hierarchy.id}"
        for sub in agent.hierarchy.sub_agents:
            visit(sub)

    visit(root)


def test_e2e_poitiers_2steps_no_api():
    """Run 2 steps of Poitiers with fake LLM provider; assert correctness invariants."""
    sb = _make_sandbox()

    results = sb.simulate(total_minutes=30, step_minutes=15)

    assert len(results) == 2, f"Expected 2 steps, got {len(results)}"

    _check_troop_conservation(sb.country_E_agent_root)
    _check_troop_conservation(sb.country_F_agent_root)

    log_dir = sb.battle_logger.log_directory_path
    assert os.path.isdir(log_dir), f"Log directory not created: {log_dir}"
    assert len(os.listdir(log_dir)) > 0, "No log files written"

    assert sb.action_interact_evaluation.skipped_casualty_rounds == 0
