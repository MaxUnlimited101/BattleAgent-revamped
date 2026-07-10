"""Phase 5 golden-file snapshot tests.

These guard the config/registry/persona refactor: if externalizing profiles or introducing the
battle registry changed any rendered text or battle data, one of these fails. Regenerate goldens
after an *intentional* change with:  REGEN_GOLDEN=1 pytest tests/test_prompt_snapshots.py
"""
import json
import os
import random

import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "data", "golden")
REGEN = os.environ.get("REGEN_GOLDEN") == "1"


def _check_or_write(name, content):
    path = os.path.join(GOLDEN_DIR, name)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        pytest.skip(f"regenerated golden {name}")
    with open(path, "r", encoding="utf-8") as f:
        expected = f.read()
    assert content == expected, f"{name} drifted from golden; rerun with REGEN_GOLDEN=1 if intended"


# --------------------------------------------------------------------------- persona parity


def test_soldier_profiles_match_pre_refactor_snapshot():
    """Externalized JSON personas must reproduce the pre-refactor dicts exactly."""
    from group_experience.individual_profile import load_soldier_profiles

    with open(os.path.join(GOLDEN_DIR, "soldier_profiles_snapshot.json"), encoding="utf-8") as f:
        snapshot = json.load(f)

    loaded = load_soldier_profiles()
    for country in ("country_E", "country_F"):
        assert len(loaded[country]) == len(snapshot[country]) == 15
        for sp, orig in zip(loaded[country], snapshot[country]):
            assert sp.name == orig["Name"]
            assert sp.age == orig["Age"]
            assert sp.family == orig["Family"]
            assert sp.personality == orig["Personality"]
            assert sp.secrets_or_scandals == orig["Secrets or Scandals"]


# --------------------------------------------------------------------------- battle registry parity


def test_battle_registry_parity_with_legacy_values():
    """The registry must reproduce the exact positions/troops the old if/elif block used."""
    from battles import get_battle

    expected = {
        "Poitiers": ([15, -10], 6000, [-10, 5], 15000),
        "Falkirk": ([0, 0], 15000, [50, 0], 6000),
        "Agincourt": ([0, -100], 6500, [15, -50], 35000),
    }
    for name, (e_pos, e_troops, f_pos, f_troops) in expected.items():
        b = get_battle(name)
        assert b.country_E_position == e_pos
        assert b.country_E_troops == e_troops
        assert b.country_F_position == f_pos
        assert b.country_F_troops == f_troops
        assert isinstance(b.map_info_json, dict)


def test_registry_includes_crecy_fourth_battle():
    from battles import BATTLES, get_battle

    assert "Crecy" in BATTLES
    assert get_battle("Crecy").country_E_position == [0, 0]


def test_get_battle_unknown_raises():
    from battles import get_battle

    with pytest.raises(ValueError):
        get_battle("Waterloo")


# --------------------------------------------------------------------------- config defaults


def test_simulation_config_defaults():
    from config import SimulationConfig

    c = SimulationConfig()
    assert c.vision_range == 100000
    assert c.max_deploy_percent == 0.6
    assert c.crushing_defeat_remaining_frac == 0.1
    assert c.crushing_defeat_lost_frac == 0.5
    assert c.sub_agent_threshold == 5
    assert c.diary_injury_prob == 0.3
    assert c.seed == 42
    assert c.round_interval == 15


def test_config_resolve_models():
    from config import SimulationConfig

    c = SimulationConfig(llm_model="fake").resolve_models()
    assert c.commander_model == c.referee_model == c.diary_model == "fake"


# --------------------------------------------------------------------------- prompt goldens


def _build_commander_prompt():
    from battles import get_battle
    from agent import (
        ConstantPromptConfig,
        Detachment_Agent,
        Detachment_AgentHierarchy,
        Detachment_AgentProfile,
        reset_agent_ids,
    )
    from prompt.Detachment_Agent_prompt import (
        RoleSetting,
        TroopInformation,
        action_instruction_block,
        json_constraint_variable,
        json_example_text,
    )
    from prompt.action_space_setting import action_list, action_property_definition

    reset_agent_ids()
    b = get_battle("Poitiers")
    cfg = ConstantPromptConfig(
        System_Setting=b.system_setting,
        History_Setting=b.history_setting,
        army_setting=b.country_E_army,
        role_setting=RoleSetting,
        troop_information=TroopInformation,
        json_constraint_variable=json_constraint_variable,
        json_example_text=json_example_text,
        action_list=action_list,
        action_property_definition=action_property_definition,
        action_instruction_block=action_instruction_block,
        map_info_json=b.map_info_json,
        additional_settings={},
    )
    profile = Detachment_AgentProfile(
        identity="country_E", position=[15, -10], original_num_of_troops=6000,
        constant_prompt_config=cfg,
    )
    profile.round_nb = 1
    hierarchy = Detachment_AgentHierarchy(level=1, parent_agent=None)
    agent = Detachment_Agent("fake", profile, hierarchy)
    hierarchy.parent_agent = agent
    agent.profile.CurrentBattlefieldSituation = "Enemy forces spotted near [0, 0]."
    return agent.construct_prompt()


def test_commander_prompt_golden():
    _check_or_write("commander_prompt_poitiers.txt", _build_commander_prompt())


def _build_soldier_prompt():
    from group_experience.individual_profile import load_soldier_profiles

    soldier = load_soldier_profiles()["country_E"][0]
    soldier.injury_list = []
    random.seed(0)  # make injury_generator deterministic
    return soldier.construct_prompt("Hold the line at the ridge.", "Muddy field, rain, enemy advancing.")


def test_soldier_prompt_golden():
    _check_or_write("soldier_prompt_country_E_01.txt", _build_soldier_prompt())
