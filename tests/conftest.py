import os

# Must be set before any import that triggers langchain or agent modules
os.environ.setdefault("OPENAI_API_KEY", "test-key-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-dummy")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-dummy")

import pytest
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# A minimal valid commander JSON — passes validate_parsed_output unchanged.
# Tests that need specific fields can copy and update this dict.
# ---------------------------------------------------------------------------
VALID_COMMANDER_JSON = {
    "agentMoral": "Medium",
    "SubAgentsRecall": [],
    "agentNextPosition": [10, 20],
    "agentNextActionType": "Advance",
    "remarks": "Moving forward",
    "targetedAgentId": "",
    "speed": 5,
    "deploySubUnit": False,
    "actions": [],
}


def _make_config():
    """Minimal ConstantPromptConfig substitute — just needs the right attrs."""
    return SimpleNamespace(
        System_Setting="",
        History_Setting="",
        army_setting="",
        role_setting="",
        troop_information="",
        json_constraint_variable="",
        json_example_text="",
        action_list=[],
        action_property_definition={},
        action_instruction_block="",
        map_info_json={},
        additional_settings="",
    )


@pytest.fixture
def minimal_config():
    return _make_config()


@pytest.fixture
def make_agent():
    """Factory fixture — returns a callable that creates Detachment_Agent instances.

    Avoids going through the full prompt-config stack by passing a SimpleNamespace
    that satisfies all attribute reads in Detachment_AgentProfile.__init__.
    """
    from agent import Detachment_Agent, Detachment_AgentProfile, Detachment_AgentHierarchy

    def _factory(
        identity="country_E",
        position=None,
        original=1000,
        deployed=0,
        lost=0,
        moral="Medium",
        stage="In Battle",
        level=1,
        parent=None,
    ):
        config = _make_config()
        profile = Detachment_AgentProfile(
            identity=identity,
            position=list(position or [0, 0]),
            original_num_of_troops=original,
            constant_prompt_config=config,
        )
        # Override defaults set in __init__
        profile.deployed_num_of_troops = deployed
        profile.lost_num_of_troops = lost
        profile.moral = moral
        profile.current_stage = stage
        profile.round_nb = 0

        hierarchy = Detachment_AgentHierarchy(level=level, parent_agent=parent)
        agent = Detachment_Agent("gpt", profile, hierarchy)
        agent.new_born = False
        return agent

    return _factory
