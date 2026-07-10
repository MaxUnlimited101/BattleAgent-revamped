"""Commander prompt construction.

Extracted from ``Detachment_Agent`` (Phase 6 decomposition). These functions take the agent as
their first argument and read/mutate only its prompt-relevant attributes (``profile``,
``hierarchy``, ``history_window``, ``_static_prefix_cache``, history dicts, retry feedback,
``additional_prompt``). ``Detachment_Agent`` keeps the same method names as thin facades that
delegate here, so callers and tests are unaffected.

The static/dynamic split (and the memoized static prefix) exists for Phase 4 prompt caching and
batching — see ``prompt_parts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json5 as json

from procoder.functional import format_prompt
from procoder.prompt import Sequential, sharp2_indexing

if TYPE_CHECKING:
    from agent import Detachment_Agent


def generate_action_list(agent: "Detachment_Agent") -> str:
    actions_text = '\n'.join([
        f"{action}"
        for action, properties in agent.profile.action_property_definition.items()
    ])
    return actions_text


def static_prefix(agent: "Detachment_Agent") -> str:
    """The invariant leading block of the prompt (battle lore, army, action space, json
    contract, mission, map). Has no per-step interpolation, so it is rendered once and
    memoized — this is also the contiguous prefix marked for Anthropic prompt caching.

    Note: System_Setting (round number), RoleSetting (position/sub-agents) and
    TroopInformation (troop counts) interpolate per-step state, so they live in the
    dynamic suffix, not here.
    """
    if agent._static_prefix_cache is not None:
        return agent._static_prefix_cache

    actions_text = generate_action_list(agent)
    prefix = format_prompt(
        Sequential(
            agent.profile.history_setting,
            agent.profile.army_setting,
            agent.profile.action_instruction_block,
            agent.profile.json_constraint_variable,
        ).set_sep("\n\n").set_indexing_method(sharp2_indexing),
        {"profile": agent.profile, "hierarchy": agent.hierarchy},
    )
    prefix += "\n\n" + "## initial mission\n" + "initial mission refers to the first task or objective assigned to the country_E Commander at the start of the game.\n" + agent.profile.initial_mission
    prefix += "\n\n" + f"## battle field infomation\n This JSON data describes a map of Battle field.  It includes geographic features, military movements, and other relevant details.\n {agent.profile.map_info_json}. The map's dimensions range from -1000 to +1000. Going beyond this range (leaving the map's boundaries) is considered a defeat of this agent."
    prefix += "\n\n" + "## Action Space\n" + "In this simulation, you can use the following actions.\n" + actions_text

    agent._static_prefix_cache = prefix
    return prefix


def history_text(agent: "Detachment_Agent") -> str:
    """Returns the history block content, truncated to the last ``history_window`` decisions.

    ``history_window > 0`` -> the last K parsed JSON decisions (compact). ``<= 0`` -> the
    legacy full raw-response dump. Full history is always retained in
    ``agent.LLM_response_history`` regardless.
    """
    if agent.history_window and agent.history_window > 0:
        recent = sorted(agent.extracted_json_history.items())[-agent.history_window:]
        return json.dumps({str(nb): decision for nb, decision in recent})
    return f"{agent.LLM_response_history}"


def dynamic_suffix(agent: "Detachment_Agent") -> str:
    """The per-step portion of the prompt: framing/role/troop state plus the live
    battlefield situation, retry feedback and truncated history."""
    suffix = format_prompt(
        Sequential(
            agent.profile.System_Setting,
            agent.profile.RoleSetting,
            agent.profile.TroopInformation,
        ).set_sep("\n\n").set_indexing_method(sharp2_indexing),
        {"profile": agent.profile, "hierarchy": agent.hierarchy},
    )

    ## assemble the invalid messages
    flat_invalid_messages = [message for sublist in agent.round_invalid_messages for message in sublist]
    invalid_messages_str = "\n".join(flat_invalid_messages)
    suffix += "\n\n" + invalid_messages_str

    suffix += "\n\n" + "## History Action Plan\n" + f"This is your analysis and planning from previous rounds. It will assist you in determining the stage of the war, helping you make significant decisions.\n {history_text(agent)}"

    suffix += "\n\n" + "## Current Battlefield Situation\n" + f"This is what's happening around you. You can discern the number of enemies and allies, as well as their current actions, within a certain range.\n {agent.profile.CurrentBattlefieldSituation}"

    suffix += "\n\n" + "War is on the verge of breaking out. To initiate an attack on the enemy, based on the speed you've estimated, you and your deployed sub-agents will advance towards the enemy, navigating by the coordinates."

    if agent.action_restrictions_require:
        suffix += "\n\n" + "## Restriction on Sub-Agent Deployment\n" + "Limitation: Due to previous extensive deployment of sub-agents, your current strategy must be confined to your main force, requiring the 'actions' array in the output JSON to include only one action specifically for your command. This constraint prohibits dispatching subsidiary agents and ensures a singular, focused action directive."

    suffix += "\n\n" + agent.additional_prompt
    return suffix


def prompt_parts(agent: "Detachment_Agent") -> tuple[str, str]:
    """Returns (static_prefix, dynamic_suffix). Callers that batch or cache use these
    separately; ``construct_prompt`` joins them for the plain string path."""
    return static_prefix(agent), dynamic_suffix(agent)


def construct_prompt(agent: "Detachment_Agent") -> str:
    prefix, suffix = prompt_parts(agent)
    return prefix + "\n\n" + suffix
