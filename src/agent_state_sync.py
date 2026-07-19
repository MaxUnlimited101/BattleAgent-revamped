"""Commander state synchronization: applying a parsed decision to agent/hierarchy state.

Extracted from ``Detachment_Agent`` (Phase 6 decomposition). Covers merge/prune
(``BranchStreamlining``), applying a validated decision (``parsed_data_sync``), and spawning
sub-agents (``create_sub_agent``). ``Detachment_Agent`` keeps ``parsed_data_sync`` /
``create_sub_agent`` as facade methods and re-exports ``BranchStreamlining``.

``create_sub_agent`` constructs ``Detachment_Agent`` instances, so the agent classes are imported
lazily (function-local) to avoid an ``agent`` <-> ``agent_state_sync`` import cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Detachment_Agent

logger = logging.getLogger(__name__)


def BranchStreamlining(agent: "Detachment_Agent", target_agent_id: str) -> str:
    logger.info("doing the BranchStreamlining")
    sub_agent_entity = None
    for sub_agent in agent.hierarchy.sub_agents:
        if sub_agent.hierarchy.id == target_agent_id:
            sub_agent_entity = sub_agent
            break

    # Handle the case where the target agent is not found
    if sub_agent_entity is None:
        return "BranchStreamlining stop."
    if sub_agent_entity.profile.moral == "Low":
        streamlining_label = "Prune"
    else:
        streamlining_label = "Merge"

    subsub_agents_list = sub_agent_entity.hierarchy.sub_agents

    if len(subsub_agents_list) > 0:
        for subsub_agent in subsub_agents_list:
            subsub_agent.hierarchy.parent_agent = agent
            agent.hierarchy.sub_agents.append(subsub_agent)

    if streamlining_label == "Merge":
        # Return sub-agent's remaining troops to parent pool; grandchildren become
        # direct sub-agents of parent, so their original counts become parent's deployed.
        agent.profile.deployed_num_of_troops -= sub_agent_entity.profile.original_num_of_troops
        agent.profile.deployed_num_of_troops += sub_agent_entity.profile.deployed_num_of_troops
        agent.profile.lost_num_of_troops += sub_agent_entity.profile.lost_num_of_troops

    elif streamlining_label == "Prune":
        # Sub-agent's remaining (not-yet-deployed) troops are lost; grandchildren
        # survive as reparented children, so count their originals as parent's deployed.
        agent.profile.deployed_num_of_troops -= sub_agent_entity.profile.original_num_of_troops
        agent.profile.deployed_num_of_troops += sub_agent_entity.profile.deployed_num_of_troops
        agent.profile.lost_num_of_troops += (
            sub_agent_entity.profile.original_num_of_troops
            - sub_agent_entity.profile.deployed_num_of_troops
        )

    # disable the subagent
    agent.hierarchy.sub_agents.remove(sub_agent_entity)
    sub_agent_entity.mergedOrPruned = True

    return "agent {agent.hierarchy.id} BranchStreamlining Done"


def parsed_data_sync(agent: "Detachment_Agent", parsed_json: dict) -> list[dict]:
    sync_results: list[dict] = []
    if parsed_json["agentMoral"]:
        agent.profile.moral = parsed_json["agentMoral"]

    if parsed_json["SubAgentsRecall"]:
        for recalled_agent_id in parsed_json["SubAgentsRecall"]:
            BranchStreamlining(agent, recalled_agent_id)
            logger.info("do the BranchStreamlining")

    if agent.profile.position != parsed_json["agentNextPosition"]:
        logger.info("Moved from %s to %s", agent.profile.position, parsed_json['agentNextPosition'])
    else:
        logger.info("position no change")

    if len(parsed_json['agentNextPosition']) == 2 and all(isinstance(item, int) for item in parsed_json['agentNextPosition']):
        agent.profile.position = parsed_json["agentNextPosition"]
        agent.profile.position_updated_hist(agent.profile.round_nb, parsed_json["agentNextPosition"])

    agent.profile.current_action = parsed_json["agentNextActionType"] + " " + parsed_json["remarks"]
    agent.hierarchy.target_agent_id = parsed_json.get("targetedAgentId", "")
    # Process each action in the actions list
    for action in parsed_json["actions"]:
        if len(action['position']) == 2 and all(isinstance(item, int) for item in action['position']):
            new_sub_agent = create_sub_agent(agent, action)

            new_sub_agent.hierarchy.target_agent_id = action["agent_id"]
            new_sub_agent.profile.troopType = action["troopType"]

            sync_results.append({"action": action["subAgent_NextActionType"], "result": "sub_agent_created", "sub_agent": new_sub_agent})
            if "deployedNum" in action and action['deployedNum'] not in ["All available", "All remaining"]:
                agent.profile.deployed_num_of_troops += int(action['deployedNum'])

    # Check for other conditions like Crushing Defeat or fleeing Off the Map
    if (agent.profile.remaining_num_of_troops < agent.profile.original_num_of_troops * agent.profile.crushing_defeat_remaining_frac
            or agent.profile.lost_num_of_troops > agent.profile.original_num_of_troops * agent.profile.crushing_defeat_lost_frac):
        agent.profile.current_stage = "Crushing Defeat"

    if agent.profile.position[0] > 1000 or agent.profile.position[1] > 1000:
        agent.profile.current_stage = "fleeing Off the Map"

    # Emit a structured decision event (no-op if no event logger was wired by the sandbox).
    if agent.event_logger is not None:
        agent.event_logger.decision(
            step=agent.current_step,
            agent_id=agent.hierarchy.id,
            identity=agent.profile.identity,
            action=parsed_json.get("agentNextActionType", ""),
            position=agent.profile.position,
            deployed=agent.profile.deployed_num_of_troops,
            remarks=parsed_json.get("remarks"),
            moral=parsed_json.get("agentMoral"),
            target=parsed_json.get("targetedAgentId"),
        )

    # Return the results of the synchronization
    return sync_results


def create_sub_agent(agent: "Detachment_Agent", action: dict[str, Any]) -> "Detachment_Agent":
    # Lazy import to avoid an agent <-> agent_state_sync import cycle.
    from agent import Detachment_Agent, Detachment_AgentHierarchy, Detachment_AgentProfile

    # Create a new profile based on the current agent's profile
    new_profile = Detachment_AgentProfile(
        identity=agent.profile.identity,
        position=action['position'],
        original_num_of_troops=int(action['deployedNum']) if action['deployedNum'] != "All available" else agent.profile.remaining_num_of_troops,  # set troop count based on action
        initial_mission=action["subAgent_NextActionType"],
        constant_prompt_config=agent.profile.constant_prompt_config,
        max_deploy_percent=agent.profile.max_deploy_percent,
        crushing_defeat_remaining_frac=agent.profile.crushing_defeat_remaining_frac,
        crushing_defeat_lost_frac=agent.profile.crushing_defeat_lost_frac,
        round_interval=agent.profile.round_interval,
    )

    # Create a new hierarchy level for the subunit
    new_hierarchy = Detachment_AgentHierarchy(
        level=agent.hierarchy.level + 1,  # Set the hierarchy level one step deeper
        parent_agent=agent  # Set the current hierarchy as the parent
    )

    # Initialize the new sub-agent
    new_sub_agent = Detachment_Agent(agent.model_type, new_profile, new_hierarchy)
    new_sub_agent.new_born = True
    new_sub_agent.log_folder_name = agent.log_folder_name
    new_sub_agent.parser_mode = agent.parser_mode
    new_sub_agent.token_accumulator = agent.token_accumulator
    new_sub_agent.history_window = agent.history_window
    new_sub_agent.prompt_caching = agent.prompt_caching
    new_sub_agent.event_logger = agent.event_logger
    new_sub_agent.current_step = agent.current_step

    # Add the new sub-agent to the current hierarchy
    agent.hierarchy.add_sub_agent(new_sub_agent)

    return new_sub_agent
