"""Versioned simulation-state serialization (Phase 6).

Replaces the old whole-object ``pickle.dump(sandbox)`` with a plain-JSON, schema-versioned
snapshot that can be reloaded to resume a run. Only the *dynamic* state is serialized (agent-tree
troop counts/positions/stages/histories, soldier journals & injuries, sim clock, RNG state). The
*static* scaffolding (procoder prompt objects, battle map, army lore) is NOT serialized — it is
rebuilt from the battle registry via :func:`simulation_controller.build_root_agents`, so a reloaded
run matches a fresh run exactly.

The legacy pickle path is kept behind ``fmt="pickle"`` for one release so existing ``.pkl``
inspection tooling still works.

Top-level imports are kept minimal (no ``sandbox``/``agent`` imports) to avoid an import cycle —
``sandbox`` imports :func:`save_state` at module load, and the heavier reconstruction imports here
are done lazily inside :func:`load_state`.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sandbox import Sandbox

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Mutable profile fields captured per agent. Static/derived fields (prompt settings, map,
# prompt_max_deploy_nb, etc.) are rebuilt by the constructor from the ones below.
_PROFILE_FIELDS = (
    "identity",
    "position",
    "init_position",
    "deployed_num_of_troops",
    "lost_num_of_troops",
    "original_num_of_troops",
    "moral",
    "current_stage",
    "round_nb",
    "round_interval",
    "current_action",
    "troopType",
    "initial_mission",
    "CurrentBattlefieldSituation",
    "max_deploy_percent",
    "crushing_defeat_remaining_frac",
    "crushing_defeat_lost_frac",
)

# Mutable agent-level (non-profile) fields.
_AGENT_FIELDS = (
    "execute_nb",
    "mergedOrPruned",
    "action_restrictions_require",
    "additional_prompt",
    "history_window",
    "prompt_caching",
    "parser_mode",
    "model_type",
    "log_folder_name",
)


# --------------------------------------------------------------------------------------
# Serialization (save side)
# --------------------------------------------------------------------------------------

def _serialize_agent(agent: Any) -> dict:
    """Depth-first serialize one agent node into a JSON-safe dict."""
    profile = {f: getattr(agent.profile, f) for f in _PROFILE_FIELDS}
    # datetime / int-keyed dicts need explicit handling for JSON.
    agent_clock = agent.profile.agent_clock
    profile["agent_clock"] = agent_clock.isoformat() if agent_clock is not None else None
    profile["position_hist_dict"] = {str(k): v for k, v in agent.profile.position_hist_dict.items()}
    profile["history_board"] = agent.profile.history_board

    node = {f: getattr(agent, f) for f in _AGENT_FIELDS}
    node["id"] = agent.hierarchy.id
    node["level"] = agent.hierarchy.level
    node["target_agent_id"] = agent.hierarchy.target_agent_id
    node["profile"] = profile
    # Int-keyed history dicts -> string keys for JSON.
    node["extracted_json_history"] = {str(k): v for k, v in agent.extracted_json_history.items()}
    node["LLM_response_history"] = {str(k): v for k, v in agent.LLM_response_history.items()}
    node["invalid_messages_history"] = agent.invalid_messages_history
    node["round_invalid_messages"] = agent.round_invalid_messages
    node["sub_agents"] = [_serialize_agent(sub) for sub in agent.hierarchy.sub_agents]
    return node


def _serialize_collector(collector: Any) -> list[dict]:
    """Serialize the mutable soldier state (journals, injuries, followed troop)."""
    soldiers = []
    for soldier in collector.soldier_agents_list:
        current_troop = soldier.hierarchy.current_troop
        # Journal is keyed by the sim clock (datetime); stringify keys for JSON.
        journal = {str(k): v for k, v in soldier.profile.journal.items()}
        soldiers.append({
            "name": soldier.profile.name,
            "journal": journal,
            "injury_list": soldier.profile.injury_list,
            "current_troop_id": current_troop.hierarchy.id if current_troop is not None else None,
        })
    return soldiers


def to_state(sandbox: "Sandbox") -> dict:
    """Build a JSON-serializable, schema-versioned snapshot of a running sandbox."""
    config = sandbox.config.to_dict() if sandbox.config is not None else None
    rng_version, rng_state, rng_gauss = random.getstate()
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "conflict_name": sandbox.config.conflict_name if sandbox.config is not None else sandbox.campaign_name,
        "model_type": sandbox.model_type,
        "system_time": sandbox.system_time.isoformat(),
        "current_step": sandbox.current_step,
        "rng_state": [rng_version, list(rng_state), rng_gauss],
        "agents": {
            "country_E": _serialize_agent(sandbox.country_E_agent_root),
            "country_F": _serialize_agent(sandbox.country_F_agent_root),
        },
        "collectors": {
            "country_E": _serialize_collector(sandbox.country_E_collector),
            "country_F": _serialize_collector(sandbox.country_F_collector),
        },
    }


def save_state(sandbox: "Sandbox", path: str, fmt: str = "json") -> str:
    """Persist ``sandbox`` state to ``path``.

    ``fmt="json"`` writes the versioned snapshot (default, resumable). ``fmt="pickle"`` writes the
    legacy whole-object dump. Returns the path written.
    """
    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(sandbox, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_state(sandbox), f, indent=2, default=str)
    logger.info("Sandbox state saved to %s (format=%s)", path, fmt)
    return path


# --------------------------------------------------------------------------------------
# Reconstruction (load side)
# --------------------------------------------------------------------------------------

def _restore_profile(profile: Any, data: dict) -> None:
    from datetime import datetime

    for f in _PROFILE_FIELDS:
        setattr(profile, f, data[f])
    profile.agent_clock = datetime.fromisoformat(data["agent_clock"]) if data["agent_clock"] else None
    profile.position_hist_dict = {int(k): v for k, v in data["position_hist_dict"].items()}
    profile.history_board = data["history_board"]
    # Keep derived prompt fields consistent with restored troop count.
    profile.prompt_max_deploy_percent = profile.max_deploy_percent * 100
    profile.prompt_max_deploy_nb = int(profile.original_num_of_troops * profile.max_deploy_percent)


def _restore_agent_fields(agent: Any, node: dict) -> None:
    for f in _AGENT_FIELDS:
        setattr(agent, f, node[f])
    agent.hierarchy.id = node["id"]
    agent.hierarchy.level = node["level"]
    agent.hierarchy.target_agent_id = node["target_agent_id"]
    agent.extracted_json_history = {int(k): v for k, v in node["extracted_json_history"].items()}
    agent.LLM_response_history = {int(k): v for k, v in node["LLM_response_history"].items()}
    agent.invalid_messages_history = node["invalid_messages_history"]
    agent.round_invalid_messages = node["round_invalid_messages"]


def _rebuild_children(parent_agent: Any, node: dict, prompt_config: Any, model_type: str) -> None:
    """Recursively rebuild sub-agents of ``parent_agent`` from serialized ``node['sub_agents']``."""
    from agent import Detachment_Agent, Detachment_AgentHierarchy, Detachment_AgentProfile

    for child in node["sub_agents"]:
        pdata = child["profile"]
        profile = Detachment_AgentProfile(
            identity=pdata["identity"],
            position=pdata["position"],
            original_num_of_troops=pdata["original_num_of_troops"],
            initial_mission=pdata["initial_mission"],
            constant_prompt_config=prompt_config,
            max_deploy_percent=pdata["max_deploy_percent"],
            crushing_defeat_remaining_frac=pdata["crushing_defeat_remaining_frac"],
            crushing_defeat_lost_frac=pdata["crushing_defeat_lost_frac"],
            round_interval=pdata["round_interval"],
        )
        _restore_profile(profile, pdata)

        hierarchy = Detachment_AgentHierarchy(level=child["level"], parent_agent=parent_agent)
        child_agent = Detachment_Agent(model_type, profile, hierarchy)
        _restore_agent_fields(child_agent, child)
        parent_agent.hierarchy.sub_agents.append(child_agent)

        _rebuild_children(child_agent, child, prompt_config, model_type)


def _restore_collector(collector: Any, soldiers: list[dict], agents_by_id: dict) -> None:
    for soldier, data in zip(collector.soldier_agents_list, soldiers):
        soldier.profile.journal = data["journal"]
        soldier.profile.injury_list = data["injury_list"]
        troop = agents_by_id.get(data["current_troop_id"])
        if troop is not None:
            soldier.hierarchy.current_troop = troop
            soldier.hierarchy.sub_troop = list(troop.hierarchy.sub_agents)


def _index_agents(agent: Any, index: dict) -> None:
    index[agent.hierarchy.id] = agent
    for sub in agent.hierarchy.sub_agents:
        _index_agents(sub, index)


def load_state(path: str) -> "Sandbox":
    """Reconstruct a resumable :class:`~sandbox.Sandbox` from a JSON snapshot written by
    :func:`save_state`."""
    from agent import reset_agent_ids
    from battles import get_battle
    from config import SimulationConfig
    from sandbox import Sandbox
    from simulation_controller import build_root_agents

    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported snapshot schema_version {state.get('schema_version')}; expected {SCHEMA_VERSION}")

    config = SimulationConfig(**state["config"]) if state["config"] is not None else SimulationConfig()
    battle = get_battle(state["conflict_name"])

    # Build fresh static scaffolding, then overlay the restored dynamic state.
    reset_agent_ids()
    country_E_root, country_F_root = build_root_agents(config, battle)

    for root, side in ((country_E_root, "country_E"), (country_F_root, "country_F")):
        node = state["agents"][side]
        _restore_profile(root.profile, node["profile"])
        _restore_agent_fields(root, node)
        _rebuild_children(root, node, root.profile.constant_prompt_config, state["model_type"])

    sandbox = Sandbox(
        state["model_type"], battle.map_info_json, battle.name if hasattr(battle, "name") else state["conflict_name"],
        "1300-01-01 12:00", country_E_root, country_F_root,
        referee_model=config.referee_model, diary_model=config.diary_model, config=config,
    )

    # Mirror the CLI's post-construction wiring so a resumed run behaves identically.
    sandbox.have_diaries = config.have_diaries
    sandbox.continue_run = config.continue_run
    sandbox.GPT4V = config.gpt4v
    sandbox.LLM_MODEL = config.llm_model
    sandbox.vision_range = config.vision_range
    sandbox.on_agent_error = config.on_agent_error
    sandbox.execution_mode = config.execution_mode
    sandbox.max_concurrency = config.max_concurrency

    from datetime import datetime
    sandbox.system_time = datetime.fromisoformat(state["system_time"])
    sandbox.current_step = state["current_step"]
    sandbox.update_all_agent_lists()

    # Restore soldier state and re-link each soldier to the agent it follows.
    agents_by_id: dict = {}
    _index_agents(country_E_root, agents_by_id)
    _index_agents(country_F_root, agents_by_id)
    _restore_collector(sandbox.country_E_collector, state["collectors"]["country_E"], agents_by_id)
    _restore_collector(sandbox.country_F_collector, state["collectors"]["country_F"], agents_by_id)

    # Continue deterministic id assignment past the highest restored id, then restore RNG state.
    _bump_agent_id_counter(agents_by_id)
    rng_version, rng_state, rng_gauss = state["rng_state"]
    random.setstate((rng_version, tuple(rng_state), rng_gauss))

    return sandbox


def _bump_agent_id_counter(agents_by_id: dict) -> None:
    """Advance the global agent-id counter past every restored id so new sub-agents don't collide."""
    import itertools

    import agent as agent_module

    max_n = 0
    for agent_id in agents_by_id:
        try:
            max_n = max(max_n, int(str(agent_id).split("-")[-1]))
        except (ValueError, IndexError):
            continue
    agent_module._agent_id_counter = itertools.count(max_n + 1)
