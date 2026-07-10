import argparse
import logging
from dotenv import load_dotenv
load_dotenv()

from prompt.Detachment_Agent_prompt import (
    RoleSetting,
    TroopInformation,
    action_instruction_block,
    json_constraint_variable,
    json_example_text,
)
from prompt.action_space_setting import action_list, action_property_definition

from agent import ConstantPromptConfig, Detachment_Agent, Detachment_AgentHierarchy, Detachment_AgentProfile, reset_agent_ids
from battles import get_battle, BATTLES
from config import SimulationConfig
from sandbox import Sandbox
from utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _make_prompt_config(battle, army_setting):
    return ConstantPromptConfig(
        System_Setting=battle.system_setting,
        History_Setting=battle.history_setting,
        army_setting=army_setting,
        role_setting=RoleSetting,
        troop_information=TroopInformation,
        json_constraint_variable=json_constraint_variable,
        json_example_text=json_example_text,
        action_list=action_list,
        action_property_definition=action_property_definition,
        action_instruction_block=action_instruction_block,
        map_info_json=battle.map_info_json,
        additional_settings={},
    )


def build_root_agents(config, battle):
    """Construct the two root ``Detachment_Agent``s (and their prompt config) for a battle.

    Shared by the CLI entrypoint and ``state_serialization.load_state`` so the reconstruction of a
    saved run matches a fresh run exactly. Callers that need reproducible ids should call
    ``reset_agent_ids()`` before this.
    """
    profile_kwargs = dict(
        max_deploy_percent=config.max_deploy_percent,
        crushing_defeat_remaining_frac=config.crushing_defeat_remaining_frac,
        crushing_defeat_lost_frac=config.crushing_defeat_lost_frac,
        round_interval=config.round_interval,
    )
    country_E_agent_profile = Detachment_AgentProfile(
        identity="country_E",
        position=list(battle.country_E_position),
        original_num_of_troops=battle.country_E_troops,
        constant_prompt_config=_make_prompt_config(battle, battle.country_E_army),
        **profile_kwargs,
    )
    country_F_agent_profile = Detachment_AgentProfile(
        identity="country_F",
        position=list(battle.country_F_position),
        original_num_of_troops=battle.country_F_troops,
        constant_prompt_config=_make_prompt_config(battle, battle.country_F_army),
        **profile_kwargs,
    )

    country_E_agent_hierarchy = Detachment_AgentHierarchy(level=1, parent_agent=None)
    country_F_agent_hierarchy = Detachment_AgentHierarchy(level=1, parent_agent=None)

    country_E_agent_root = Detachment_Agent(config.commander_model, country_E_agent_profile, country_E_agent_hierarchy)
    country_F_agent_root = Detachment_Agent(config.commander_model, country_F_agent_profile, country_F_agent_hierarchy)
    for root in (country_E_agent_root, country_F_agent_root):
        root.parser_mode = config.parser_mode
        root.history_window = config.history_window
        root.prompt_caching = config.prompt_caching

    country_E_agent_hierarchy.parent_agent = country_E_agent_root
    country_F_agent_hierarchy.parent_agent = country_F_agent_root

    return country_E_agent_root, country_F_agent_root


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a conflict simulation.')
    parser.add_argument('--conflict_name', type=str, choices=sorted(BATTLES), default= "Poitiers",help='choose conflict name')
    parser.add_argument('--LLM_MODEL', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default="gpt", help='Language model to use')
    parser.add_argument("--is_GPT4V_activate", type=int, default=0, help="Use GPT-4 V instead of GPT-4")
    parser.add_argument('--simulation_time', type=int, default=90, help='Number of minutes to simulate')
    parser.add_argument('--update_interval', type=int, default=15, help='Interval for simulation updates')
    
    parser.add_argument('--have_diaries', type=int, default=0, help='Whether to have diaries')
    parser.add_argument('--continue_run', type=int, default=1, help='Whether to continue run')
    parser.add_argument('--vision_range', type=int, default=100000, help='Fog-of-war radius in map units (default 100000 = omniscient)')
    parser.add_argument('--on_agent_error', type=str, choices=['continue', 'abort'], default='continue', help='What to do when an agent raises AgentExecutionError')
    parser.add_argument('--snapshot-format', dest='snapshot_format', type=str, choices=['json', 'pickle'], default='json', help='Per-step state snapshot format: json=versioned & resumable, pickle=legacy whole-object')
    parser.add_argument('--parser', type=str, choices=['legacy', 'structured'], default='legacy', help='JSON validation mode: legacy=manual checks, structured=Pydantic models')
    parser.add_argument('--execution-mode', dest='execution_mode', type=str, choices=['sequential', 'parallel'], default='sequential', help='sequential=one agent at a time; parallel=batch commander decisions per step')
    parser.add_argument('--max-concurrency', dest='max_concurrency', type=int, default=8, help='Max concurrent LLM calls per batch (parallel mode and diary batching)')
    parser.add_argument('--history-window', dest='history_window', type=int, default=3, help='Last-K parsed JSON decisions kept in the commander prompt; <=0 = full raw history')
    parser.add_argument('--prompt-caching', dest='prompt_caching', action='store_true', help='Send the static prompt prefix as a cache_control block (Anthropic prompt caching)')
    parser.add_argument('--commander_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for commander agents (defaults to --LLM_MODEL)')
    parser.add_argument('--referee_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for referee (defaults to --LLM_MODEL)')
    parser.add_argument('--diary_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for diary soldiers (defaults to --LLM_MODEL)')

    # Phase 5 tunable constants (defaults reproduce prior hardcoded behavior).
    parser.add_argument('--max_deploy_percent', type=float, default=0.6, help='Max fraction of original troops an agent may deploy')
    parser.add_argument('--crushing_defeat_remaining_frac', type=float, default=0.1, help='Remaining-troop fraction below which an agent is a Crushing Defeat')
    parser.add_argument('--crushing_defeat_lost_frac', type=float, default=0.5, help='Lost-troop fraction above which an agent is a Crushing Defeat')
    parser.add_argument('--sub_agent_threshold', type=int, default=5, help='Sub-agent count that triggers action restrictions')
    parser.add_argument('--diary_injury_prob', type=float, default=0.3, help='Per-diary probability a soldier sustains an injury')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the simulation RNG')

    args = parser.parse_args()

    config = SimulationConfig.from_args(args)
    configure_logging()  # console handler now; the per-run file handler is added by the sandbox
    # Deterministic agent ids: reset the counter before any agent is constructed.
    reset_agent_ids()
    
    GPT4V = config.gpt4v

    if config.llm_model not in ("gpt", "fake") and GPT4V:
        raise ValueError("GPT-4 V is only available for GPT model.")

    model_name_to_log = "gpt4V" if (config.llm_model == "gpt" and GPT4V) else config.llm_model
    LOG_FOLER_NMAE = f"{config.conflict_name}_{model_name_to_log}_{config.simulation_time}_{config.update_interval}"
    logger.info("LOG_FOLER_NMAE: %s", LOG_FOLER_NMAE)

    battle = get_battle(config.conflict_name)

    country_E_agent_root, country_F_agent_root = build_root_agents(config, battle)

    # conflict name and time only used for logging
    sandbox = Sandbox(config.llm_model, battle.map_info_json, LOG_FOLER_NMAE, "1300-01-01 12:00", country_E_agent_root, country_F_agent_root,
                      referee_model=config.referee_model, diary_model=config.diary_model, config=config)

    sandbox.have_diaries = config.have_diaries
    sandbox.continue_run = config.continue_run
    sandbox.GPT4V = GPT4V
    sandbox.LLM_MODEL = config.llm_model
    sandbox.vision_range = config.vision_range
    sandbox.on_agent_error = config.on_agent_error
    sandbox.execution_mode = config.execution_mode
    sandbox.max_concurrency = config.max_concurrency

    if config.execution_mode == "parallel" and GPT4V:
        logger.warning("parallel execution mode is incompatible with GPT4V; falling back to sequential.")

    simulation_results = sandbox.simulate(config.simulation_time, config.update_interval)