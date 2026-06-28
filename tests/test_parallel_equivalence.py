"""Phase 4 gate: sequential vs parallel execution mode must be state-identical under scripted
(fake) responses. Parallel mode only hoists the commander LLM call into a batch and computes
vision at start-of-step; with the deterministic fake provider neither affects the outcome, so the
final agent trees must match exactly. Also exercises the batched diary path end-to-end.

Run with: pytest tests/test_parallel_equivalence.py -q
"""
import random

SEED = 42


def _make_sandbox(execution_mode, have_diaries):
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
        "test_poitiers_equivalence",
        "1356-09-19 08:00",
        e_root,
        f_root,
    )
    sb.have_diaries = have_diaries
    sb.continue_run = True
    sb.GPT4V = False
    sb.LLM_MODEL = "fake"
    sb.on_agent_error = "abort"
    sb.execution_mode = execution_mode
    return sb


def _snapshot(root):
    """Serialize an agent tree into a comparable nested structure."""
    def visit(agent):
        # Note: agent ids come from uuid4 (not the seeded `random` module) so they differ on
        # every run regardless of execution mode — excluded from the comparison on purpose.
        p = agent.profile
        return {
            "identity": p.identity,
            "position": list(p.position),
            "remaining": p.remaining_num_of_troops,
            "deployed": p.deployed_num_of_troops,
            "lost": p.lost_num_of_troops,
            "moral": p.moral,
            "stage": p.current_stage,
            "subs": [visit(s) for s in agent.hierarchy.sub_agents],
        }

    return visit(root)


def _run(execution_mode, have_diaries):
    random.seed(SEED)
    sb = _make_sandbox(execution_mode, have_diaries)
    sb.simulate(total_minutes=30, step_minutes=15)
    return (
        _snapshot(sb.country_E_agent_root),
        _snapshot(sb.country_F_agent_root),
        sb.action_interact_evaluation.skipped_casualty_rounds,
    )


def test_sequential_parallel_state_identical_no_diaries():
    seq = _run("sequential", have_diaries=False)
    par = _run("parallel", have_diaries=False)
    assert seq == par


def test_sequential_parallel_state_identical_with_diaries():
    """Diaries are batched in both modes via _post_decision -> _run_diaries; final battle state
    must still match (journals don't feed back into troop state)."""
    seq = _run("sequential", have_diaries=True)
    par = _run("parallel", have_diaries=True)
    assert seq == par
