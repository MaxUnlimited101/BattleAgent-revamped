"""Round-trip tests for the Phase 6 versioned state serialization.

Builds a small Poitiers sandbox from the battle registry (fake LLM, zero tokens), runs one step,
then asserts that ``load_state(save_state(sandbox))`` faithfully reconstructs the run — both by
re-serializing to an identical snapshot and by spot-checking the agent tree, and by confirming a
subsequent step produces identical state on the reconstructed sandbox.
"""
import json

import pytest


def _build_sandbox():
    from agent import reset_agent_ids
    from battles import get_battle
    from config import SimulationConfig
    from sandbox import Sandbox
    from simulation_controller import build_root_agents

    config = SimulationConfig(
        conflict_name="Poitiers",
        llm_model="fake",
        have_diaries=True,   # exercise collector (soldier journal/injury) serialization too
        continue_run=True,
        on_agent_error="abort",
    ).resolve_models()

    battle = get_battle("Poitiers")
    reset_agent_ids()
    e_root, f_root = build_root_agents(config, battle)

    sb = Sandbox(
        config.llm_model, battle.map_info_json, "test_roundtrip",
        "1356-09-19 08:00", e_root, f_root,
        referee_model=config.referee_model, diary_model=config.diary_model, config=config,
    )
    sb.have_diaries = config.have_diaries
    sb.continue_run = config.continue_run
    sb.GPT4V = config.gpt4v
    sb.LLM_MODEL = config.llm_model
    sb.on_agent_error = config.on_agent_error
    return sb


def _agent_signature(agent):
    """A structural fingerprint of one agent subtree (order-preserving)."""
    return {
        "id": agent.hierarchy.id,
        "level": agent.hierarchy.level,
        "position": list(agent.profile.position),
        "original": agent.profile.original_num_of_troops,
        "deployed": agent.profile.deployed_num_of_troops,
        "lost": agent.profile.lost_num_of_troops,
        "moral": agent.profile.moral,
        "stage": agent.profile.current_stage,
        "target": agent.hierarchy.target_agent_id,
        "sub_agents": [_agent_signature(s) for s in agent.hierarchy.sub_agents],
    }


def _normalize(obj):
    """Round-trip through JSON so tuples/int-keys compare equal to their reloaded form."""
    return json.loads(json.dumps(obj, default=str))


def test_snapshot_reload_is_exact(tmp_path):
    from state_serialization import SCHEMA_VERSION, load_state, save_state, to_state

    sb = _build_sandbox()
    sb.simulate(total_minutes=15, step_minutes=15)  # one step

    snap_path = str(tmp_path / "snap.state.json")
    save_state(sb, snap_path, fmt="json")

    with open(snap_path) as f:
        saved = json.load(f)
    assert saved["schema_version"] == SCHEMA_VERSION

    sb2 = load_state(snap_path)

    # Re-serializing the reconstructed sandbox must yield the identical snapshot.
    assert _normalize(to_state(sb2)) == saved

    # And the agent trees must match structurally.
    assert _agent_signature(sb2.country_E_agent_root) == _agent_signature(sb.country_E_agent_root)
    assert _agent_signature(sb2.country_F_agent_root) == _agent_signature(sb.country_F_agent_root)

    # Scalars restored.
    assert sb2.current_step == sb.current_step
    assert sb2.system_time == sb.system_time


def test_reloaded_sandbox_continues_identically(tmp_path):
    import random

    from state_serialization import load_state, save_state

    sb = _build_sandbox()
    sb.simulate(total_minutes=15, step_minutes=15)

    snap_path = str(tmp_path / "snap.state.json")
    save_state(sb, snap_path, fmt="json")

    # load_state restores global RNG to the snapshot state; capture it so both sandboxes
    # advance their next step from the same RNG.
    sb2 = load_state(snap_path)
    rng_at_snapshot = random.getstate()

    sb2.simulate(total_minutes=15, step_minutes=15)
    sig2 = [_agent_signature(sb2.country_E_agent_root), _agent_signature(sb2.country_F_agent_root)]

    random.setstate(rng_at_snapshot)
    sb.simulate(total_minutes=15, step_minutes=15)
    sig1 = [_agent_signature(sb.country_E_agent_root), _agent_signature(sb.country_F_agent_root)]

    assert sig2 == sig1


def test_load_rejects_unknown_schema_version(tmp_path):
    from state_serialization import load_state

    bad = tmp_path / "bad.state.json"
    bad.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError):
        load_state(str(bad))
