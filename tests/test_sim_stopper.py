"""Tests for ceasefire_decision_maker thresholds and hierarchy traversal."""

import pytest
from support_agents.sim_stopper import ceasefire_decision_maker


class TestCeasefireDecisionSingleAgent:
    def test_healthy_agent_yields_no_decision(self, make_agent):
        agent = make_agent(stage="In Battle", original=1000, lost=0)
        results, decision = ceasefire_decision_maker(agent)
        assert decision is None

    def test_crushing_defeat_triggers_command_decision(self, make_agent):
        # "Crushing Defeat" != "In Battle" → command_structure_impact also fires.
        # command_structure_impact is checked before morale_collapse_impact in the code,
        # so even though both thresholds are crossed the command decision wins.
        agent = make_agent(stage="Crushing Defeat", original=1000, lost=0)
        _, decision = ceasefire_decision_maker(agent)
        assert decision == "Breakdown of Command Structure"

    def test_non_in_battle_triggers_command_decision(self, make_agent):
        agent = make_agent(stage="Routing", original=1000, lost=0)
        _, decision = ceasefire_decision_maker(agent)
        assert decision == "Breakdown of Command Structure"

    def test_crushing_defeat_raises_both_thresholds(self, make_agent):
        """Both morale and command thresholds are crossed for Crushing Defeat."""
        agent = make_agent(stage="Crushing Defeat", original=1000, lost=0)
        results, _ = ceasefire_decision_maker(agent)
        assert results["morale_collapse_impact"] > 0.8
        assert results["command_structure_impact"] > 0.8

    def test_low_troops_triggers_heavy_casualties(self, make_agent):
        # remaining_num_of_troops < 50 → heavy_casualties_count >= 1 / total_agents > 0.8
        agent = make_agent(original=100, lost=60)
        agent.profile.current_stage = "In Battle"
        _, decision = ceasefire_decision_maker(agent)
        assert decision == "Heavy Casualties among High-Ranking Soldiers"

    def test_agent_with_50_troops_at_threshold(self, make_agent):
        # exactly 50 remaining → NOT < 50, so no heavy-casualties trigger
        agent = make_agent(original=100, lost=50)
        agent.profile.current_stage = "In Battle"
        _, decision = ceasefire_decision_maker(agent)
        assert decision is None

    def test_agent_with_49_troops_triggers_heavy_casualties(self, make_agent):
        agent = make_agent(original=100, lost=51)
        agent.profile.current_stage = "In Battle"
        _, decision = ceasefire_decision_maker(agent)
        assert decision == "Heavy Casualties among High-Ranking Soldiers"


class TestCeasefireResultStructure:
    def test_results_contain_all_keys(self, make_agent):
        agent = make_agent()
        results, _ = ceasefire_decision_maker(agent)
        for key in ("total_agents", "command_structure_impact", "morale_collapse_impact",
                    "heavy_casualties_count", "total_troops"):
            assert key in results

    def test_total_agents_counts_root_only_when_no_children(self, make_agent):
        agent = make_agent()
        results, _ = ceasefire_decision_maker(agent)
        assert results["total_agents"] == 1

    def test_total_troops_reflects_remaining(self, make_agent):
        agent = make_agent(original=1000, lost=200)
        results, _ = ceasefire_decision_maker(agent)
        # remaining_num_of_troops = 1000 - 0 - 200 = 800
        assert results["total_troops"] == 800


class TestCeasefireHierarchy:
    def test_hierarchy_sums_agents(self, make_agent):
        root = make_agent(original=1000)
        child = make_agent(original=500)
        root.hierarchy.sub_agents.append(child)
        child.hierarchy.parent_agent = root

        results, _ = ceasefire_decision_maker(root)
        assert results["total_agents"] == 2

    def test_healthy_hierarchy_no_decision(self, make_agent):
        root = make_agent(stage="In Battle", original=1000)
        child = make_agent(stage="In Battle", original=500)
        root.hierarchy.sub_agents.append(child)

        _, decision = ceasefire_decision_maker(root)
        assert decision is None

    def test_partial_hierarchy_crushing_defeat_below_threshold(self, make_agent):
        # 1 of 2 agents has Crushing Defeat → morale_collapse_impact = 0.5 < 0.8
        root = make_agent(stage="In Battle", original=1000)
        child = make_agent(stage="Crushing Defeat", original=500)
        root.hierarchy.sub_agents.append(child)

        _, decision = ceasefire_decision_maker(root)
        # 0.5 < 0.8 threshold → no morale decision
        assert decision != "Morale Collapse"

    def test_command_impact_diminishes_with_level(self, make_agent):
        # Level-2 child's impact is halved: 1/1 + 1/2 = 1.5 for 2 agents
        # but command_impact threshold is > 0.8 so both crossing it means decision
        root = make_agent(stage="Routing")   # command_impact += 1/1
        child = make_agent(stage="Routing")  # command_impact += 1/2
        root.hierarchy.sub_agents.append(child)

        results, decision = ceasefire_decision_maker(root)
        assert results["command_structure_impact"] == pytest.approx(1.5)
        assert decision == "Breakdown of Command Structure"

    def test_total_troops_sums_across_hierarchy(self, make_agent):
        root = make_agent(original=1000, lost=0)
        child = make_agent(original=500, lost=100)
        root.hierarchy.sub_agents.append(child)

        results, _ = ceasefire_decision_maker(root)
        # root remaining = 1000, child remaining = 400
        assert results["total_troops"] == 1400
