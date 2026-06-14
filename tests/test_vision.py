"""Tests for vision/distance math (shared_func.py).

Bug #6: fog of war is dead code — AgentInfoCollector default threshold_distance=100000
covers the entire ±1000 map, making all agents omniscient.
"""

import pytest
from utils.shared_func import AgentInfoCollector, vision_filter


class TestVisionFilter:
    def test_active_agent_is_visible(self, make_agent):
        agent = make_agent(stage="In Battle", original=100)
        assert vision_filter(agent) is True

    def test_merged_agent_invisible(self, make_agent):
        agent = make_agent()
        agent.mergedOrPruned = True
        assert vision_filter(agent) is False

    def test_crushing_defeat_invisible(self, make_agent):
        agent = make_agent(stage="Crushing Defeat")
        assert vision_filter(agent) is False

    def test_fleeing_off_map_invisible(self, make_agent):
        agent = make_agent(stage="fleeing Off the Map")
        assert vision_filter(agent) is False

    def test_dead_agent_invisible(self, make_agent):
        # remaining_num_of_troops = original - deployed - lost = 0
        agent = make_agent(original=100, lost=100)
        assert vision_filter(agent) is False

    def test_active_with_many_troops_visible(self, make_agent):
        agent = make_agent(stage="In Battle", original=5000)
        assert vision_filter(agent) is True


class TestAgentInfoCollector:
    def test_self_excluded_from_results(self, make_agent):
        requester = make_agent(position=[0, 0])
        all_agents = [requester]
        result = AgentInfoCollector(all_agents, requester)
        assert result["friendly"] == []
        assert result["enemy"] == []

    def test_friendly_agent_in_friendly_list(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        ally = make_agent(identity="country_E", position=[10, 0])
        result = AgentInfoCollector([requester, ally], requester)
        assert len(result["friendly"]) == 1
        assert result["friendly"][0]["agent_id"] == ally.hierarchy.id

    def test_enemy_agent_in_enemy_list(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        enemy = make_agent(identity="country_F", position=[30, 0])
        result = AgentInfoCollector([requester, enemy], requester)
        assert len(result["enemy"]) == 1

    def test_merged_agent_filtered_out(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        merged = make_agent(identity="country_F", position=[5, 0])
        merged.mergedOrPruned = True
        result = AgentInfoCollector([requester, merged], requester)
        assert result["enemy"] == []

    def test_crushing_defeat_agent_filtered_out(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        defeated = make_agent(identity="country_F", position=[5, 0], stage="Crushing Defeat")
        result = AgentInfoCollector([requester, defeated], requester)
        assert result["enemy"] == []

    def test_dead_agent_filtered_out(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        dead = make_agent(identity="country_F", position=[5, 0], original=100, lost=100)
        result = AgentInfoCollector([requester, dead], requester)
        assert result["enemy"] == []

    def test_distance_calculation(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        other = make_agent(identity="country_F", position=[3, 4])
        result = AgentInfoCollector([requester, other], requester)
        # Euclidean: sqrt(9 + 16) = 5.0
        assert result["enemy"][0]["distance"] == pytest.approx(5.0, abs=0.1)

    def test_bearing_calculation(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        # Agent directly to the right (east) — bearing should be 0 degrees
        other = make_agent(identity="country_F", position=[10, 0])
        result = AgentInfoCollector([requester, other], requester)
        assert result["enemy"][0]["bearing"] == pytest.approx(0.0, abs=0.1)

    def test_result_self_field(self, make_agent):
        requester = make_agent(position=[7, 8])
        result = AgentInfoCollector([requester], requester)
        assert result["self"]["position"] == [7, 8]
        assert result["self"]["id"] == requester.hierarchy.id

    def test_default_threshold_covers_full_map(self, make_agent):
        """Characterization of Bug #6: default threshold 100000 means fog of war is off."""
        requester = make_agent(identity="country_E", position=[0, 0])
        # Place enemy at the far corner of a ±1000 map
        far_enemy = make_agent(identity="country_F", position=[999, 999])
        result = AgentInfoCollector([requester, far_enemy], requester)
        # With 100000 threshold, far enemy is visible — this is the bug
        assert len(result["enemy"]) == 1

    def test_explicit_threshold_hides_distant_enemy(self, make_agent):
        requester = make_agent(identity="country_E", position=[0, 0])
        far_enemy = make_agent(identity="country_F", position=[999, 999])
        # With a tight threshold, the far enemy should be hidden
        result = AgentInfoCollector([requester, far_enemy], requester, threshold_distance=100)
        assert result["enemy"] == []
