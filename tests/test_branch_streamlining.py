"""Tests for BranchStreamlining (merge/prune) logic.

Troop conservation invariant: sum of remaining_num_of_troops across all
non-merged active agents should not change after a streamlining operation.
Bug #5 in the plan violates this — those tests are marked xfail.
"""

import pytest
from agent import BranchStreamlining


def _total_remaining(agent):
    """Recursively sum remaining troops across all active (non-merged) agents."""
    total = agent.profile.remaining_num_of_troops
    for sub in agent.hierarchy.sub_agents:
        if not sub.mergedOrPruned:
            total += _total_remaining(sub)
    return total


class TestBranchStreamliningBasics:
    def test_nonexistent_target_returns_early(self, make_agent):
        parent = make_agent(original=1000)
        result = BranchStreamlining(parent, "ARMY-nonexistent")
        assert result == "BranchStreamlining stop."
        assert parent.hierarchy.sub_agents == []

    def test_target_must_be_direct_child(self, make_agent):
        parent = make_agent(original=1000)
        grandchild = make_agent(original=100)
        # grandchild is NOT in parent.sub_agents
        result = BranchStreamlining(parent, grandchild.hierarchy.id)
        assert result == "BranchStreamlining stop."

    def test_merge_chosen_for_high_moral(self, make_agent):
        parent = make_agent(original=1000, deployed=400)
        sub = make_agent(original=400, lost=50, moral="High")
        parent.hierarchy.sub_agents.append(sub)
        sub.hierarchy.parent_agent = parent

        BranchStreamlining(parent, sub.hierarchy.id)

        # sub removed from parent's sub_agents
        assert sub not in parent.hierarchy.sub_agents
        assert sub.mergedOrPruned is True

    def test_prune_chosen_for_low_moral(self, make_agent):
        parent = make_agent(original=1000, deployed=400)
        sub = make_agent(original=400, lost=50, moral="Low")
        parent.hierarchy.sub_agents.append(sub)
        sub.hierarchy.parent_agent = parent

        BranchStreamlining(parent, sub.hierarchy.id)

        assert sub not in parent.hierarchy.sub_agents
        assert sub.mergedOrPruned is True

    def test_merge_chosen_for_medium_moral(self, make_agent):
        parent = make_agent(original=1000, deployed=400)
        sub = make_agent(original=400, moral="Medium")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)
        assert sub.mergedOrPruned is True

    def test_sub_agent_removed_from_parent_list(self, make_agent):
        parent = make_agent(original=1000, deployed=500)
        sub_a = make_agent(original=300, moral="High")
        sub_b = make_agent(original=200, moral="High")
        parent.hierarchy.sub_agents = [sub_a, sub_b]

        BranchStreamlining(parent, sub_a.hierarchy.id)

        assert sub_a not in parent.hierarchy.sub_agents
        assert sub_b in parent.hierarchy.sub_agents


class TestBranchStreamliningTroopAccounting:
    def test_merge_absorbs_sub_losses(self, make_agent):
        parent = make_agent(original=1000, deployed=400, lost=0)
        sub = make_agent(original=400, lost=100, moral="High")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)

        assert parent.profile.lost_num_of_troops == 100

    def test_prune_marks_all_sub_original_as_lost(self, make_agent):
        parent = make_agent(original=1000, deployed=300, lost=0)
        sub = make_agent(original=300, lost=50, moral="Low")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)

        # Prune: parent gains sub.original_num_of_troops (300) as losses
        assert parent.profile.lost_num_of_troops == 300

    def test_merge_decrements_deployed_by_sub_original(self, make_agent):
        parent = make_agent(original=1000, deployed=500, lost=0)
        sub = make_agent(original=500, lost=0, moral="High")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)

        assert parent.profile.deployed_num_of_troops == 0

    def test_prune_decrements_deployed_by_sub_original(self, make_agent):
        parent = make_agent(original=1000, deployed=300, lost=0)
        sub = make_agent(original=300, lost=0, moral="Low")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)

        assert parent.profile.deployed_num_of_troops == 0


class TestBranchStreamliningGrandchildren:
    def test_grandchildren_reparented_to_parent(self, make_agent):
        parent = make_agent(original=1000, deployed=500)
        sub = make_agent(original=500, deployed=200, moral="High")
        grandchild = make_agent(original=200)
        parent.hierarchy.sub_agents.append(sub)
        sub.hierarchy.parent_agent = parent
        sub.hierarchy.sub_agents.append(grandchild)
        grandchild.hierarchy.parent_agent = sub

        BranchStreamlining(parent, sub.hierarchy.id)

        assert grandchild in parent.hierarchy.sub_agents
        assert grandchild.hierarchy.parent_agent is parent

    def test_no_grandchildren_leaves_parent_sub_agents_empty(self, make_agent):
        parent = make_agent(original=1000, deployed=400)
        sub = make_agent(original=400, moral="High")
        parent.hierarchy.sub_agents.append(sub)

        BranchStreamlining(parent, sub.hierarchy.id)

        assert parent.hierarchy.sub_agents == []

    def test_merge_conserves_total_troops(self, make_agent):
        # Bug #5 from the plan is already fixed: parent now correctly adjusts
        # deployed by sub.original - sub.deployed so grandchildren are not double-counted.
        parent = make_agent(original=1000, deployed=500)
        sub = make_agent(original=500, deployed=200, moral="High")
        grandchild = make_agent(original=200)
        parent.hierarchy.sub_agents.append(sub)
        sub.hierarchy.parent_agent = parent
        sub.hierarchy.sub_agents.append(grandchild)
        grandchild.hierarchy.parent_agent = sub

        before = _total_remaining(parent)
        BranchStreamlining(parent, sub.hierarchy.id)
        after = _total_remaining(parent)

        assert before == after

    def test_prune_accounts_sub_remaining_as_casualties(self, make_agent):
        # Prune is semantically "abandon the sub-agent" so its remaining troops become
        # casualties on the parent. Total remaining DECREASES by sub.remaining, which is correct.
        parent = make_agent(original=1000, deployed=300)
        sub = make_agent(original=300, deployed=100, moral="Low")
        grandchild = make_agent(original=100)
        parent.hierarchy.sub_agents.append(sub)
        sub.hierarchy.parent_agent = parent
        sub.hierarchy.sub_agents.append(grandchild)
        grandchild.hierarchy.parent_agent = sub

        sub_remaining = sub.profile.remaining_num_of_troops  # 300-100=200
        parent_lost_before = parent.profile.lost_num_of_troops

        BranchStreamlining(parent, sub.hierarchy.id)

        assert parent.profile.lost_num_of_troops == parent_lost_before + sub_remaining
