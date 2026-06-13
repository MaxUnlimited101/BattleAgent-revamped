"""Tests for Pydantic schema models in src/schemas.py."""
import pytest
from pydantic import ValidationError

from schemas import CasualtiesResponse, CommanderDecision, SubAgentAction

VALID_COMMANDER = {
    "agentNextActionType": "Defend",
    "remarks": "Holding position",
    "SubAgentsRecall": [],
    "agentMoral": "Medium",
    "speed": 5,
    "agentNextPosition": [15, -10],
    "deploySubUnit": False,
    "targetedAgentId": "",
    "actions": [],
}

VALID_ACTION = {
    "subAgent_NextActionType": "Charge",
    "troopType": "cavalry",
    "speed": 8,
    "deployedNum": 200,
    "ownPotentialLostNum": 10,
    "enemyPotentialLostNum": 50,
    "position": [5, 10],
    "agent_id": "ARMY-target",
    "remarks": "Flanking manoeuvre",
}


class TestCommanderDecision:
    def test_valid_data_parses(self):
        cd = CommanderDecision.model_validate(VALID_COMMANDER)
        assert cd.agentMoral == "Medium"
        assert cd.speed == 5
        assert cd.actions == []

    def test_invalid_moral_raises(self):
        with pytest.raises(ValidationError):
            CommanderDecision.model_validate({**VALID_COMMANDER, "agentMoral": "Scared"})

    def test_all_moral_values_accepted(self):
        for moral in ("High", "Medium", "Low"):
            cd = CommanderDecision.model_validate({**VALID_COMMANDER, "agentMoral": moral})
            assert cd.agentMoral == moral

    def test_negative_speed_raises(self):
        with pytest.raises(ValidationError):
            CommanderDecision.model_validate({**VALID_COMMANDER, "speed": -1})

    def test_zero_speed_accepted(self):
        cd = CommanderDecision.model_validate({**VALID_COMMANDER, "speed": 0})
        assert cd.speed == 0

    def test_position_wrong_length_raises(self):
        for bad in ([], [1], [1, 2, 3]):
            with pytest.raises(ValidationError):
                CommanderDecision.model_validate({**VALID_COMMANDER, "agentNextPosition": bad})

    def test_targeted_agent_id_defaults_empty(self):
        data = {k: v for k, v in VALID_COMMANDER.items() if k != "targetedAgentId"}
        cd = CommanderDecision.model_validate(data)
        assert cd.targetedAgentId == ""

    def test_valid_action_accepted(self):
        cd = CommanderDecision.model_validate({**VALID_COMMANDER, "actions": [VALID_ACTION]})
        assert len(cd.actions) == 1
        assert cd.actions[0].troopType == "cavalry"


class TestSubAgentAction:
    def test_valid_action_parses(self):
        action = SubAgentAction.model_validate(VALID_ACTION)
        assert action.deployedNum == 200

    def test_action_negative_speed_raises(self):
        with pytest.raises(ValidationError):
            SubAgentAction.model_validate({**VALID_ACTION, "speed": -5})

    def test_action_position_wrong_length_raises(self):
        with pytest.raises(ValidationError):
            SubAgentAction.model_validate({**VALID_ACTION, "position": [1, 2, 3]})


class TestCasualtiesResponse:
    def test_valid_casualties_parses(self):
        data = {"casualties_result": [
            {"agent_id": "ARMY-X", "casualties": 50, "estimated_loss_percentage": 0.1}
        ]}
        cr = CasualtiesResponse.model_validate(data)
        assert cr.casualties_result[0].agent_id == "ARMY-X"
        assert cr.casualties_result[0].casualties == 50

    def test_empty_list_parses(self):
        cr = CasualtiesResponse.model_validate({"casualties_result": []})
        assert cr.casualties_result == []

    def test_missing_estimated_loss_defaults_zero(self):
        data = {"casualties_result": [{"agent_id": "X", "casualties": 10}]}
        cr = CasualtiesResponse.model_validate(data)
        assert cr.casualties_result[0].estimated_loss_percentage == 0.0

    def test_missing_casualties_result_key_raises(self):
        with pytest.raises(ValidationError):
            CasualtiesResponse.model_validate({})
