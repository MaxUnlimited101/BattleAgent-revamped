from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, field_validator


class SubAgentAction(BaseModel):
    subAgent_NextActionType: str
    troopType: str
    speed: int
    deployedNum: int
    ownPotentialLostNum: int
    enemyPotentialLostNum: int
    position: List[int]
    agent_id: str
    remarks: str

    @field_validator("position")
    @classmethod
    def position_two_ints(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("position must be exactly 2 integers")
        return v

    @field_validator("speed")
    @classmethod
    def speed_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("speed must be >= 0")
        return v


class CommanderDecision(BaseModel):
    agentNextActionType: str
    remarks: str
    SubAgentsRecall: List[str]
    agentMoral: Literal["High", "Medium", "Low"]
    speed: int
    agentNextPosition: List[int]
    deploySubUnit: bool
    targetedAgentId: str = ""
    actions: List[SubAgentAction] = []

    @field_validator("agentNextPosition")
    @classmethod
    def position_two_ints(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("agentNextPosition must be exactly 2 integers")
        return v

    @field_validator("speed")
    @classmethod
    def speed_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("speed must be >= 0")
        return v


class CasualtiesResult(BaseModel):
    agent_id: str
    casualties: int
    estimated_loss_percentage: float = 0.0


class CasualtiesResponse(BaseModel):
    casualties_result: List[CasualtiesResult]
