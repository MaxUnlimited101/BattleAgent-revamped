from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_COST_PER_MILLION: dict = {
    "claude":     {"input": 15.0, "output": 75.0},
    "gpt":        {"input": 10.0, "output": 30.0},
    "openrouter": None,
    "ollama":     None,
    "fake":       {"input": 0.0,  "output": 0.0},
}


@dataclass
class RoleTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_t: int, output_t: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.calls += 1


@dataclass
class TokenAccumulator:
    commander: RoleTokens = field(default_factory=RoleTokens)
    referee: RoleTokens = field(default_factory=RoleTokens)
    diary: RoleTokens = field(default_factory=RoleTokens)

    def add(self, role: str, input_t: int, output_t: int) -> None:
        bucket = getattr(self, role, None)
        if bucket is not None:
            bucket.add(input_t, output_t)

    @property
    def total_input(self) -> int:
        return self.commander.input_tokens + self.referee.input_tokens + self.diary.input_tokens

    @property
    def total_output(self) -> int:
        return self.commander.output_tokens + self.referee.output_tokens + self.diary.output_tokens

    def summary_dict(self) -> dict:
        return {
            "commander": {"in": self.commander.input_tokens, "out": self.commander.output_tokens, "calls": self.commander.calls},
            "referee":   {"in": self.referee.input_tokens,   "out": self.referee.output_tokens,   "calls": self.referee.calls},
            "diary":     {"in": self.diary.input_tokens,     "out": self.diary.output_tokens,     "calls": self.diary.calls},
            "total":     {"in": self.total_input,            "out": self.total_output},
        }

    def cost_estimate(self, model_type: str) -> Optional[float]:
        prices = _COST_PER_MILLION.get(model_type)
        if prices is None:
            return None
        return (self.total_input * prices["input"] + self.total_output * prices["output"]) / 1_000_000
