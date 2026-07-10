"""Structured JSONL event stream for a simulation run.

Each simulation writes one ``events.jsonl`` file in its run directory, one JSON object per line.
Events are emitted at the existing simulation seams (agent decisions, referee casualty assessments,
per-step token usage). The typed stream is the machine-readable counterpart to the human log and the
foundation for downstream analysis / Unity replay (research direction #3).
"""
import json
import os


class EventLogger:
    """Append-only JSONL writer. A ``None`` path makes every method a no-op (used in tests)."""

    def __init__(self, log_dir=None, filename="events.jsonl"):
        self.path = os.path.join(log_dir, filename) if log_dir else None
        if self.path:
            # Truncate any stale file from a previous run in the same directory.
            open(self.path, "w", encoding="utf-8").close()

    def emit(self, event_type, step=None, **fields):
        """Write one event. ``step`` is the simulation timestep; ``fields`` are event-specific."""
        if not self.path:
            return
        record = {"type": event_type, "t": step}
        record.update(fields)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Convenience wrappers for the three event families in the plan.
    def decision(self, step, agent_id, identity, action, position, deployed):
        self.emit("decision", step=step, agent_id=agent_id, identity=identity,
                  action=action, position=position, deployed=deployed)

    def casualties_assessed(self, step, agent_id, casualties, estimated_loss_percentage=None):
        self.emit("casualties_assessed", step=step, agent_id=agent_id, casualties=casualties,
                  estimated_loss_percentage=estimated_loss_percentage)

    def tokens(self, step, **usage):
        self.emit("tokens", step=step, **usage)
