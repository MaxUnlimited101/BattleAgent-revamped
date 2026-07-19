"""Structured JSONL event stream for a simulation run.

Each simulation writes one ``events.jsonl`` file in its run directory, one JSON object per line.
Events are emitted at the existing simulation seams (agent decisions, referee casualty assessments,
per-step token usage). The typed stream is the machine-readable counterpart to the human log and the
foundation for downstream analysis / Unity replay (research direction #3).
"""
import json
import os


class EventLogger:
    """Append-only JSONL writer. A ``None`` path makes every method a no-op (used in tests).

    When the ``BATTLE_LIVE_VIZ`` environment variable is set (to any non-empty value), each event
    is also streamed to connected browsers over WebSocket via a :class:`~utils.live_viz.LiveBroadcaster`
    (port from ``BATTLE_LIVE_VIZ_PORT``, default 8765). Live streaming is best-effort and never
    affects the on-disk log or the simulation.
    """

    def __init__(self, log_dir=None, filename="events.jsonl"):
        self.path = os.path.join(log_dir, filename) if log_dir else None
        if self.path:
            # Truncate any stale file from a previous run in the same directory.
            open(self.path, "w", encoding="utf-8").close()

        self._broadcaster = None
        if os.getenv("BATTLE_LIVE_VIZ"):
            from utils.live_viz import LiveBroadcaster
            port = int(os.getenv("BATTLE_LIVE_VIZ_PORT", "8765"))
            self._broadcaster = LiveBroadcaster(port=port)

    def emit(self, event_type, step=None, **fields):
        """Write one event. ``step`` is the simulation timestep; ``fields`` are event-specific."""
        record = {"type": event_type, "t": step}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._broadcaster is not None:
            self._broadcaster.publish(line)

    # Convenience wrappers for the three event families in the plan.
    def decision(self, step, agent_id, identity, action, position, deployed,
                 remarks=None, moral=None, target=None):
        # ``remarks`` is the commander's own rationale for the action (the "why" the
        # visualizer surfaces); ``moral`` and ``target`` add tactical context. All three
        # are optional so older call sites / logs remain valid.
        self.emit("decision", step=step, agent_id=agent_id, identity=identity,
                  action=action, position=position, deployed=deployed,
                  remarks=remarks, moral=moral, target=target)

    def casualties_assessed(self, step, agent_id, casualties, estimated_loss_percentage=None):
        self.emit("casualties_assessed", step=step, agent_id=agent_id, casualties=casualties,
                  estimated_loss_percentage=estimated_loss_percentage)

    def tokens(self, step, **usage):
        self.emit("tokens", step=step, **usage)
