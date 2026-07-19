# BattleAgent Decision Viewer

A single-file, dependency-free browser tool for watching LLM commander decisions play out on
the battlefield — the spatial map on the left, the reasoning feed (`remarks`, morale, deployed
counts) and token/cost HUD on the right. Works in two modes:

- **File replay** — load any run's `events.jsonl` and scrub through it (play/pause/step/speed).
- **Live** — stream a simulation in real time over WebSocket.

It consumes the structured event stream written by
[`src/utils/event_log.py`](../src/utils/event_log.py) (`decision`, `casualties_assessed`,
`tokens`). No Unity, no build step.

## File replay (no setup)

1. Open `viz/battle_viewer.html` in any browser.
2. Click **📂 Load events.jsonl** and pick a run's file, e.g. `src/logs/<run>/events.jsonl`,
   or the bundled demo `viz/sample_events.jsonl` to see a full mini-battle immediately.
3. Use **▶︎ / ❚❚**, **Step ▸**, the speed slider, and **⟲** to drive playback.

## Live streaming

1. Install the optional dep (guarded — the sim runs fine without it):
   ```bash
   pip install websockets      # already in requirements.txt
   ```
2. Start a simulation with live streaming enabled:
   ```bash
   BATTLE_LIVE_VIZ=1 python src/main.py ...      # your usual run command
   # optional: BATTLE_LIVE_VIZ_PORT=8765 (default)
   ```
3. Open `viz/battle_viewer.html`, confirm the URL (`ws://localhost:8765`), click **🔴 Live**.

A browser that connects mid-run is replayed the full backlog first, so it never starts blank.
Live streaming is best-effort: if `websockets` is missing or the port is taken, the run logs a
warning and proceeds normally, still writing `events.jsonl`.

## Event schema

One JSON object per line:

| type | key fields |
|------|-----------|
| `decision` | `agent_id`, `identity` (`country_E`/`country_F`), `action`, `position` `[x,y]`, `deployed`, `remarks`, `moral`, `target` |
| `casualties_assessed` | `agent_id`, `casualties`, `estimated_loss_percentage` |
| `tokens` | `step_tokens`, `cumulative`, `wall_seconds` |

All events carry `t` (simulation step). Agents are colored by `identity`; positions auto-fit to
the canvas (Y inverted so north is up).
