# BattleAgent Improvement Plan

## Context

BattleAgent (arxiv 2404.15532) is a multi-agent LLM simulation of historical battles (Poitiers, Falkirk, Agincourt): hierarchical commander agents decide tactics each timestep, an LLM referee judges casualties, soldier "diaries" add human-perspective narratives, and a Unity demo visualizes results. ~4,500 LOC Python in `src/`, no tests, no CI.

The owner has migrated the LLM layer (`src/utils/LLM_api.py`, `src/utils/VLM_api.py`) to LangChain **on another machine; that work has not landed in this repo yet** (repo still uses native `openai`/`anthropic` SDKs on the only branch, `main`). This plan assumes the LangChain layer lands and builds around it. The owner wants improvements across: engineering hygiene, LLM-call reliability, performance/cost, architecture, and new feature/research directions.

**Guiding priority** (research codebase): correctness of simulation results > reproducibility > API cost > polish. Simulation stays runnable after every phase. `demo/` (Unity) is out of scope except where noted.

## Verified correctness bugs (highest research value — these skew published results)

1. **Referee uses the wrong map**: `external_construct_judgment_prompt` formats the module-level global `map_info_json` (default/Crecy map) at [referee.py:50-53](src/support_agents/referee.py#L50-L53) instead of the `map_info` parameter passed via `self.map_info_json`. Every Falkirk/Agincourt run judges casualties against the wrong geography. Weapon lore (longbow/mud) is also hardcoded for all battles ([referee.py:90](src/support_agents/referee.py#L90)).
2. **Referee retries are no-ops**: retry loop at [referee.py:163-174](src/support_agents/referee.py#L163-L174) never reassigns `parsed_json`, so successful retries are discarded → casualty rounds silently skipped.
3. **`targetedAgentId` chain broken**: `parsed_data_sync` sets `self.target_agent_id` (never read) instead of `self.hierarchy.target_agent_id` (what the referee reads) — root agents' own attacks are never evaluated by the referee. Also unvalidated → KeyError risk (agent.py:408).
4. **Agent retry loop broken**: in `execute()` (agent.py:540-572) the max-attempts raise is unreachable, failures fall through to silent `return None`; validation errors are appended *after* the retry prompt is built so retries never see the newest errors.
5. **Merge/prune double-counts troops**: `BranchStreamlining` (agent.py:59-97) returns the sub-agent's original count while reparented grandchildren keep their troops → conservation violated.
6. **Fog of war is dead code**: `AgentInfoCollector(threshold_distance=100000)` ([shared_func.py:11](src/utils/shared_func.py#L11)) covers the whole ±1000 map; `Sandbox.is_within_vision` (range 100) is never called. Agents are omniscient.
7. **Soldier-transfer logic dead**: `Soldier_Hierarchy` aliases the live `sub_agents` list (individual_profile.py:469-477), so `monitor_structure_change` always returns `[]` — diaries never follow detached units.

## Integration contract for the incoming LangChain layer

The migration must satisfy these seams (or Phase 3 adapts call sites):
- Keep a string-in/string-out shim `run_LLM(model_type, prompt) -> str` — called from exactly 3 places: agent.py:38 (`BasicAgent.run_model`), referee.py:180, individual_profile.py:557.
- Expose the underlying `BaseChatModel` via a per-role factory (`commander`/`referee`/`diary`) so we can use `.with_structured_output()`, `.batch()`, callbacks, and substitute `FakeListChatModel` in tests.
- **No module-level client construction** (current VLM_api.py:8 makes `import agent` crash without `OPENAI_API_KEY`). Lazy-init only.
- Model ids/temperature/seed configurable and logged per run (currently hardcoded deprecated models: `claude-3-opus-20240229`, `gpt-4-1106-preview`, `gpt-4-vision-preview`).
- Transport retries/backoff/rate limiting (`max_retries`, `InMemoryRateLimiter`) live in this layer; app keeps only semantic validation retries.
- GPT4V path (`execute_WithGpt4V`, agent.py:587) is broken today (malformed image payload + hardcoded Windows path); either reimplement with LangChain multimodal messages or gate behind `NotImplementedError`.

## Phased roadmap

### Phase 0 — Hygiene quick wins (~0.5 day, no behavior change, no LangChain dependency)
- Add root `.gitignore` (`__pycache__/`, `*.pyc`, `src/logs/`, `*.pkl`, `.env`, `.venv/`); `git rm -r --cached` the 92 tracked `.pyc` files.
- Fix `requirements.txt`: pin versions, add missing actually-imported deps (`json5`, `treelib`, `tqdm`, `numpy`, `matplotlib`). Add `requirements-dev.txt` (`pytest`, `ruff`). Leave langchain pins to the migration PR (comment placeholder).
- Delete verified dead code: duplicate `import pickle` (sandbox.py:5,9), dead `run_gpt4v` imports (sandbox.py:23, referee.py:12, individual_profile.py:11), dead `Sandbox.is_within_vision`, sandbox `__main__` block loading a nonexistent pickle, unused `import uuid` in sim_stopper.py.
- Fix `src/run_*.sh` (hardcode another user's conda paths).

### Phase 1 — Test harness + characterization tests (1–2 days, zero API tokens)
- `tests/` + pytest config (`pythonpath=["src"]`, dummy API keys in conftest before imports).
- Fake the LLM by monkeypatching the 3 `run_model` seams with a scripted-response fake (mirrors `FakeListChatModel` so the Phase 3 swap is trivial). `Detachment_Agent.execute(LLM_response=...)` already accepts injected responses.
- ~60–80 tests over pure logic: JSON parsing (agent.py:356-385, referee parse), `validate_parsed_output`, `parsed_data_sync` + stage thresholds, `BranchStreamlining` merge/prune with a **troop-conservation invariant helper**, `sim_stopper` 0.8 thresholds, vision/distance math, referee casualty application, soldier transfer/injury logic. Mark known bugs as `xfail` (characterization).
- CI: `.github/workflows/ci.yml` — ruff (lenient, `F` only) + pytest on pinned Python. No secrets.

### Phase 2 — Correctness fixes (1–2 days; each flips a Phase 1 xfail; changes experiment behavior intentionally)
- Fix the 7 verified bugs above (referee map global → parameter; referee retry reassignment + WARNING & `skipped_casualty_rounds` counter; `targetedAgentId` validation + correct attribute; execute() retry feeds errors back under a labeled prompt section + raises explicit `AgentExecutionError` after max attempts; merge/prune conservation; soldier-list copy + mutable-default fix).
- Sandbox exception policy (sandbox.py:303-305): stop swallowing everything — catch `AgentExecutionError` → log + count, let unexpected exceptions propagate; `--on-agent-error {continue,abort}`; end-of-run summary (agents failed, casualty rounds skipped).
- Parameterize vision range (`--vision_range`, default keeps current 100000 behavior — don't silently change results).
- `tests/test_e2e_offline.py`: 2-agent Poitiers sandbox, scripted responses, 2 steps, assert no exceptions + troop conservation + logs written. This is the "still runnable" gate for all later phases.

### Phase 3 — Build on the LangChain layer (1–2 days; gated on migration landing; rebase P0–P2 onto it)
- Point the 3 `run_model` call sites at the new layer; delete leftover native shims.
- **Structured output**: `src/schemas.py` with Pydantic models (`CommanderDecision`, `SubAgentAction`, `CasualtiesResult`) mirroring `json_constraint_variable` (Detachment_Agent_prompt.py:57-100); use `.with_structured_output()` (fallback `PydanticOutputParser`+`OutputFixingParser`). Semantic checks (position bounds, deploy limits, morale enum) become validators. Keep legacy parser behind `--parser legacy|structured` for one release to A/B.
- Transport retries/rate limits move to model config; app loop retries only on semantic failure.
- **Token/cost accounting**: `UsageMetadataCallbackHandler` through `Sandbox.simulate`; per-step + cumulative tokens/cost in logs and run summary. Per-role models (cheap model for referee/diaries — diaries are the biggest cost lever: up to 30 calls per agent-step).
- `--LLM_MODEL fake` provider (`FakeListChatModel`) → full CLI runnable with zero tokens; convert test fakes to this seam.

### Phase 4 — Performance & cost (2–3 days, after P3)
- **Parallel agent decisions**: gather prompts per step, `chat_model.batch(..., max_concurrency=N)`, apply syncs sequentially. Ship as `--execution-mode {sequential,parallel}` (default sequential — batching changes semantics: simultaneous decisions vs seeing earlier same-step moves). Referee + diary batching has no semantic caveat — do unconditionally.
- **Prompt cost**: `construct_prompt` (agent.py:308-352) embeds the entire raw text of every previous response → unbounded growth. Replace with last-K extracted JSON decisions (full history stays on disk). Split static prefix (system/army/action space) from dynamic suffix; enable Anthropic prompt caching (`cache_control`) on the prefix; precompute prefix once per agent.
- Wall-time per step in run summary; compare tokens/step vs Phase 3 baseline.

### Phase 5 — Config-driven architecture & reproducibility (2–3 days, after P2; parallel to P3/P4)
- **Battle registry** (`src/battles.py`): name → dataclass (positions, troops, map, profiles), replacing the if/elif at simulation_controller.py:19-88. Battle #4 becomes one entry.
- Externalize the 30 soldier-profile dicts (individual_profile.py:14-465) to `src/data/soldier_profiles/*.json` (~450 lines deleted).
- `SimulationConfig`/constants: vision range, deploy cap 0.6 (agent.py:151), defeat thresholds 0.1/0.5 (agent.py:423), sub-agent threshold 5, injury probs, seeds — plumbed from argparse.
- Replace 42 `print()` with `logging`; add structured JSONL event stream per run (decisions, casualties, tokens).
- Reproducibility: `run_config.json` per run dir (args, model ids, temperature, seeds, git SHA); deterministic agent ids (seeded, not `uuid4`).
- Golden-file prompt snapshot tests guard the refactor.

### Phase 6 — Optional structural decomposition (3–5 days, only with continued investment)
Split `Detachment_Agent` (prompting/parsing/state-sync) and `Sandbox` (engine/logging/persistence); replace whole-object pickling with versioned state serialization; incremental type hints; kill wildcard imports. **Keep procoder** — self-contained and working; porting prompts to LangChain templates is high-churn, low-value.

## New feature / research directions (ranked by value-to-effort)

| # | Direction | Type | Effort | Needs |
|---|---|---|---|---|
| 1 | **Per-role model routing** — cheap/local (Ollama) diarists, strong commanders, neutral referee, via role-keyed registry | Cost/product | S | LangChain layer |
| 2 | **LangSmith tracing + cost rollups** tagged `{battle, run_id, step, agent_id, role}` | Product | S | LangChain layer |
| 3 | **Structured event log → Unity replay** — typed JSONL events (`unit_moved`, `casualties_assessed`…) emitted from `parsed_data_sync`/`BranchStreamlining`/referee; exporter to the action vocabulary `SimulationManager.cs` already supports (currently hand-coded Crecy script); later WebSocket live streaming | Product, keystone | M (+M live Unity) | — |
| 4 | **Multi-run experiment runner + statistics** — N seeds × battle × config via subprocess isolation; pandas analysis of outcomes/casualty curves; prompt-sensitivity study (publishable on its own; procoder's `replaced_submodule` was built for this) | Research infra | M | 3 |
| 5 | **Model-vs-model tournaments** — country E vs F commanded by different models, fixed neutral referee; win rates, casualty efficiency, JSON-validity rates. Architecture is already two-sided; only the CLI conflates sides | Research/benchmark | S–M | 1,3,4 |
| 6 | **Real fog of war + messenger comms** — per-type vision radii, terrain occlusion, stale last-known positions, order/report delivery delays. One-line lever exists (`threshold_distance`); prompts must stop assuming omniscience | Research | M | better with 4 |
| 7 | **Historical fidelity benchmark** — per-battle ground truth (winner, casualty ranges, signature dynamics) + metrics; forces the referee map/weapon-lore fixes; turns the sim into an instrument | Research | M–L | 3,4 |
| 8 | **Counterfactual history sweeps** — base battle + config overrides ("French longbows at Agincourt"); the project's founding pitch, cheap once battles are data | Research | S–M | 4 (9 helps) |
| 9 | **Battles as pure data + LLM-assisted battle authoring** — YAML battle packs (geography, armies, positions, referee lore, persona pool) + generic loader; stretch: structured-output generator drafting packs from Wikipedia for review. Also: generated soldier personas at scale (current 15/side are reused) | Product | M | — |
| 10 | **Agent memory & cross-battle learning** — replace raw history dump with summarized memory; persist per-commander "lessons" across runs; measure learning curves | Research | M–L | 1,4 |

Suggested sequencing: 1+2 land with the migration itself → 3 (keystone artifact) → 4 (turns repo into an experiment platform) → then 5/7/8 are mostly sweep specs + notebooks. 6 and 9 are independent parallel tracks.

## Execution order & dependencies

```
P0 hygiene → P1 tests+CI → P2 correctness ──→ P5 config+repro (independent of P3/P4)
                                          └─→ P3 LangChain adoption (gated on migration landing) → P4 perf+cost
Research directions: after P3 (1,2) and P5 (3,4); rest follow.
```

P0–P2 should start immediately — they don't wait for the LangChain branch and touch the `run_model` seams only via monkeypatching, so rebase conflicts will be minimal.

## Critical files

- [src/agent.py](src/agent.py) — parse/validate/retry/sync/merge; most P1–P3 work
- [src/sandbox.py](src/sandbox.py) — main loop, exception policy, parallelism, token accounting
- [src/support_agents/referee.py](src/support_agents/referee.py) — map global bug, no-op retries, casualty correctness
- [src/group_experience/individual_profile.py](src/group_experience/individual_profile.py) — soldier-transfer aliasing, 30 profiles to externalize, diary cost
- [src/simulation_controller.py](src/simulation_controller.py) — battle registry, CLI/config, fake provider
- [src/utils/shared_func.py](src/utils/shared_func.py) — fog-of-war lever
- [requirements.txt](requirements.txt), new: `.gitignore`, `tests/`, `.github/workflows/ci.yml`, `src/schemas.py`, `src/battles.py`

## Verification

- After every phase: `pytest -q` green (xfails shrink each phase) and the offline e2e gate: `cd src && python simulation_controller.py --LLM_MODEL fake --conflict_name Poitiers --simulation_time 30 --update_interval 15 --have_diaries 0` (zero tokens, from P3; before P3, the monkeypatched e2e test covers this).
- P0: fresh venv installs from requirements.txt; all modules import with dummy keys; `git status` shows no tracked `.pyc`.
- P2: troop-conservation invariant passes on e2e run; referee prompt snapshot contains the correct battle's geography.
- P4: scripted-decision runs produce state-identical results in sequential vs parallel mode; token report shows reduced tokens/step vs P3 baseline.
- Final: one real cheap-model smoke run per battle; inspect cost lines in `src/logs/<run>/`.
