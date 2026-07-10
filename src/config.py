"""Central simulation configuration.

``SimulationConfig`` captures every CLI argument plus constants that used to be magic numbers
scattered across ``agent.py``/``sandbox.py``/``individual_profile.py``. Defaults exactly match the
values in the pre-Phase-5 code, so a run with default arguments reproduces prior behavior. The
config is serialized into ``run_config.json`` per run for reproducibility.
"""
from dataclasses import dataclass, asdict


@dataclass
class SimulationConfig:
    # --- run / battle ---
    conflict_name: str = "Poitiers"
    simulation_time: int = 90
    update_interval: int = 15

    # --- models ---
    llm_model: str = "gpt"
    commander_model: str = None
    referee_model: str = None
    diary_model: str = None

    # --- behavior toggles ---
    have_diaries: bool = False
    continue_run: bool = True
    gpt4v: bool = False
    parser_mode: str = "legacy"
    execution_mode: str = "sequential"
    max_concurrency: int = 8
    history_window: int = 3
    prompt_caching: bool = False
    on_agent_error: str = "continue"
    snapshot_format: str = "json"   # per-step state snapshot: "json" (versioned) or "pickle" (legacy)

    # --- tunable constants (previously hardcoded literals) ---
    vision_range: int = 100000            # was simulation_controller.py CLI default
    max_deploy_percent: float = 0.6       # was agent.py Detachment_AgentProfile
    crushing_defeat_remaining_frac: float = 0.1   # was agent.py parsed_data_sync
    crushing_defeat_lost_frac: float = 0.5        # was agent.py parsed_data_sync
    sub_agent_threshold: int = 5          # was sandbox.Sandbox default
    diary_injury_prob: float = 0.3        # was individual_profile.injury_generator
    seed: int = 42                        # was module-level random.seed(42) in sandbox.py
    round_interval: int = 15              # was agent.py Detachment_AgentProfile

    def resolve_models(self):
        """Fill per-role models from ``llm_model`` when not explicitly set."""
        self.commander_model = self.commander_model or self.llm_model
        self.referee_model = self.referee_model or self.llm_model
        self.diary_model = self.diary_model or self.llm_model
        return self

    @classmethod
    def from_args(cls, args):
        """Build a config from an argparse Namespace."""
        cfg = cls(
            conflict_name=args.conflict_name,
            simulation_time=args.simulation_time,
            update_interval=args.update_interval,
            llm_model=args.LLM_MODEL,
            commander_model=args.commander_model,
            referee_model=args.referee_model,
            diary_model=args.diary_model,
            have_diaries=bool(args.have_diaries),
            continue_run=bool(args.continue_run),
            gpt4v=bool(args.is_GPT4V_activate),
            parser_mode=args.parser,
            execution_mode=args.execution_mode,
            max_concurrency=args.max_concurrency,
            history_window=args.history_window,
            prompt_caching=args.prompt_caching,
            on_agent_error=args.on_agent_error,
            snapshot_format=args.snapshot_format,
            vision_range=args.vision_range,
            max_deploy_percent=args.max_deploy_percent,
            crushing_defeat_remaining_frac=args.crushing_defeat_remaining_frac,
            crushing_defeat_lost_frac=args.crushing_defeat_lost_frac,
            sub_agent_threshold=args.sub_agent_threshold,
            diary_injury_prob=args.diary_injury_prob,
            seed=args.seed,
        )
        return cfg.resolve_models()

    def to_dict(self):
        return asdict(self)
