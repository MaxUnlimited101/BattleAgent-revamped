"""Per-run reproducibility artifacts: ``run_config.json`` and git SHA capture."""
import json
import os
import subprocess
from datetime import datetime


def get_git_sha():
    """Return the current git commit SHA, or ``None`` if unavailable."""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_run_config(log_dir, config):
    """Write ``run_config.json`` capturing the config, resolved models, seed, timestamp and git SHA.

    ``config`` is a ``SimulationConfig`` (already model-resolved). Returns the written dict.
    """
    payload = {
        "config": config.to_dict(),
        "resolved_models": {
            "commander": config.commander_model,
            "referee": config.referee_model,
            "diary": config.diary_model,
        },
        "seed": config.seed,
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
    }
    with open(os.path.join(log_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload
