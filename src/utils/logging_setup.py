"""Central logging configuration for a simulation run.

App code obtains a module logger via ``logging.getLogger(__name__)`` and calls it; ``configure_logging``
wires the root logger to the console plus a per-run ``simulation.log`` file. This replaces the ad-hoc
``print()`` calls that used to be the only output. ``BattleLogger`` (sandbox.py) still writes its own
timestamped log files; this handler is additive.
"""
import logging
import os

_CONFIGURED = False


def configure_logging(log_dir=None, level=logging.INFO):
    """Configure the root logger once. Idempotent: repeated calls only (re)attach the file handler.

    ``log_dir`` — if given, a ``simulation.log`` file handler is attached inside it.
    """
    global _CONFIGURED
    root = logging.getLogger()

    if not _CONFIGURED:
        root.setLevel(level)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(console)
        _CONFIGURED = True

    if log_dir:
        # Avoid attaching a duplicate file handler for the same run directory.
        target = os.path.join(log_dir, "simulation.log")
        for h in root.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(target):
                break
        else:
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)

    return root
