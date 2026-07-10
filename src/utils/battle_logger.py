"""Human-readable battle log writer.

Extracted verbatim from ``sandbox.py`` (Phase 6 decomposition) to separate the logging concern
from the simulation engine. ``BattleLogger`` owns the per-run log directory and appends
timestamped, human-readable entries (actions, hierarchy trees, war-situation summaries). The
structured JSONL stream lives separately in ``utils.event_log.EventLogger``.
"""

import logging
import os
from datetime import datetime

from treelib import Tree

logger = logging.getLogger(__name__)


class BattleLogger:
    def __init__(self, campaign_name: str):
        self.campaign_name = campaign_name
        self.logs: list[str] = []
        self.log_subdirectory = self._setup_logging()

    def _setup_logging(self) -> str:
        """Set up logging directories and files."""
        real_world_time_at_creation = datetime.now().strftime('%m%d-%H%M_%S')
        log_subdirectory = f"{real_world_time_at_creation}_{self.campaign_name}"

        script_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        log_directory = os.path.join(script_directory, "logs")
        self.log_directory_path = os.path.join(log_directory, log_subdirectory)
        os.makedirs(self.log_directory_path, exist_ok=True)
        return log_subdirectory

    def log_action(self, action, info=None, system_time=None) -> None:
        time_str = system_time.strftime('%Y-%m-%d %H:%M') if system_time else datetime.now().strftime('%Y-%m-%d %H:%M')
        log_entry = f"{time_str}: {action}"
        if info is not None:
            log_entry += f" - {info}"

        self.logs.append(log_entry)
        self.save_log_to_file(log_entry, system_time)

    def log_tree(self, hierarchy_root, label, system_time) -> None:
        """
        Builds a tree from the hierarchy root and logs it as a string.
        """
        def build_tree(hierarchy_node, parent_id=None, tree=None):
            if tree is None:
                tree = Tree()
            tree.create_node(tag=hierarchy_node.hierarchy.id, identifier=hierarchy_node.hierarchy.id, parent=parent_id)
            for sub_agent in hierarchy_node.hierarchy.sub_agents:
                build_tree(sub_agent, hierarchy_node.hierarchy.id, tree)
            return tree

        # Build the tree
        tree = build_tree(hierarchy_root)

        # Convert the tree to a string
        tree_str = tree.show(stdout=False)

        # Create the log entry
        log_entry = f"Tree Structure for {label}:\n{tree_str}"

        # Log the action
        self.log_action("Tree Structure Logged", log_entry, system_time)

    def log_war_situation(self, label, war_situation, decision, system_time) -> None:
        situation_summary = (
            f"{label} War Situation - Total Agents: {war_situation['total_agents']}, "
            f"Command Structure Impact: {war_situation['command_structure_impact']}, "
            f"Morale Collapse Impact: {war_situation['morale_collapse_impact']}, "
            f"Heavy Casualties: {war_situation['heavy_casualties_count']}, "
            f"Total Troops: {war_situation['total_troops']}"
        )
        self.log_action("War Situation and Decision", situation_summary, system_time)

    def save_log_to_file(self, log_entry, system_time) -> None:
        filename = os.path.join(self.log_directory_path, f"{system_time.strftime('%Y%m%d-%H%M')}_simulation.log") if system_time else "general.log"
        with open(filename, 'a') as file:
            file.write(log_entry + "\n")
