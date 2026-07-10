"""Battle registry: name -> BattleConfig.

Replaces the two parallel ``if/elif`` blocks that used to live in ``ConflictConfig``
(simulation_controller.py). Adding a battle is now a single data entry here. Each entry references
the pre-existing prompt/map modules; no battle prose is duplicated.
"""
from dataclasses import dataclass
from typing import Any

from prompt.map_setting_of_other_battles import (
    map_info_json_Agincourt,
    map_info_json_Falkirk,
    map_info_json_Poitiers,
)
from prompt.agent_profile_Poitiers import (
    country_E_Army_Poitiers,
    country_F_Army_Poitiers,
    System_Setting_Poitiers,
    History_Setting_Poitiers,
)
from prompt.agent_profile_Falkirk import (
    country_E_Army_Falkirk,
    country_F_Army_Falkirk,
    System_Setting_Falkirk,
    History_Setting_Falkirk,
)
from prompt.agent_profile_Agincourt import (
    country_E_Army_Agincourt,
    country_F_Army_Agincourt,
    System_Setting_Agincourt,
    History_Setting_Agincourt,
)
# Crécy is the original/default battle: its map and armies are the module-level defaults that
# predate the other three, reused here so the 4th entry needs no invented data.
from prompt.map_setting import map_info_json as map_info_json_Crecy
from prompt.agent_profile import (
    country_E_Army as country_E_Army_Crecy,
    country_F_Army as country_F_Army_Crecy,
    System_Setting as System_Setting_Crecy,
    History_Setting as History_Setting_Crecy,
)


@dataclass(frozen=True)
class BattleConfig:
    """Immutable definition of a single battle scenario."""
    name: str
    country_E_position: list
    country_E_troops: int
    country_F_position: list
    country_F_troops: int
    map_info_json: dict
    system_setting: Any
    history_setting: Any
    country_E_army: Any
    country_F_army: Any


BATTLES = {
    "Poitiers": BattleConfig(
        name="Poitiers",
        country_E_position=[15, -10],
        country_E_troops=6000,
        country_F_position=[-10, 5],
        country_F_troops=15000,
        map_info_json=map_info_json_Poitiers,
        system_setting=System_Setting_Poitiers,
        history_setting=History_Setting_Poitiers,
        country_E_army=country_E_Army_Poitiers,
        country_F_army=country_F_Army_Poitiers,
    ),
    "Falkirk": BattleConfig(
        name="Falkirk",
        country_E_position=[0, 0],
        country_E_troops=15000,
        country_F_position=[50, 0],
        country_F_troops=6000,
        map_info_json=map_info_json_Falkirk,
        system_setting=System_Setting_Falkirk,
        history_setting=History_Setting_Falkirk,
        country_E_army=country_E_Army_Falkirk,
        country_F_army=country_F_Army_Falkirk,
    ),
    "Agincourt": BattleConfig(
        name="Agincourt",
        country_E_position=[0, -100],
        country_E_troops=6500,
        country_F_position=[15, -50],
        country_F_troops=35000,
        map_info_json=map_info_json_Agincourt,
        system_setting=System_Setting_Agincourt,
        history_setting=History_Setting_Agincourt,
        country_E_army=country_E_Army_Agincourt,
        country_F_army=country_F_Army_Agincourt,
    ),
    # Placeholder 4th battle demonstrating registry extensibility. Positions come from the default
    # map (country_E holds Village C at [0,0]; country_F targets Village E at [180,0]). Troop counts
    # are placeholder Crécy-scale values, not verified ground truth.
    "Crecy": BattleConfig(
        name="Crecy",
        country_E_position=[0, 0],
        country_E_troops=12000,
        country_F_position=[180, 0],
        country_F_troops=30000,
        map_info_json=map_info_json_Crecy,
        system_setting=System_Setting_Crecy,
        history_setting=History_Setting_Crecy,
        country_E_army=country_E_Army_Crecy,
        country_F_army=country_F_Army_Crecy,
    ),
}


def get_battle(name):
    """Return the BattleConfig for ``name`` or raise ValueError listing valid names."""
    try:
        return BATTLES[name]
    except KeyError:
        valid = ", ".join(sorted(BATTLES))
        raise ValueError(f"Unknown battle '{name}'. Valid battles: {valid}.")
