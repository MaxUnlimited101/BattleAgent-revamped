# Standard library imports
import logging
import pickle
import itertools
from abc import ABC, abstractmethod


# Local application/library-specific imports

# Conflict info

import agent_prompting
import agent_parsing
# BranchStreamlining is re-exported so `from agent import BranchStreamlining` keeps working
# (redundant alias marks it as an intentional re-export for linters).
from agent_state_sync import BranchStreamlining as BranchStreamlining
from agent_state_sync import create_sub_agent, parsed_data_sync
from utils.LLM_api import run_LLM
from utils.VLM_api import run_gpt4v
from utils.surrounding_visualization import plot_tactical_positions

logger = logging.getLogger(__name__)

# Deterministic agent-id generation (replaces uuid4 for reproducibility). Ids are assigned in
# creation order; identical inputs produce identical ids. Reset between runs/tests via reset_agent_ids().
_agent_id_counter = itertools.count(1)


def reset_agent_ids():
    """Reset the monotonic agent-id counter (call at the start of a run or test for reproducibility)."""
    global _agent_id_counter
    _agent_id_counter = itertools.count(1)


def next_agent_id():
    return f"ARMY-{next(_agent_id_counter):04d}"


class AgentExecutionError(Exception):
    """Raised when an agent fails to produce valid output after max retry attempts."""
    pass


# base agent class
class BasicAgent(ABC):
    def __init__(self, identity, model_type):
        self.identity = identity
        self.model_type = model_type
        self.history = []
        self.parser_mode = "legacy"
        self.token_accumulator = None
        # Set by the sandbox each step so parsed_data_sync can emit structured events (optional).
        self.event_logger = None
        self.current_step = None
        
    @abstractmethod
    def construct_prompt(self):
        pass

    def run_model(self, prompt):
        return run_LLM(self.model_type, prompt, self.token_accumulator, "commander")
            
    @abstractmethod
    def parse_llm_output(self):
        # return json result
        pass

    @abstractmethod
    def validate_parsed_output(self):
        pass

    @abstractmethod
    def parsed_data_sync(self, parse_data):
        pass
    
    @abstractmethod
    def execute(self):
        pass


class ConstantPromptConfig:
    def __init__(self, System_Setting, History_Setting, army_setting, role_setting, troop_information, json_constraint_variable, json_example_text, action_list, action_property_definition, action_instruction_block, map_info_json, additional_settings):
        # Overall Setting
        self.System_Setting = System_Setting
        self.History_Setting = History_Setting

        # Army Setting
        self.army_setting = army_setting
        self.role_setting = role_setting
        self.troop_information = troop_information

        # Prompt setting 
        self.json_constraint_variable = json_constraint_variable
        self.action_list = action_list
        self.action_property_definition = action_property_definition
        self.action_instruction_block = action_instruction_block
        self.map_info_json = map_info_json

        # Additional Settings
        self.additional_settings = additional_settings
        


class Detachment_AgentProfile:
    def __init__(self, identity, position, original_num_of_troops, initial_mission = "win the battles",  constant_prompt_config = None,
                 max_deploy_percent=0.6, crushing_defeat_remaining_frac=0.1, crushing_defeat_lost_frac=0.5, round_interval=15):
        ### Army Information
        self.identity = identity
        self.position = position
        self.deployed_num_of_troops = 0
        self.lost_num_of_troops = 0
        self.original_num_of_troops = original_num_of_troops
        
        ### Prompt Part
        self.constant_prompt_config = constant_prompt_config
        
        self.System_Setting = constant_prompt_config.System_Setting
        self.history_setting = constant_prompt_config.History_Setting
        self.army_setting = constant_prompt_config.army_setting
        self.RoleSetting = constant_prompt_config.role_setting
        self.TroopInformation = constant_prompt_config.troop_information
        self.json_constraint_variable = constant_prompt_config.json_constraint_variable
        self.action_list = constant_prompt_config.action_list
        self.action_property_definition = constant_prompt_config.action_property_definition
        self.action_instruction_block = constant_prompt_config.action_instruction_block
        self.map_info_json = constant_prompt_config.map_info_json
        
        ###
        self.initial_mission = initial_mission
        
        ###
        self.max_deploy_percent = max_deploy_percent
        self.prompt_max_deploy_percent = self.max_deploy_percent * 100
        self.prompt_max_deploy_nb = int(self.original_num_of_troops * self.max_deploy_percent)
        # Defeat thresholds (previously hardcoded 0.1 / 0.5 in parsed_data_sync).
        self.crushing_defeat_remaining_frac = crushing_defeat_remaining_frac
        self.crushing_defeat_lost_frac = crushing_defeat_lost_frac

        ### system setting
        self.current_stage = "In Battle"
        self.agent_clock = None
        self.round_nb = None
        self.round_interval = round_interval
        self.CurrentBattlefieldSituation = ""
        
        self.history_board = {}
        self.history_board["initial_mission"] = self.initial_mission
        
        self.current_action = initial_mission # tracks the current action
        self.troopType = ""
        
        self.init_position = position
        self.position_hist_dict = {}
        
        # self.streamlining_label = "Merge" #default is merge
        self.moral = "Medium"
        
    def position_updated_hist(self, round_nb,new_position):
        self.position_hist_dict[round_nb] = new_position
    
    def get_position_hist(self):
        return self.init_position, self.position_hist_dict    
            
    @property
    def lapse_time(self):
        return self.round_nb * self.round_interval
    
    @property
    def remaining_num_of_troops(self):
        return self.original_num_of_troops - self.deployed_num_of_troops - self.lost_num_of_troops
    

class Detachment_AgentHierarchy:
    def __init__(self, level, parent_agent=None):
        self.id = next_agent_id()  # deterministic, creation-ordered id (reproducible)
        
        self.level = level
        self.parent_agent = parent_agent
        self.sub_agents = []
        
        # self.target_agent_name_list = []
        self.target_agent_id = ""
        
    @property
    def target_agent_list(self):
        pass
    
    
    @property
    def parent_agent_id(self):
        """Returns the ID of the parent agent if it exists, otherwise None."""
        return self.parent_agent.hierarchy.id if self.parent_agent else None

    @property
    def sub_agent_ids(self):
        """Returns a list of IDs of sub-agents."""
        return [agent.hierarchy.id for agent in self.sub_agents]
    
    def add_sub_agent(self, sub_agent):
        self.sub_agents.append(sub_agent)
        sub_agent.parent_agent = self

class Detachment_Agent(BasicAgent):
    def __init__(self, model_type, profile, hierarchy):
        super().__init__(profile.identity, model_type)
        self.model_type = model_type
        self.profile = profile
        self.hierarchy = hierarchy
        self.LLM_response = None

        self.execute_nb = 0
        self.extracted_json_history = {}
        self.LLM_response_history = {}
        self.round_invalid_messages = []
        self.invalid_messages_history = []
        self.additional_prompt = "" # temporarily stores extra information, can be added to the profile later
        
        
        self.action_restrictions_require = False
        self.mergedOrPruned = False

        # Phase 4 perf/cost knobs
        self.history_window = 3       # last-K extracted JSON decisions in the prompt; <=0 = full raw history
        self.prompt_caching = False   # send the static prefix as a separate cache_control block (Claude)
        self._static_prefix_cache = None

        self.log_folder_name = ""

    def __eq__(self, other):
        return isinstance(other, Detachment_Agent) and self.hierarchy.id == other.hierarchy.id
    
    def __hash__(self):
        return hash(self.hierarchy.id)
    
    @property
    def prompt(self) -> str:
        return self.construct_prompt()

    # --- Parsing / validation (delegates to agent_parsing) ---
    def validate_parsed_output(self, json_data: dict) -> list:
        return agent_parsing.validate_parsed_output(json_data)

    def parse_llm_output(self, LLM_response: str):
        return agent_parsing.parse_llm_output(LLM_response)

    # --- Prompt construction (delegates to agent_prompting) ---
    def generate_action_list(self) -> str:
        return agent_prompting.generate_action_list(self)

    def _static_prefix(self) -> str:
        return agent_prompting.static_prefix(self)

    def _history_text(self) -> str:
        return agent_prompting.history_text(self)

    def _dynamic_suffix(self) -> str:
        return agent_prompting.dynamic_suffix(self)

    def prompt_parts(self) -> tuple[str, str]:
        """Returns (static_prefix, dynamic_suffix). Callers that batch or cache use these
        separately; ``construct_prompt`` joins them for the plain string path."""
        return agent_prompting.prompt_parts(self)

    def construct_prompt(self) -> str:
        return agent_prompting.construct_prompt(self)

    # --- State synchronization (delegates to agent_state_sync) ---
    def parsed_data_sync(self, parsed_json: dict) -> list:
        return parsed_data_sync(self, parsed_json)

    ### Misc method
    def create_sub_agent(self, action: dict) -> "Detachment_Agent":
        return create_sub_agent(self, action)


    def DEVELOPING_MODE_save_sync_results_with_pickle(self, sync_results, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(sync_results, file)
        
        
    def calculate_troop_deployments(self, actions):
        new_troops_deployed = 0
        for action in actions:
            # Extract the number of troops from each action and add to the total
            number_of_troops_in_this_action = action.get('number', -1)
            if number_of_troops_in_this_action == -1:
                raise ValueError("number of troop don't decide")
            
            if number_of_troops_in_this_action not in ["All available","All"]:
                new_troops_deployed += int(number_of_troops_in_this_action)
        return new_troops_deployed

    def get_logged_attributions(self):
        # Fetching attributes from the profile
        identity = self.profile.identity
        position = self.profile.position
        deployed_troops = self.profile.deployed_num_of_troops
        original_troops = self.profile.original_num_of_troops
        remaining_troops = self.profile.remaining_num_of_troops
        lost_num_of_troops = self.profile.lost_num_of_troops
        current_battlefield_situation = self.profile.CurrentBattlefieldSituation
        current_stage = self.profile.current_stage
        agent_clock = self.profile.agent_clock

        # Fetching attributes from the hierarchy
        agent_level = self.hierarchy.level
        
        parent_agent = self.hierarchy.parent_agent.hierarchy.id if self.hierarchy.parent_agent else None
        sub_agents = [agent.hierarchy.id for agent in self.hierarchy.sub_agents] if self.hierarchy.sub_agents else []


        # Organizing the fetched information into a dictionary
        attributes = {
            "identity": identity,
            "position": position,
            "troop_info": {
                "deployed": deployed_troops,
                "original": original_troops,
                "lost": lost_num_of_troops,
                "remaining": remaining_troops
            },
            "battlefield_situation": current_battlefield_situation,
            "hierarchy_info": {
                "level": agent_level,
                "parent_agent": parent_agent,
                "sub_agents": sub_agents
            },
            "system_setting": {
                "current_stage": current_stage,
                "agent_clock": agent_clock
            }
        }
        
        text_msg = f"This level {attributes['hierarchy_info']['level']} {attributes['identity']} unit is {attributes['system_setting']['current_stage']}. It is at {attributes['position']}. It initially has {attributes['troop_info']['original']} soldiers, out of which {attributes['troop_info']['deployed']} were deployed. {attributes['troop_info']['lost']} troops have been lost, and {attributes['troop_info']['remaining']} are still remaining.'"

        return attributes, text_msg

    
    def _invoke_decision(self):
        """Issue one commander decision call. With prompt caching enabled the static prefix is
        sent as a separate cache_control block; otherwise the plain run_model path is used (which
        tests monkeypatch)."""
        if self.prompt_caching:
            prefix, suffix = self.prompt_parts()
            return run_LLM(self.model_type, suffix, self.token_accumulator, "commander", cache_prefix=prefix)
        return self.run_model(self.prompt)

    def execute(self, LLM_response: str = None) -> dict:
        max_attempts = 3
        attempts = 0

        # update execution count
        self.hierarchy.target_agent_id = ""

        self.execute_nb += 1

        if LLM_response in [None, ""]:
            LLM_response = self._invoke_decision()


        # record LLM response history
        self.LLM_response_history[self.execute_nb] = LLM_response
        
        while attempts < max_attempts:
            extracted_json = self.parse_llm_output(LLM_response)
            logger.debug("[Execute #%s] Attempt #%s: Extracted JSON: %s", self.execute_nb, attempts + 1, extracted_json)
            if self.parser_mode == "structured":
                from pydantic import ValidationError
                from schemas import CommanderDecision
                try:
                    if not isinstance(extracted_json, dict):
                        raise TypeError(f"parse returned {type(extracted_json).__name__}, expected dict")
                    CommanderDecision.model_validate(extracted_json)
                    invalid_messages = []
                except (ValidationError, TypeError) as e:
                    invalid_messages = [str(e)]
            else:
                invalid_messages = self.validate_parsed_output(extracted_json)

            self.extracted_json_history[self.execute_nb] = extracted_json

            if invalid_messages != []:
                logger.warning("[Execute #%s] Attempt #%s: Invalid messages detected: %s", self.execute_nb, attempts + 1, invalid_messages)
                self.round_invalid_messages.append(invalid_messages)
                attempts += 1
                if attempts >= max_attempts:
                    raise AgentExecutionError(
                        f"Agent {self.hierarchy.id} failed to produce valid output after {max_attempts} attempts. "
                        f"Last errors: {invalid_messages}"
                    )
                LLM_response = self._invoke_decision()
            else:
                self.invalid_messages_history.append(self.round_invalid_messages)
                self.round_invalid_messages = []

                sync_results = self.parsed_data_sync(extracted_json)

                return {
                    "success": True,
                    "sync_results": sync_results,
                    "attempts": attempts,
                    "invalid_messages": invalid_messages
                }
        raise AgentExecutionError(f"Agent {self.hierarchy.id} exhausted retry loop without returning.")

    def execute_WithGpt4V(self):
        logger.info("execute_WithGpt4V log folder: %s", self.log_folder_name)

        max_attempts = 5
        attempts = 0

        # update execution count
        self.hierarchy.target_agent_id = ""

        self.execute_nb += 1
        plot_tactical_positions(self.profile.CurrentBattlefieldSituation, img_save_path = f"{self.log_folder_name}", img_name = f"{self.profile.identity}_{self.hierarchy.id}_{self.execute_nb}")

        image_path = f"{self.log_folder_name}/{self.profile.identity}_{self.hierarchy.id}_{self.execute_nb}.png"
        LLM_response = run_gpt4v(image_path, self.prompt)

        self.LLM_response_history[self.execute_nb] = LLM_response
        
        while attempts < max_attempts:
            extracted_json = self.parse_llm_output(LLM_response)
            logger.debug("[Execute #%s] Attempt #%s: Extracted JSON: %s", self.execute_nb, attempts + 1, extracted_json)

            self.extracted_json_history[self.execute_nb] = extracted_json

            if "Error" in extracted_json:
                invalid_messages = True
            else:
                invalid_messages = False

            if invalid_messages:
                logger.warning("[Execute #%s] Attempt #%s: Invalid messages detected: %s", self.execute_nb, attempts + 1, invalid_messages)
                if attempts == max_attempts:
                    raise Exception("Invalid messages after maximum attempts.")
                else:
                    LLM_response = run_gpt4v(r"logs\tactical_positions_plot.png", self.prompt)
                    attempts += 1
            else:
                # Synchronize data based on parsed JSON
                sync_results = self.parsed_data_sync(extracted_json)
                
                return {
                    "success": True,
                    "sync_results": sync_results,
                    "attempts": attempts,
                    "invalid_messages": invalid_messages
                }
        return

# tree visiualization
def build_tree(tree, hierarchy_node, parent_id=None):
    tree.create_node(tag=hierarchy_node.id, identifier=hierarchy_node.id, parent=parent_id)
    for sub_agent in hierarchy_node.sub_agents:
        build_tree(tree, sub_agent, parent_id=hierarchy_node.id)
