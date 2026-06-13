import argparse
from dotenv import load_dotenv
load_dotenv()

from prompt.map_setting_of_other_battles import map_info_json_Agincourt, map_info_json_Falkirk, map_info_json_Poitiers
from prompt.agent_profile_Poitiers import country_E_Army_Poitiers, country_F_Army_Poitiers, System_Setting_Poitiers, History_Setting_Poitiers
from prompt.agent_profile_Falkirk import country_E_Army_Falkirk, country_F_Army_Falkirk, System_Setting_Falkirk, History_Setting_Falkirk
from prompt.agent_profile_Agincourt import country_E_Army_Agincourt, country_F_Army_Agincourt, System_Setting_Agincourt, History_Setting_Agincourt
from prompt.Detachment_Agent_prompt import (
    RoleSetting,
    TroopInformation,
    action_instruction_block,
    json_constraint_variable,
    json_example_text,
)
from prompt.action_space_setting import action_list, action_property_definition

from agent import ConstantPromptConfig, Detachment_Agent, Detachment_AgentHierarchy, Detachment_AgentProfile
from sandbox import Sandbox

class ConflictConfig():
    def __init__(self, battle_name):
        self.battle_name = battle_name

    def get_opposing_agent_profile(self):
        # Battle configuration
        if self.battle_name == "Poitiers":
            country_E_agent_profile = Detachment_AgentProfile(
                identity="country_E",
                position=[15, -10],  
                original_num_of_troops=6000,  
                constant_prompt_config = country_E_constant_prompt_config
            )

            country_F_agent_profile = Detachment_AgentProfile(
            identity="country_F", 
                position=[-10, 5],  
                original_num_of_troops=15000,  
                constant_prompt_config = country_F_constant_prompt_config 
            )
        elif self.battle_name == "Falkirk":
            country_E_agent_profile = Detachment_AgentProfile(
                identity="country_E",
                position=[0, 0],  
                original_num_of_troops=15000,  
                constant_prompt_config = country_E_constant_prompt_config
            )

            country_F_agent_profile = Detachment_AgentProfile(
            identity="country_F", 
                position=[50, 0],  
                original_num_of_troops=6000,  
                constant_prompt_config = country_F_constant_prompt_config 
            )
            
        elif self.battle_name == "Agincourt":
            country_E_agent_profile = Detachment_AgentProfile(
                identity="country_E",
                position= [0,-100],  
                original_num_of_troops=6500,  
                constant_prompt_config = country_E_constant_prompt_config
            )

            country_F_agent_profile = Detachment_AgentProfile(
                identity="country_F", 
                position=[15, -50], 
                original_num_of_troops=35000, 
                constant_prompt_config = country_F_constant_prompt_config 
            )
        else:
            return "Invalid conflict name. Please choose from 'Poitiers', 'Falkirk', or 'Agincourt'."

        return country_E_agent_profile, country_F_agent_profile

    def get_prompt_config_args(self):
        if self.battle_name == "Poitiers":
            map_info_json_Type = map_info_json_Poitiers
            System_Setting_Type = System_Setting_Poitiers
            History_Setting_Type = History_Setting_Poitiers
            country_E_Army_Type = country_E_Army_Poitiers
            country_F_Army_Type = country_F_Army_Poitiers
        elif self.battle_name == "Falkirk":
            map_info_json_Type = map_info_json_Falkirk
            System_Setting_Type = System_Setting_Falkirk
            History_Setting_Type = History_Setting_Falkirk
            country_E_Army_Type = country_E_Army_Falkirk
            country_F_Army_Type = country_F_Army_Falkirk
        elif self.battle_name == "Agincourt":
            map_info_json_Type = map_info_json_Agincourt
            System_Setting_Type = System_Setting_Agincourt
            History_Setting_Type = History_Setting_Agincourt
            country_E_Army_Type = country_E_Army_Agincourt
            country_F_Army_Type = country_F_Army_Agincourt
        else:
            return "Invalid conflict name. Please choose from 'Poitiers', 'Falkirk', or 'Agincourt'."
        return System_Setting_Type, History_Setting_Type, country_E_Army_Type, country_F_Army_Type, map_info_json_Type


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run a conflict simulation.') 
    parser.add_argument('--conflict_name', type=str, choices=['Poitiers', 'Falkirk', 'Agincourt'], default= "Poitiers",help='choose conflict name') 
    parser.add_argument('--LLM_MODEL', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default="gpt", help='Language model to use')
    parser.add_argument("--is_GPT4V_activate", type=int, default=0, help="Use GPT-4 V instead of GPT-4")
    parser.add_argument('--simulation_time', type=int, default=90, help='Number of minutes to simulate')
    parser.add_argument('--update_interval', type=int, default=15, help='Interval for simulation updates')
    
    parser.add_argument('--have_diaries', type=int, default=0, help='Whether to have diaries')
    parser.add_argument('--continue_run', type=int, default=1, help='Whether to continue run')
    parser.add_argument('--vision_range', type=int, default=100000, help='Fog-of-war radius in map units (default 100000 = omniscient)')
    parser.add_argument('--on_agent_error', type=str, choices=['continue', 'abort'], default='continue', help='What to do when an agent raises AgentExecutionError')
    parser.add_argument('--parser', type=str, choices=['legacy', 'structured'], default='legacy', help='JSON validation mode: legacy=manual checks, structured=Pydantic models')
    parser.add_argument('--commander_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for commander agents (defaults to --LLM_MODEL)')
    parser.add_argument('--referee_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for referee (defaults to --LLM_MODEL)')
    parser.add_argument('--diary_model', type=str, choices=["claude", "gpt", "openrouter", "ollama", "fake"], default=None, help='LLM for diary soldiers (defaults to --LLM_MODEL)')

    args = parser.parse_args()
    
    LLM_MODEL = args.LLM_MODEL
    simulation_time = args.simulation_time
    update_interval = args.update_interval
    conflict_name = args.conflict_name
    GPT4V = bool(args.is_GPT4V_activate)
    Is_have_diaries = bool(args.have_diaries)
    Does_continue_run = bool(args.continue_run)
    vision_range = args.vision_range
    on_agent_error = args.on_agent_error
    parser_mode = args.parser
    commander_model = args.commander_model or LLM_MODEL
    referee_model = args.referee_model or LLM_MODEL
    diary_model = args.diary_model or LLM_MODEL

    if LLM_MODEL not in ("gpt", "fake") and GPT4V == True:
        raise ValueError("GPT-4 V is only available for GPT model.")
    
    if LLM_MODEL == "gpt" and GPT4V == True:
        model_name_to_log = "gpt4V"
    else:
        model_name_to_log = LLM_MODEL
    
    LOG_FOLER_NMAE = f"{conflict_name}_{model_name_to_log}_{simulation_time}_{update_interval}"
    print(f"LOG_FOLER_NMAE: {LOG_FOLER_NMAE}")

    conflict_config = ConflictConfig(conflict_name)
    System_Setting_Type, History_Setting_Type, country_E_Army_Type, country_F_Army_Type,map_info_json_Type = conflict_config.get_prompt_config_args()
    
    country_E_constant_prompt_config = ConstantPromptConfig(
        System_Setting=System_Setting_Type,
        History_Setting=History_Setting_Type,
        
        army_setting=country_E_Army_Type,
        
        role_setting=RoleSetting,
        troop_information=TroopInformation,
        json_constraint_variable=json_constraint_variable,
        json_example_text=json_example_text,
        action_list=action_list,
        action_property_definition=action_property_definition,
        action_instruction_block=action_instruction_block,
        map_info_json=map_info_json_Type,
        additional_settings={} 
    )

    country_F_constant_prompt_config = ConstantPromptConfig(
        System_Setting=System_Setting_Type,
        History_Setting=History_Setting_Type,
        
        army_setting=country_F_Army_Type,
        
        role_setting=RoleSetting,
        troop_information=TroopInformation,
        json_constraint_variable=json_constraint_variable,
        json_example_text=json_example_text,
        action_list=action_list,
        action_property_definition=action_property_definition,
        action_instruction_block=action_instruction_block,
        map_info_json=map_info_json_Type,
        additional_settings={}  
    )
    
    country_E_agent_profile, country_F_agent_profile = conflict_config.get_opposing_agent_profile()

    country_E_agent_hierarchy = Detachment_AgentHierarchy(level = 1, parent_agent= None)
    country_F_agent_hierarchy = Detachment_AgentHierarchy(level = 1, parent_agent= None)
    
    country_E_agent_root = Detachment_Agent(commander_model, country_E_agent_profile, country_E_agent_hierarchy)
    country_F_agent_root = Detachment_Agent(commander_model, country_F_agent_profile, country_F_agent_hierarchy)
    country_E_agent_root.parser_mode = parser_mode
    country_F_agent_root.parser_mode = parser_mode

    country_E_agent_hierarchy.parent_agent = country_E_agent_root
    country_F_agent_hierarchy.parent_agent = country_F_agent_root

    # conflict name and time only used for logging
    sandbox = Sandbox(LLM_MODEL, map_info_json_Type, LOG_FOLER_NMAE, "1300-01-01 12:00", country_E_agent_root, country_F_agent_root,
                      referee_model=referee_model, diary_model=diary_model)

    sandbox.have_diaries = Is_have_diaries
    sandbox.continue_run = Does_continue_run
    sandbox.GPT4V = GPT4V
    sandbox.LLM_MODEL = LLM_MODEL
    sandbox.vision_range = vision_range
    sandbox.on_agent_error = on_agent_error
    
    simulation_results = sandbox.simulate(simulation_time, update_interval)