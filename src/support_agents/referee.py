import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)


from utils.LLM_api import run_LLM
import json5 as json
import logging

logger = logging.getLogger(__name__)

def external_action_binding(agent, enemy_troop_list, friendly_troop_list, shuffled_agent_list):
    attacked_by_agent_list = []
    target_agent_id = agent.hierarchy.target_agent_id

    logger.debug("target_agent_id: %s", target_agent_id)
    logger.debug("enemy_list %s", [i.hierarchy.id for i in enemy_troop_list])
    if target_agent_id  != None:
        attacked_agent = next((agent for agent in enemy_troop_list if agent.hierarchy.id == target_agent_id), None)
        if attacked_agent:
            attacked_by_agent_list.append(attacked_agent)
    attacker_dict = {}
    logger.debug("attacked_by_agent_list: %s", attacked_by_agent_list)
    return attacker_dict, attacked_by_agent_list

def external_construct_judgment_prompt(agent, attacker_dict, attacked_by_agent_list ,  map_info):
    """
    Generate a detailed prompt for military casualty assessment involving multiple agents.

    Parameters:
    - agent: The primary agent involved in the conflict.
    - attackers: A list of attackers including their actions and remaining troops.
    - map_info: Information about the map and strategic locations.
    - weapon_info: Details on weaponry and its implications on the conflict.

    Returns:
    - A string containing the detailed prompt for analysis.
    """
    
    agent_id = agent.hierarchy.id
    agent_location = agent.profile.position
    remaining_troops = agent.profile.remaining_num_of_troops  
    current_action = agent.profile.current_action  
    troopType = agent.profile.troopType

    map_info_summary = "Map features include: " + ", ".join([
        f"{location} (Coordinates: {details.get('coordinates', '[Not specified]')}): {details['description']}"
        for location, details in map_info["Geography"].items()
    ]) + ". One coordinate unit is equivalent to 10 yards in the simulated environment."

    agent_details = f"""Agent {agent_id}, with {remaining_troops} {troopType} soldier at coordinates {agent_location}, is executing a "{current_action}"."""
     
    under_attacked_agent_info = ""
    for under_attacked_agent in attacked_by_agent_list:
        defense_action = under_attacked_agent.profile.current_action 
        under_attacked_agent_remaining_troops = under_attacked_agent.profile.remaining_num_of_troops 
        under_attacked_agent_info += f"""Agent {under_attacked_agent.hierarchy.id}, with {under_attacked_agent_remaining_troops} troops at coordinates {under_attacked_agent.profile.position}, is executing a "{defense_action}" for defense.\n"""
    
    logger.debug("referee prompt display: agent_details=%s attackers_info=%s", agent_details, under_attacked_agent_info)

    
    # Instructions
    instructions = """
Your analysis should begin with a step-by-step assessment of the actions taken by each agent, focusing on:
- The sequence of events.
- The remaining number of troops.
- The impact of each action on strategic objectives.
- Potential losses from both offensive and defensive actions.

Quantifying your assessment is the first priority. Consider the balance of forces, deployment of tactics, terrain, and weapon utilization in your predictions.
"""

    # Constructing the final prompt
    prompt = (
"You are tasked to evaluate the actions and predict the military casualties involving the upcoming conflict between multiple agents.\n\n"

f"## agent information\n{agent_details}\n"

f"{under_attacked_agent_info}\n"

f"## Map information \n{map_info_summary}\n\n"

f"## Weapon Information \n Here's some weapon information for your reference if previously mentioned. \n (1)The longbow was capable of discharging up to ten arrows per minute, reaching distances well over 300 meters. The longbowmen not only outranged their opponents but also had a rate of fire more than three times greater than their crossbowmen, leading to significant casualties. \n (2)The heavy cavalry faced disadvantages on muddy terrain, particularly at the bottom of slopes. \n\n"

f"{instructions}\n\n"

"Please summarize the predicted casualties, including explanations for each agent's outcomes, in the specified JSON format.\n\n"



    "# {\n"
    "#   \"casualties_result\": [\n"
    "#     {\n"
    "#       \"agent_id\": \"string // <unique identifier for the military unit or agent>\",\n"
    "#       \"estimated_loss_percentage\": \"number // <estimated percentage of the unit lost or injured>\"\n"
    "#       \"casualties\": \"int // <number of individuals lost or injured>\",\n"
    "#     },\n"
    "#     //other agent_id...\n"
    "#   ]\n"
    "# }\n"
    )

    return prompt

class Action_Interact_Evaluation:
    def __init__(self, model_type ,shuffled_agent_list, country_E_agent_list, country_F_agent_list, map_info_json, BattleLoggerAPI = None):
        self.shuffled_agent_list = shuffled_agent_list
        self.country_E_agent_list = country_E_agent_list
        self.country_F_agent_list = country_F_agent_list
        self.map_info_json = map_info_json
        
        self.model_type = model_type
        self.token_accumulator = None

        self.message_list = []
        self.LLM_response_history = []

        self.prompt_history = []
        self.skipped_casualty_rounds = 0
        # Set by the sandbox for the structured event stream (optional).
        self.event_logger = None
        self.current_step = None
    def para_update(self, shuffled_agent_list, country_E_agent_list, country_F_agent_list):
        self.shuffled_agent_list = shuffled_agent_list
        self.country_E_agent_list = country_E_agent_list
        self.country_F_agent_list = country_F_agent_list
    
    def action_binding(self, agent, enemy_troop_list, friendly_troop_list): #self.shuffled_agent_list
        return external_action_binding(agent, enemy_troop_list, friendly_troop_list, self.shuffled_agent_list)
    
    def construct_judgment_prompt(self, agent, attacker_dict, attacked_by_agent_list): #self.map_info_json
        return external_construct_judgment_prompt(agent, attacker_dict, attacked_by_agent_list, self.map_info_json)
                
    def single_agent_evaluate(self, agent):

        message = None

        if agent in self.country_E_agent_list:
            enemy_troop_list = self.country_F_agent_list
            friendly_troop_list = self.country_E_agent_list
        else:
            enemy_troop_list = self.country_E_agent_list
            friendly_troop_list = self.country_F_agent_list

        _, attacked_by_agent_list = self.action_binding(agent, enemy_troop_list, friendly_troop_list)
        
        # if attacker_dict or attacked_by_agent_list:
        if attacked_by_agent_list:
            prompt = self.construct_judgment_prompt(agent, None, attacked_by_agent_list)
            self.prompt_history.append((agent,prompt))
            
            
            LLM_response = self.run_model(prompt)

            logger.debug("referee LLM response: %s", LLM_response)

            self.LLM_response_history.append((agent,LLM_response))
            
            parsed_json = self.parse_llm_output(LLM_response)
            if parsed_json == {}:
                attempts = 0
                while attempts < 3:
                    LLM_response = self.run_model(prompt)
                    parsed_json = self.parse_llm_output(LLM_response)
                    if parsed_json:
                        break
                    attempts += 1
                if parsed_json == {}:
                    logger.warning("referee failed to parse casualty JSON after retries for agent %s; skipping round.", agent.hierarchy.id)
                    self.skipped_casualty_rounds += 1
                    return None

            logger.debug("referee parsed casualties: %s", parsed_json)
            message = self.parsed_data_sync(agent, parsed_json)
            
            self.message_list.append(message)
            
        return message
    
    def run_model(self, prompt):
        return run_LLM(self.model_type, prompt, self.token_accumulator, "referee")
    
    
    def parse_llm_output(self, LLM_response):
        def extract_json_from_text(text):
            """
            Extracts the JSON part from a given string which may contain mixed content.
            It assumes that the JSON part starts with '{' and ends with '}'.
            """
            try:
                start_index = text.index('{')
                end_index = text.rindex('}') + 1
                json_part = text[start_index:end_index]
                parsed_json = json.loads(json_part)
                return parsed_json
            except (ValueError, Exception) as e:
                logger.warning("Error extracting or parsing referee JSON: %s", e)
                return {}

        extracted_json = extract_json_from_text(LLM_response)

        if isinstance(extracted_json, dict):
            return extracted_json
        else:
            logger.warning("Referee LLM response does not contain valid JSON data.")
            # You could either return an empty dict or handle this scenario explicitly in your code.
            return {}
    
    def parsed_data_sync(self, agent, parsed_json):
        casualties_result = parsed_json.get("casualties_result", [])
        message = ""

        if not casualties_result:
            return "No casualties data to process."

        for agent_casualty_result in casualties_result:
            if "agent_id" in agent_casualty_result:
                for shuffled_agent in self.shuffled_agent_list:  
                    if shuffled_agent.hierarchy.id == agent_casualty_result["agent_id"]:
                        try:
                            total_casualties = agent_casualty_result.get("casualties", 0)
                            if isinstance(total_casualties, str):
                                total_casualties = int(total_casualties) 
                            elif not isinstance(total_casualties, int):
                                raise ValueError("Casualties must be an integer or string that represents an integer")
                        except ValueError:
                            message += f"casualty skipped for {shuffled_agent.hierarchy.id} agent this round."
                            continue  

                        shuffled_agent.profile.lost_num_of_troops += total_casualties
                        message += f" {shuffled_agent.profile.identity} agent {shuffled_agent.hierarchy.id} lost {total_casualties}."
                        if self.event_logger is not None:
                            self.event_logger.casualties_assessed(
                                step=self.current_step,
                                agent_id=shuffled_agent.hierarchy.id,
                                casualties=total_casualties,
                                estimated_loss_percentage=agent_casualty_result.get("estimated_loss_percentage"),
                            )
                        break

        return message if message else "Agent ID not found or no updates made."
