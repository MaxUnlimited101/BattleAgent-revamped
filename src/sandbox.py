# Standard library imports
import os
import pickle
import random
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm

# Third-party imports
from treelib import Tree

# Local application/library-specific imports
from agent import AgentExecutionError
from utils.shared_func import vision_decoder
from group_experience.individual_profile import Soldier_Profiles, SoldierCollector
from support_agents.referee import Action_Interact_Evaluation
from support_agents.sim_stopper import ceasefire_decision_maker
random.seed(42)


                
class BattleLogger:
    def __init__(self, campaign_name):
        self.campaign_name = campaign_name
        self.logs = []
        self.log_subdirectory = self._setup_logging()

    def _setup_logging(self):
        """Set up logging directories and files."""
        real_world_time_at_creation = datetime.now().strftime('%m%d-%H%M_%S')
        log_subdirectory = f"{real_world_time_at_creation}_{self.campaign_name}"

        script_directory = os.path.dirname(os.path.realpath(__file__))
        log_directory = os.path.join(script_directory, "logs")
        self.log_directory_path = os.path.join(log_directory, log_subdirectory)
        os.makedirs(self.log_directory_path, exist_ok=True)
        return log_subdirectory

    def log_action(self, action, info=None, system_time=None):
        time_str = system_time.strftime('%Y-%m-%d %H:%M') if system_time else datetime.now().strftime('%Y-%m-%d %H:%M')
        log_entry = f"{time_str}: {action}"
        if info is not None:
            log_entry += f" - {info}"
        
        self.logs.append(log_entry)
        self.save_log_to_file(log_entry, system_time)
        
    def log_tree(self, hierarchy_root, label, system_time):
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
        
        
    def log_war_situation(self, label, war_situation, decision, system_time):
        situation_summary = (
            f"{label} War Situation - Total Agents: {war_situation['total_agents']}, "
            f"Command Structure Impact: {war_situation['command_structure_impact']}, "
            f"Morale Collapse Impact: {war_situation['morale_collapse_impact']}, "
            f"Heavy Casualties: {war_situation['heavy_casualties_count']}, "
            f"Total Troops: {war_situation['total_troops']}"
        )
        self.log_action("War Situation and Decision", situation_summary, system_time)

    def save_log_to_file(self, log_entry, system_time):
        filename = os.path.join(self.log_directory_path, f"{system_time.strftime('%Y%m%d-%H%M')}_simulation.log") if system_time else "general.log"
        with open(filename, 'a') as file:
            file.write(log_entry + "\n")


class Sandbox:
    def __init__(self, model_type, map_info_json, campaign_name, start_time_str, country_E_agent_root, country_F_agent_root, subAgentNBThreshold=5, referee_model=None, diary_model=None):
        from utils.token_accounting import TokenAccumulator

        # Basic properties
        self.model_type = model_type
        self.campaign_name = campaign_name
        self.map_info_json = map_info_json  # Map information

        # Agent roots and lists
        self.country_E_agent_root = country_E_agent_root
        self.country_F_agent_root = country_F_agent_root

        self.country_E_agent_list = []
        self.country_F_agent_list = []

        # Time initialization
        self.system_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')

        # Logging setup
        self.battle_logger = BattleLogger(self.campaign_name)

        self.country_F_collector = SoldierCollector(Soldier_Profiles["country_F"], initial_root_agent=country_F_agent_root, model_type=self.model_type)
        self.country_E_collector = SoldierCollector(Soldier_Profiles["country_E"], initial_root_agent=country_E_agent_root, model_type=self.model_type)

        self.continue_run = False
        self.have_diaries = True
        self.GPT4V = None
        self.vision_range = 100000
        self.on_agent_error = "continue"

        self.subAgentNBThreshold = subAgentNBThreshold

        self.action_interact_evaluation = Action_Interact_Evaluation(self.model_type, self.shuffled_agent_list, self.country_E_agent_list, self.country_F_agent_list, self.map_info_json)

        # Apply per-role model overrides
        if referee_model:
            self.action_interact_evaluation.model_type = referee_model
        if diary_model:
            self.country_E_collector.model_type = diary_model
            self.country_F_collector.model_type = diary_model
            for soldier_agent in self.country_E_collector.soldier_agents_list:
                soldier_agent.profile.model_type = diary_model
            for soldier_agent in self.country_F_collector.soldier_agents_list:
                soldier_agent.profile.model_type = diary_model

        # Token accounting — wire accumulator to all sub-components
        self.token_accumulator = TokenAccumulator()
        self.country_E_agent_root.token_accumulator = self.token_accumulator
        self.country_F_agent_root.token_accumulator = self.token_accumulator
        self.action_interact_evaluation.token_accumulator = self.token_accumulator
        self.country_E_collector.token_accumulator = self.token_accumulator
        self.country_F_collector.token_accumulator = self.token_accumulator
        for soldier_agent in self.country_E_collector.soldier_agents_list:
            soldier_agent.profile.token_accumulator = self.token_accumulator
        for soldier_agent in self.country_F_collector.soldier_agents_list:
            soldier_agent.profile.token_accumulator = self.token_accumulator
        
    @property
    def shuffled_agent_list(self):
        combined_list = self.country_E_agent_list + self.country_F_agent_list
        random.shuffle(combined_list)
        return combined_list

    def advance_time(self, minutes):
        # Advance time forward and update all agents
        self.system_time += timedelta(minutes=minutes)
        self.battle_logger.log_action("Time advanced", minutes, self.system_time)

    def update_agent_list_recursive(self, agent_node, agent_list):
        """
        Recursively traverse the agent hierarchy and update the agent list.
        """
        agent_list.append(agent_node)
        for sub_agent in agent_node.hierarchy.sub_agents:
            self.update_agent_list_recursive(sub_agent, agent_list)

    def update_all_agent_lists(self):
        """
        Update both country_E and France agent lists from their root nodes.
        """
        self.country_E_agent_list.clear()
        self.country_F_agent_list.clear()
        self.update_agent_list_recursive(self.country_E_agent_root, self.country_E_agent_list)
        self.update_agent_list_recursive(self.country_F_agent_root, self.country_F_agent_list)

    def update_agent_list_after_execution(self, agent):
        """
        Update the agent list after executing an agent's actions.
        Check for changes in agent.hierarchy.sub_agents.
        """
        if agent.profile.identity in ["country_E"]:
            agent_list = self.country_E_agent_list
        elif agent.profile.identity in ["country_F"]:
            agent_list = self.country_F_agent_list
        else:
            # Log the unknown nationality for debugging
            self.battle_logger.log_action("Unknown nationality encountered", agent.profile.identity, self.system_time)
            raise ValueError(f"Unknown nationality: {agent.profile.identity}")

        current_sub_agents = set(agent.hierarchy.sub_agents)
        existing_sub_agents = set([a for a in agent_list if a.hierarchy.parent_agent == agent])

        if current_sub_agents != existing_sub_agents:
            # Update agent list if there's a change
            agent_list.clear()
            self.update_agent_list_recursive(self.country_E_agent_root if agent.profile.identity in ["country_E"] else self.country_F_agent_root, agent_list)

    def my_vision_decoder(self, visionRequest_agent, extCallType=None):
        return vision_decoder(visionRequest_agent, self.shuffled_agent_list, extCallType, vision_range=self.vision_range)
        
    def save_to_file(self):
        formatted_time = self.system_time.strftime('%Y-%m-%d %H_%M_%S')
        filename = os.path.join(self.battle_logger.log_directory_path, f"{formatted_time}_sandbox.pkl")
        with open(filename, 'wb') as file: 
            pickle.dump(self, file)
        print(f"Sandbox saved to {filename}")
    
    def run_referee_and_log(self, agent):
        self.action_interact_evaluation.para_update(self.shuffled_agent_list, self.country_E_agent_list, self.country_F_agent_list)
        message = self.action_interact_evaluation.single_agent_evaluate(agent)
        if message!=None:
            self.battle_logger.log_action("Action Interaction Evaluation", message, self.system_time)
                        
    def simulate(self, total_minutes, step_minutes):

        results = []
        steps = total_minutes // step_minutes
        failed_agent_count = 0

        self.update_all_agent_lists()
        
        country_E_init_vision_info = self.my_vision_decoder(self.country_E_agent_root)
        self.country_E_agent_root.profile.CurrentBattlefieldSituation =  country_E_init_vision_info
        
        country_F_init_vision_info = self.my_vision_decoder(self.country_F_agent_root)
        self.country_F_agent_root.profile.CurrentBattlefieldSituation =  country_F_init_vision_info
        
        self.country_F_agent_root.log_folder_name  = self.battle_logger.log_directory_path
        self.country_E_agent_root.log_folder_name  = self.battle_logger.log_directory_path
        
        self.battle_logger.log_action("parameter", f"GPT4V:{self.GPT4V}, LLM_MODEL: {self.LLM_MODEL} continue_run:{self.continue_run}, have_diaries:{self.have_diaries}", self.system_time)
        
        for step in range(steps):
            self.advance_time(step_minutes)
            step_results = {
                "step": step + 1, 
                "time": self.system_time.strftime('%Y-%m-%d %H:%M'), 
                "agent_states": []
            }

            # for agent in self.shuffled_agent_list:
            shuffled_list = self.shuffled_agent_list
            for agent in tqdm(shuffled_list):
                diary_index = 0
                print(f"current agent list length: {len(self.shuffled_agent_list)}")
                
                if agent.mergedOrPruned == True:
                    continue
                
                if agent.profile.remaining_num_of_troops <= 0:
                    continue
                
                try:
                    if agent.profile.current_stage not in ["Crushing Defeat", "fleeing Off the Map"]: 
                        
                        if len(agent.hierarchy.sub_agents) < self.subAgentNBThreshold:
                            agent.action_restrictions_require = False
                        else:
                            agent.action_restrictions_require = True
                            
                        
                        vision_info = self.my_vision_decoder(agent)
                        # print(f"vision_info:{vision_info}")
                        agent.profile.CurrentBattlefieldSituation = vision_info
                        
                        #sandbox-agent sync
                        agent.profile.agent_clock = self.system_time  # Update agent clock
                        agent.profile.round_nb = step
                        agent.profile.round_interval = step_minutes
                        
                        if self.GPT4V:
                            agent.execute_WithGpt4V()    
                        else:
                            agent.execute()  
                        
                        self.update_agent_list_after_execution(agent)
                        
                        if agent.profile.identity == self.country_E_agent_root.profile.identity: #["country_E"]:
                            collector = self.country_E_collector
                            # same_identity_agent_list = self.country_E_agent_list
                        else:
                            collector = self.country_F_collector
                            # same_identity_agent_list = self.country_F_agent_list
                        
                        # MergeFunc(agent)
                        
                        self.run_referee_and_log(agent)
                        for sub_agent in agent.hierarchy.sub_agents:
                            if sub_agent.new_born:
                                self.run_referee_and_log(sub_agent)
                                sub_agent.new_born = False

                        soldier_agents_list = collector.get_soldiers(agent)
                        if self.have_diaries:
                            for soldier_agent in soldier_agents_list:
                                soldier_agent.execute(agent, agent.profile.current_action, vision_info, self.system_time)
                                print(f"diary {diary_index} has been collected")
                                print("--------------")
                                diary_index += 1
                                self.battle_logger.log_action("diary", f"diary {diary_index} has been collected", self.system_time)

                        # Collect agent-specific logs
                        agent_logs, text_msg = agent.get_logged_attributions()
                        
                        log_content = text_msg #+ str(agent_logs)
                        self.battle_logger.log_action(f"{agent.profile.identity} {agent.hierarchy.id} executed {agent.profile.current_action}", log_content, self.system_time)

                        # Collect agent states and actions
                        step_results["agent_states"].append({
                            "agent_id": agent.hierarchy.id, 
                            # "state": agent.state
                        })

                except AgentExecutionError as e:
                    failed_agent_count += 1
                    error_msg = f"AgentExecutionError for agent {agent.hierarchy.id} at step {step + 1}: {e}"
                    self.battle_logger.log_action("AgentExecutionError", error_msg, self.system_time)
                    if self.on_agent_error == "abort":
                        raise
                except Exception as e:
                    error_msg = f"Unexpected error with agent {agent.hierarchy.id} at step {step + 1}: {e}"
                    self.battle_logger.log_action("UnexpectedError", f"{error_msg}\n{traceback.format_exc()}", self.system_time)
                    raise
            
            country_F_war_situation, country_F_decision = ceasefire_decision_maker(self.country_F_agent_root)
            country_E_war_situation, country_E_decision = ceasefire_decision_maker(self.country_E_agent_root)

            self.battle_logger.log_war_situation("country_F", country_F_war_situation, country_F_decision, self.system_time)
            self.battle_logger.log_war_situation("country_E", country_E_war_situation, country_E_decision, self.system_time)

            if country_F_decision is not None or country_E_decision is not None:
                decision_summary = "Ceasefire/Surrender Decisions: "
                if country_F_decision:
                    decision_summary += f"country_F forces decision: {country_F_decision}. "
                if country_E_decision:
                    decision_summary += f"country_E forces decision: {country_E_decision}."

                # Log the decision
                self.battle_logger.log_action("Ceasefire/Surrender Decision", decision_summary, self.system_time)
                
                # Break out of the simulation loop if any decision is made
                if country_F_decision or country_E_decision:
                    print("Decision made, simulation stopped")
                    if not self.continue_run:
                        break

            results.append(step_results)

            self.battle_logger.log_action("TokenUsage", str(self.token_accumulator.summary_dict()), self.system_time)
            self.battle_logger.log_action("Simulation step completed", f"Step {step + 1}", self.system_time)
            self.battle_logger.log_tree(self.country_E_agent_root, "country_E", self.system_time)
            self.battle_logger.log_tree(self.country_F_agent_root, "country_F", self.system_time)

            
            self.save_to_file()

        skipped_rounds = self.action_interact_evaluation.skipped_casualty_rounds
        token_summary = str(self.token_accumulator.summary_dict())
        cost = self.token_accumulator.cost_estimate(self.model_type)
        cost_str = f" | Estimated cost: ${cost:.4f}" if cost is not None else ""
        self.battle_logger.log_action("FinalTokenSummary", token_summary + cost_str, self.system_time)

        summary = (
            f"Run complete: {steps} steps, {failed_agent_count} agent execution failures, "
            f"{skipped_rounds} casualty rounds skipped."
        )
        print(summary)
        self.battle_logger.log_action("RunSummary", summary, self.system_time)

        return results


