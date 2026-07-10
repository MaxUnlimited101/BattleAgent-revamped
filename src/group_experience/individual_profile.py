import sys
import os
import json
import random


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)

from utils.LLM_api import run_LLM, run_LLM_batch



class Soldier_Hierarchy:
    def __init__(self, current_troop=None, sub_agents=None):
        self.current_troop = current_troop
        self.sub_troop = list(current_troop.hierarchy.sub_agents)

    def monitor_structure_change(self, current_troop):
        new_sub_agents = current_troop.hierarchy.sub_agents
        new_detached_troop_list = [agent for agent in new_sub_agents if agent not in self.sub_troop]
        self.sub_troop = list(new_sub_agents)
        return new_detached_troop_list

    def calculate_transfer_probability_and_decide(self, current_troop, new_detached_troop_list):
        original_num_of_troops = current_troop.profile.original_num_of_troops
        total_troops = sum(new_troop.profile.original_num_of_troops for new_troop in new_detached_troop_list) + original_num_of_troops
        stay_probability = original_num_of_troops / total_troops
        
        probabilities = [(current_troop, stay_probability)]  # includes the probability of staying in place
        for new_troop in new_detached_troop_list:
            transfer_probability = new_troop.profile.original_num_of_troops / total_troops
            probabilities.append((new_troop, transfer_probability))

        # decide the destination based on probability
        decision = random.choices(population=probabilities, weights=[prob[1] for prob in probabilities], k=1)[0]
        return decision[0]  # return the selected troop

class Soldier_Profile:
    def __init__(self, profile, model_type = None):
        self.name = profile['Name']
        self.age = profile['Age']
        self.family = profile['Family']
        self.occupation = profile['Occupation']
        self.personality =  profile['Personality']
        self.social_status = profile['Social Status']
        self.potential_illness = profile['Potential Illness']
        self.body_condition = profile['Body Condition']
        self.hobbies_and_interests = profile['Hobbies and Interests']
        self.style_of_talking = profile['Style of Talking']
        self.unique_quirks = profile['Unique Quirks']
        self.secrets_or_scandals = profile['Secrets or Scandals']
        self.journal = {}

        self.model_type = model_type
        self.token_accumulator = None
        self.injury_prob = 0.3  # overridable via SimulationConfig.diary_injury_prob

        self.injury_list = []
        
    def injury_generator(self):
        # injury location: [left leg, right leg, head....]
        # injury cause: [blunt weapon, knife, sword, trampling, horse]
        # injury severity: [minor injury, moderate injury, severe injury]
        # post-injury condition: [restricted movement, pain, dizziness, excessive blood loss]
        injury_part = ['left leg', 'right leg', 'head', 'left arm', 'right arm', 'chest', 'back', 'abdomen']
        injury_reason = ['blunt weapon', 'knife', 'sword', 'trampling', 'horse']
        injury_degree = ['minor injury', 'moderate injury', 'severe injury']
        injury_status = ['inconvenient movement', 'pain', 'dizziness', 'excessive blood loss']
        # injury probability configurable via SimulationConfig.diary_injury_prob (default 0.3)
        if random.random() < self.injury_prob:
            part = random.choice(injury_part)
            reason = random.choice(injury_reason)
            degree = random.choice(injury_degree)
            status = random.choice(injury_status)
            self.injury_list.append([part, reason, degree, status])

    def construct_prompt(self, command, surrounding, previous_log=''):
        self.injury_generator()
        injury_situation = "This is your injury situation:\n" + '\n'.join([f"Injury: {injury[0]}\nReason: {injury[1]}\nDegree: {injury[2]}\nStatus: {injury[3]}\n" for injury in self.injury_list]) if self.injury_list else "You are not injured."

        system_info = "Suppose you are a soldier in the medieval time in a battle. You are summoned for this battle from your normal life and you must obey whatever the command tells you to do. You will be given a profile before you are summoned as a soldier for this battle. You will write down your thoughts and feelings during this battle in a journal."
        personal_definition = "This is your bio:\n" + f"Name: {self.name}\nAge: {self.age}\nFamily: {self.family}\nOccupation: {self.occupation}\nPersonality: {self.personality}\nSocial Status: {self.social_status}\nPotential Illness: {self.potential_illness}\nBody Condition: {self.body_condition}\nHobbies and Interests: {self.hobbies_and_interests}\nStyle of Talking: {self.style_of_talking}\nUnique Quirks: {self.unique_quirks}\nSecrets or Scandals: {self.secrets_or_scandals}\n"

        command = f"Now you are given the following command: {command}\nThis is the surrounding of you: {surrounding}"
        jounal_command = "You will write down your thoughts and feelings after being given this command and working based on the command in a journal. Associate your thoughts,feelings with the bio you are given."
        return system_info + "\n" + personal_definition + "\n" + injury_situation +  "\n" + command + '\n' + jounal_command

    def generate_journal(self, current_prompt):
        journal = self.run_model(current_prompt)
        return journal

    @staticmethod
    def generate_journal_batch(soldier_profiles, prompts, accumulator=None, max_concurrency=8):
        """Batch journal generation for many soldiers in one concurrent fan-out.

        ``soldier_profiles`` and ``prompts`` are parallel lists; returns journals in order.
        Token usage is bucketed under the "diary" role on ``accumulator``. All soldiers share
        the same ``model_type`` (diary model), so a single batch call suffices."""
        if not prompts:
            return []
        model_type = soldier_profiles[0].model_type
        return run_LLM_batch(model_type, prompts, accumulator, "diary", max_concurrency)

    def collect_journal(self, time, journal):
        self.journal[time] = journal

    def summarize_journal(self):
        whole_journal = '\n\n'.join([time + time_journal for time, time_journal in self.journal.items()])
        prompt = 'Summarize this journal in a paragraph:\n' + whole_journal + '\nSummarization:'
        summary = self.run_model(prompt)
        prompt = "Infer the current mental state and physical state based on the journal:\n" + whole_journal + '\n\nMental State:\nPhysical State:'
        states = self.run_model(prompt)
        return summary, states
    
    def run_model(self, prompt):
        return run_LLM(self.model_type, prompt, self.token_accumulator, "diary")
    
    

    
class Soldier_Agent():
    def __init__(self,profile, hierarchy):
        self.profile = profile
        self.hierarchy = hierarchy
        
    def prepare(self, executed_troop, command, surrounding):
        """Apply structure-change/transfer logic (mutates hierarchy, must stay sequential) and
        return the constructed journal prompt. Split out from ``execute`` so callers can batch
        the LLM journal generation across many soldiers."""
        # detect and obtain structural changes
        new_detached_troop_list = self.hierarchy.monitor_structure_change(executed_troop)

        # if structural changes are detected (i.e., the list is not empty), calculate transfer probabilities
        if new_detached_troop_list:
            # use calculate_transfer_probability_and_decide to decide whether to transfer
            decision = self.hierarchy.calculate_transfer_probability_and_decide(self.hierarchy.current_troop, new_detached_troop_list)

            # if the decision is to transfer (decision is not the current troop), update the current troop
            if decision and decision != self.hierarchy.current_troop:
                # update the soldier's current troop
                self.hierarchy.current_troop = decision
                # also update the hierarchy structure
                self.hierarchy.sub_troop = decision.hierarchy.sub_agents

        return self.profile.construct_prompt(command, surrounding)

    def collect(self, time, journal_entity):
        self.profile.collect_journal(time, journal_entity)

    def execute(self, executed_troop, command, surrounding, time):
        # build prompt, generate and collect journal
        current_prompt = self.prepare(executed_troop, command, surrounding)
        journal_entity = self.profile.generate_journal(current_prompt)
        self.collect(time, journal_entity)

        # more logic can be added here, such as making further decisions based on journal results
        
        

# Soldier personas live as JSON under src/data/soldier_profiles/<country>/*.json
# (externalized from ~450 lines of dict literals in Phase 5).
_PROFILE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "soldier_profiles",
)


def load_soldier_profiles(data_dir=None):
    """Load soldier personas from JSON, grouped by country.

    Returns ``{"country_E": [Soldier_Profile, ...], "country_F": [...]}`` in the deterministic
    (zero-padded filename) order the files are stored on disk."""
    import glob

    data_dir = data_dir or _PROFILE_DATA_DIR
    profiles = {}
    for country in ("country_E", "country_F"):
        country_dir = os.path.join(data_dir, country)
        files = sorted(glob.glob(os.path.join(country_dir, "*.json")))
        country_profiles = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                country_profiles.append(Soldier_Profile(json.load(f)))
        profiles[country] = country_profiles
    return profiles


# create instances of soldier profiles and store them in a dictionary, grouped by country
Soldier_Profiles = load_soldier_profiles()


class SoldierCollector:
    def __init__(self, soldier_profiles_for_nationality, initial_root_agent, model_type=None):
        # Now expects profiles for a single nationality
        self.soldier_profiles = soldier_profiles_for_nationality
        self.model_type = model_type
        self.token_accumulator = None

        self.initial_root_agent = initial_root_agent
        self.soldier_agents_list = self._initialize_soldier_agents()

    def _initialize_soldier_agents(self):
        """Initialize Soldier Agents based on the soldier profiles provided."""
        soldier_agents = []
        for profile in self.soldier_profiles:
            profile.model_type = self.model_type
            profile.token_accumulator = self.token_accumulator
            hierarchy = Soldier_Hierarchy(self.initial_root_agent)  # Assuming this is correctly implemented
            soldier_agent = Soldier_Agent(profile, hierarchy)
            soldier_agents.append(soldier_agent)
        return soldier_agents
    
    def get_soldiers(self, obj):
        """Return a list of all soldier agents matching the given object's hierarchy ID. Return an empty list if no matching agents are found."""
        matching_soldiers = []  # initialize an empty list to store matching soldier agents
        for soldier in self.soldier_agents_list:
            current_troop = soldier.hierarchy.current_troop
            if current_troop.hierarchy.id == obj.hierarchy.id:
                matching_soldiers.append(soldier)  # add the matching soldier agent to the list
        return matching_soldiers  # return the list containing all matching soldier agents
    
if __name__ == '__main__':

    # Create SoldierAgency instances for each nationality
    country_E_collector = SoldierCollector(Soldier_Profiles["country_E"], initial_root_agent=None)  # Adjust the root agent as necessary
    country_F_collector = SoldierCollector(Soldier_Profiles["country_F"], initial_root_agent=None)  # Adjust the root agent as necessary
