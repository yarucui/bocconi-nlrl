# External imports
import json
from pathlib import Path
import re

# Internal imports
from models.model import LanguageModel

class LanguagePolicy:

    def __init__(self, llm : LanguageModel, config : str):
        #
        # LLM to query for actions
        #
        self.llm = llm
        #
        #
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt_file']).read_text(encoding='utf-8')
        self.user_prompt = Path(self.config['user_prompt_file']).read_text(encoding='utf-8')    

    #
    # Given a state, select an action from a set of possible actions.
    #
    #    Example input - Maze
    #
    #       state = """ #####
    #                   #o  #
    #                   #   #
    #                   #  x#
    #                   #####
    #               """
    #
    #       actions = {0: "Move up", 1: "Move down", 2: "Move right", 3: "Move left"}
    #
    def get_action(self, state : str, actions : dict) -> int:
        #
        # Query the LLM with the given state and actions
        #
        response = self.llm.generate_response(self.system_prompt, 
                                              self.user_prompt.format(state=state, actions=actions))
        #
        # Extract the action from the LLM response
        #
        match = re.search(r'Best action:\s*\d+', response)
        if match:
            action = int(match.group(1))
        else:
            raise ValueError(f"Policy LLM returned an ill-formatted response. Response:\n{response}")
        #
        # Check that the selected action is in the given set of possible actions.
        #
        assert action in actions.keys(), f"Policy LLM selected an invalid action. Got {action}. Expected one of these {actions}"
        #
        # Otherwise, the selected action is valid.
        #
        return action

    #
    # Given a batch of policy targets, update the policy.
    #
    def update(self, policy_targets):
        pass
