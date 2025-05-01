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
        # Get the language policy's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt_file']).read_text(encoding='utf-8')
        self.user_prompt = Path(self.config['user_prompt_file']).read_text(encoding='utf-8')    

    #
    # Given a state, return an action from a set of possible actions and
    # a string explaining the reasoning.
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
    #     Example output - Maze
    #
    #         action = 3
    #
    #         reason = """
    #           The player is currently standing next to a free space on the right side of 
    #           the maze. The goal square is also located on the right side of the maze. 
    #           Therefore, the best action for the player is to move right, as it will 
    #           bring them closer to the goal square and allow them to explore more of 
    #           the maze.
    #         """
    #
    def get_action(self, state : str, actions : dict) -> tuple[int, str]:
        #
        # Query the LLM with the given state and actions
        #
        response = self.llm.generate_response(self.system_prompt, 
                                              self.user_prompt.format(state=state, 
                                                                      actions=actions))
        #
        # Log
        #
        print('-------------------')
        print('--> LLM Policy')
        print()
        print('Input state:')
        print(state)
        print()
        #
        # Extract the action from the LLM response
        #
        action_match = re.search(r'Best action:\s*(\d+)', response)
        if action_match:
            action = int(action_match.group(1))
        else:
            raise ValueError(f"Missing action. Policy LLM returned an ill-formatted response. Response:\n'{response}'")
        #
        # Extract the reasoning
        #
        # The reasoning should always follow the reason identifier, "Reason: "
        #
        reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
        if reason_match:
            reason =  str(reason_match.group(1))
        else:
            raise ValueError(f"Missing reasoning. Policy LLM return an ill-formatted response. Response:\n'{response}'")
        #
        # Check that the selected action is in the given set of possible actions.
        #
        assert action in actions.keys(), f"Policy LLM selected an invalid action. Got {action}. Expected one of these {actions}"
        #
        # Log
        #
        print('Action:', actions[action])
        print()
        print('Reason:')
        print(reason)
        print()
        #
        # Otherwise, the selected action is valid.
        #
        return action, reason

    #
    # Given a batch of policy targets, update the policy LLM
    #
    # policy_targets = [(state, available actions, policy_target),
    #                    ...]
    #
    def update(self, policy_targets: list[tuple[str, str]]) -> None:
        #
        # Format the targets into a list that can be used to create
        # a Hugging Face dataset object.
        #
        # data = [
        #          {'system_prompt': ..., 'user_prompt': ..., 'response': ...},
        #           ...
        #        ]
        #
        data = [
            {
                'system_prompt': self.system_prompt,
                'user_prompt': self.user_prompt.format(state=state, 
                                                       actions=actions),
                'response': policy_target
            } for state, actions, policy_target in policy_targets
        ]
        #
        # Train the LLM on the policy target data
        #
        self.llm.train(data)