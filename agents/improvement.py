# External imports
import json
from pathlib import Path
import re

# Internal import
from models.model import LanguageModel

class ImprovementOperator:
    def __init__(self, llm : LanguageModel, config : str):
        #
        # LLM to query
        #
        self.llm = llm
        #
        # Get the improvement operator's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt']).read_text(encoding='utf-8')
        self.improvement_prompt = Path(self.config['improvement_prompt']).read_text(encoding='utf-8')
        self.evaluation_description = Path(self.config['describe_evaluation']).read_text(encoding='utf-8')

    #
    # Given the state-action pairs for a state and descriptions of their values from the language
    # value function, perform chain-of-thought reasoning to determine the best action.
    #
    # Return the strategic reasoning for what's the best action and why.
    #
    def reason(self, state, actions, values) -> str:
        #
        # Format the state-action pair evaluations into a string that can be plugged into
        # the user prompt.
        #
        evals_text = self.evaluations_to_text(actions, values)
        #
        # Query the LLM with the state-action pair evaluations
        #
        response = self.llm.generate_response(self.system_prompt,
                                              self.improvement_prompt.format(state=state,
                                                                             actions=actions,
                                                                             evaluations=evals_text))
        #
        # Log
        #
        """
        print('-------------------')
        print('--> LLM Policy Improvement Operator')
        print()
        print('Input state:')
        print(state)
        print()
        print('Input action evaluations:')
        print(evals_text)
        print()
        #
        # Verify the response's formatting by extracting the best action and reasoning.
        #
        action = self.extract_action_from_response(response, actions)
        reason = self.extract_reason_from_response(response)
        #
        # Log
        #
        print('Selected action:', actions[action])
        print()
        print('Reason:')
        print(reason)
        print()
        import ipdb; ipdb.set_trace()
        """
        #
        # Return the response
        #
        return response

    #
    # Format the given state-action pair evaluations into a single string.
    #
    def evaluations_to_text(self, actions: dict[int, str], values: list[str]) -> str:
        #
        # Use the evaluations description to format each 
        # action evaluation into text.
        #
        evaluations_text = ""
        for i, (action_id, action_str) in enumerate(actions.items()):
            #
            # Add this evaluation to the text.
            #
            evaluations_text += self.evaluation_description.format(action_id=action_id,
                                                                   action_str=action_str,
                                                                   evaluation=values[i])
            evaluations_text += '\n'
        #
        # Return the final text containing all the evaluations
        #
        return evaluations_text
    
    #
    # Extract the selected action from the LLM response text.
    #
    # Raise an error if the action isn't found or if the extracted
    # action is not in the given actions dictionary.
    #
    def extract_action_from_response(self, response: str, actions: dict[int, str]) -> int:
        #
        # Response must contain this pattern
        #
        action_match = re.search(r'Best action:\s*(\d+)', response)
        if action_match:
            #
            # Success case - match found.
            #
            action = int(action_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error.
            #
            raise ValueError(f"Missing action. Improvement Operator LLM returned an ill-formatted response.\n Response:\n'{response}'")
        #
        # Check that the found action id is valid.
        #
        if action not in actions.keys():
            raise ValueError(f"Improvement Operator LLM selected an invalid action. Got {action}. Expected one of these {actions}\n Response: {response}")
        #
        # Return the action id.
        #
        return action

    #
    # Extract the reasoning from the LLM response text.
    #
    # Raise an error if the reasoning isn't found.
    #
    def extract_reason_from_response(self, response: str) -> int:
        #
        # Response must contain this pattern.
        #
        reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
        if reason_match:
            #
            # Success case - match found, extract the reasoning string.
            #
            reason =  str(reason_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error.
            #
            raise ValueError(f"Missing reasoning. Policy LLM return an ill-formatted response. Response:\n'{response}'")
        #
        # Return the reasoning string
        #
        return reason