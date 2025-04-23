# External imports
import json
from pathlib import Path
import re

# Internal imports
from models.model import LanguageModel

class LanguageValueFunction:
    def __init__(self, llm : LanguageModel, config : str):
        #
        # LLM to query for values and MC estimates
        #
        self.llm = llm
        #
        # Get the language value function's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt']).read_text(encoding='utf-8')
        self.value_prompt = Path(self.config['value_prompt']).read_text(encoding='utf-8')
        self.mc_estimate_prompt = Path(self.config['mc_estimate_prompt']).read_text(encoding='utf-8')
        self.transition_description = Path(self.config['describe_transition']).read_text(encoding='utf-8')


    #
    # Given a state-action pair and a set of example trajectories, 
    # return the LLM's response estimating the Monte-Carlo value.
    #
    def mc_estimate(self, state : str, 
                          action : int, 
                          actions : dict[int, str], 
                          trajectory_samples : list[tuple[str, int, float]]) -> str:
        #
        # Describe the given sampled trajectories in text
        #
        traj_text = self.trajectories_to_text(actions, trajectory_samples)
        #
        # Query the LLM with the given state-action pair and the example trajectories
        #
        response = self.llm.generate_response(self.system_prompt.format(actions=actions.values()),
                                              self.mc_estimate_prompt.format(state=state, 
                                                                             action=actions[action], 
                                                                             examples=traj_text))
        #
        # Log
        #
        print('-------------------')
        print('--> LLM MC Estimate')
        print()
        print('Input state-action pair:')
        print(state)
        print()
        print('Action:', actions[action])
        print()
        #
        # Verify the response's formatting by extracting the value
        # and reasoning.
        #
        value = self.extract_value_from_response(response)
        reason = self.extract_reason_from_response(response)
        #
        # Log
        #
        print('Value:', value)
        print()
        print('Reason:')
        print(reason)
        #
        # Otherwise, the selected action is valid.
        #
        return response

    #
    # Given a list of trajectories, return a string describing each trajectory.
    #
    def trajectories_to_text(self, actions : dict, trajectories : list[tuple[str, int, int]]) -> str:
        #
        # Use the tranisition description to format
        # each trajectory sample into text.
        #
        traj_samples_text = ""
        for i, trajectory in enumerate(trajectories, 1):
            #
            # Trajectory header to delineate it from other samples.
            #
            traj_samples_text += f"Trajectory example {i}:\n"
            #
            # Add the description of each transition to the trajectory text.
            #
            for state, action, reward in trajectory:
                #
                # Describe this transition in text.
                #
                traj_samples_text += self.transition_description.format(state=state,
                                                                        action=actions[action],
                                                                        reward=reward)
                traj_samples_text += '\n'
        #
        # Return the final description.
        #
        return traj_samples_text
    
    #
    # Given a response from the LLM, extract the value.
    #
    def extract_value_from_response(self, response : str) -> float:
        value_match = re.search(r'Value:\s*([-+]?\d+(?:\.\d+)?)', response)
        if value_match:
            value = float(value_match.group(1))
        else:
            raise ValueError(f"Missing value. MC Estimate LLM returned an ill-formatted response. Response:\n'{response}'")
        return value

    #
    # Given a response form the LLM, extract the reasoning.
    #
    def extract_reason_from_response(self, response : str) -> str:
        reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
        if reason_match:
            reason =  str(reason_match.group(1))
        else:
            raise ValueError(f"Missing reasoning. MC Estimate LLM return an ill-formatted response. Response:\n'{response}'")
        return reason

    #
    # Given a batch of target values, update the value function.
    #
    #    target_values = [
    #        (state, action, value),
    #         ...
    #    ]
    #
    #    where:
    #       - state = string describing the state
    #       - action = action id from the environment
    #       - value = string describing the Monte-Carlo estimate of the state-action pair
    #
    def update(self, target_values : list[tuple], actions : dict[int, str]) -> None:
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
                'system_prompt': self.system_prompt.format(actions=actions.values()),
                'user_prompt': self.value_prompt.format(state=state, action=action),
                'response': value_target
            } for state, action, value_target in target_values
        ]
        #
        # Train the LLM on the data
        #
        self.llm.train(data)

    #
    # Given a state-action pair, return the value from the value function
    #
    def get_value(self, state, action):
        pass