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
    # return the Monte-Carlo value estimation.
    #
    def mc_estimate(self, state, action, actions, trajectory_samples):
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
        # Extract the action from the LLM response
        #
        value_match = re.search(r'Value:\s*([-+]?\d+(?:\.\d+)?)', response)
        if value_match:
            value = float(value_match.group(1))
        else:
            raise ValueError(f"Missing value. MC Estimate LLM returned an ill-formatted response. Response:\n'{response}'")
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
        # Log
        #
        print('Value:', value)
        print()
        print('Reason:')
        print(reason)
        #
        # Otherwise, the selected action is valid.
        #
        return value, reason

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
    # Given a batch of target values, update the value function.
    #
    def update(self, target_values : list[tuple]) -> None:
        pass

    #
    # Given a state-action pair, return the value from the value function
    #
    def get_value(self, state, action):
        pass