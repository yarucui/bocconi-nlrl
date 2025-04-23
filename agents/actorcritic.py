# External imports
from collections import Counter
import json
import numpy as  np
from pathlib import Path

# Internal imports
from envs.environment import Environment
from agents.improvement import ImprovementOperator
from agents.language_policy import LanguagePolicy
from agents.language_value_function import LanguageValueFunction
from models.model import LanguageModel

class ActorCriticAgent:

    def __init__(self, env : Environment, llm : LanguageModel, agent_config_file : str):
        #
        # Environment
        #
        self.env = env
        #
        # Read agent configuration file
        #
        with open(agent_config_file, 'r') as file:
            self.config = json.load(file)
        #
        # Text describing the task to be accomplished.
        #
        self.task_instruction = Path(self.config['task_instruction_file']).read_text(encoding='utf-8')
        #
        # Core components
        #
        self.lang_policy = LanguagePolicy(llm, self.config['policy_config'])
        self.lang_values = LanguageValueFunction(llm, self.config['values_config'])
        self.improvement_op = ImprovementOperator(llm, self.config['improvement_config'])
    
    #
    # Main actor-critic training loop
    #
    def train(self, T,  # Number of training iterations 
                    N,  # Number of trajectories per iteration
                    K): # Number of Monte-Carlo trajectories
        #
        # Training hyperparameters
        #
        #   - N_ESTIMATE_SAMPLES - num. of sampled trajectories to estimate the state-action value
        #   - N_ACTION_SAMPLES - num. of actions to sample to estimate policy action probability
        #   - TOP_N_ACTIONS - select top N most probable actions to perform the policy update.
        #   - VALUE_BATCH_SIZE - num. of value targets to use per training iteration
        #   - POLICY_BATCH_SIZE - num. of policy targets to use per training iteration
        #   - KEEP_N_ITER_HISTORY - num. of training iterations until a target is evicted from its buffer.
        #
        N_ESTIMATE_SAMPLES = 1
        N_ACTION_SAMPLES = 4
        TOP_N_ACTIONS = 4
        VALUE_BATCH_SIZE = 'all'
        POLICY_BATCH_SIZE = 'all'
        KEEP_N_ITER_HISTORY = 1
        #
        # Store value targets and policy targets
        #
        value_buffer = []  # [(train_idx, (s, a, v), ...]
        policy_buffer = [] # [(train_idx, (s, policy target, strategic reasoning), ...]
        #
        # Main training loop
        #
        for train_idx in range(T): # Main training loop
            #
            # Collect trajectories
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 1: COLLECT TRAJECTORIES')
            trajectories = [] # [[(s, a, r), ..], ...]
            for _ in range(N):
                #
                # Initialize the environment to its starting state
                #
                self.env.reset()
                #
                # Rollout the state to completion
                #
                trajectories.append(self.rollout())
            #
            # Build value estimation targets
            #
            #    Compute value estimates for each state-action
            #    pair that was observed during rollouts.
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 2: COMPUTE VALUE TARGETS')
            value_targets = [] # [(s, a, v), ...]
            for trajectory in trajectories:
                for transition in trajectory:
                    state, action = transition[0], transition[1]
                    #
                    # For Monte-Carlo estimates, we evaluate the state-action
                    # pair using a set of sample trajectories.
                    #
                    sample_trajectories = []
                    for _ in range(N_ESTIMATE_SAMPLES):
                        #
                        # Set the environment to the given state
                        #
                        self.env.set_state(state)
                        #
                        # Start by applying the action in the state
                        #
                        _, reward = self.env.act(action)
                        #
                        # Combine this with the rest of the rollout
                        # to sample the rest of the trajectory.
                        #
                        sample_trajectories.append([(state, action, reward)] + self.rollout())
                    #
                    # Given the example trajectories, evaluate how good or bad
                    # taking the action is in this state.
                    #
                    value = self.lang_values.mc_estimate(state, 
                                                         action, 
                                                         self.env.actions(), 
                                                         sample_trajectories)
                    #
                    # Save the result.
                    #
                    value_targets.append((train_idx, (state, action, value)))
            #
            # Save the value targets from this training iteration to the value buffer.
            #
            value_buffer += value_targets
            #
            # Update the value function using the value targets
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 3: TRAIN VALUE MODEL')
            if VALUE_BATCH_SIZE == 'all':
                value_targets_batch = [value_buffer[idx][1] for idx in range(len(value_buffer))]
            else:
                sample_idxs = np.random.choice(range(len(value_buffer)), size=VALUE_BATCH_SIZE, replace=False)
                value_targets_batch = [value_buffer[idx][1] for idx in sample_idxs]
            import ipdb; ipdb.set_trace()
            self.lang_values.update(value_targets_batch, self.env.actions())
            import ipdb; ipdb.set_trace()
            #
            # Use the updated value function to improve the policy
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 4: COMPUTE POLICY TARGETS')
            for trajectory in trajectories:
                for transition in trajectory:
                    state = transition[0]
                    #
                    # Sample actions from the policy to estimate
                    # action probabilities.
                    #
                    # Recall - the language policy does not directly assign a probability 
                    #          distribution over the action space.
                    #
                    sampled_actions = [self.lang_policy.get_action(state)[0] for _ in range(N_ACTION_SAMPLES)]
                    #
                    # Get the top N most frequent actions
                    #
                    actions = [action for action, _ in Counter(sampled_actions).most_common(TOP_N_ACTIONS)]
                    #
                    # Get the value estimates for each action in this state
                    #
                    values = [self.lang_values.get_value(state, action) for action in actions]
                    #
                    # Query the language improvement operator to get strategic reasoning text and
                    # a policy target.
                    #
                    strategic_reasoning, policy_target = self.improvement_op.reason(values, self.task_instruction)
                    #
                    # Store the policy target triplet in the policy buffer.
                    #
                    policy_buffer.append((train_idx, (state, policy_target, strategic_reasoning)))
            #
            # Update the policy using the policy targets
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 5: TRAIN POLICY MODEL')
            sample_idxs = np.random.choice(range(len(policy_buffer)), size=POLICY_BATCH_SIZE, replace=False)
            policy_targets_batch = [policy_buffer[idx][1] for idx in sample_idxs]
            self.lang_policy.update(policy_targets_batch)
            #
            # Evict old targets
            #
            threshold = train_idx - KEEP_N_ITER_HISTORY
            value_buffer = [target for target in value_buffer if target[0] >= threshold]
            policy_buffer = [target for target in policy_buffer if target[0] >= threshold]

    #
    # Use the agent's policy to rollout the environment
    # state to completion. Return the observed trajectory.
    #
    def rollout(self, max_trajectory_length=5):
        #
        # Store the observed transitions
        #
        trajectory = []
        #
        # Continue to act until the game terminates.
        # Or we hit the maximum allowed trajectory length
        #
        while not self.env.is_terminal() and len(trajectory) < max_trajectory_length:
            #
            # Get the current state
            #
            current_state = self.env.state
            #
            # Get the set of actions available in the current state
            #
            actions = self.env.actions()
            #
            # Get an action from our policy given the current state
            #
            action = self.lang_policy.get_action(current_state, actions)[0]
            #
            # Apply the action to the environment to collect a
            # reward and the next state.
            #
            _, reward = self.env.act(action)
            #
            # Store the transition
            #
            trajectory.append((current_state, action, reward))
        #
        # Return the observed trajectory
        #
        return trajectory