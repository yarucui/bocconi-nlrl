# External imports
from collections import Counter
import numpy as  np

# Internal imports
from language_policy import LanguagePolicy
from language_value_function import LanguageValueFunction
from improvement import ImprovementOperator
from envs.environment import Environment


class ActorCriticAgent:

    def __init__(self, env : Environment, task_instruction : str):
        #
        # Environment
        #
        self.env = env
        #
        # Text describing the task to be accomplished.
        #
        self.task_instruction = task_instruction
        #
        # Core components
        #
        self.lang_policy = LanguagePolicy()
        self.lang_values = LanguageValueFunction()
        self.improvement_op = ImprovementOperator()
    
    #
    # Main actor-critic training loop
    #
    def train(self, T,  # Number of training iterations 
                    N,  # Number of trajectories per iteration
                    K): # Number of Monte-Carlo trajectories
        #
        # Training hyperparameters
        #
        #   - N_ESTIMATE_SAMPLES - num. of sampled transitions to estimate state-action value
        #   - N_ACTION_SAMPLES - num. of actions to sample to estimate policy action probability
        #   - TOP_N_ACTIONS - select top N most probable actions to perform the policy update.
        #   - VALUE_BATCH_SIZE - num. of value targets to use per training iteration
        #   - POLICY_BATCH_SIZE - num. of policy targets to use per training iteration
        #   - KEEP_N_ITER_HISTORY - num. of training iterations until a target is evicted from its buffer.
        #
        N_ESTIMATE_SAMPLES = 5
        N_ACTION_SAMPLES = 4
        TOP_N_ACTIONS = 4
        VALUE_BATCH_SIZE = 1
        POLICY_BATCH_SIZE = 1
        KEEP_N_ITER_HISTORY = 1
        #
        # Store trajectories
        #
        value_buffer = [] # [(train_idx, (s, a, v), ...]
        #
        # Main training loop
        #
        for train_idx in range(T): # Main training loop
            #
            # Collect trajectories
            #
            trajectories = [] # [[(s, a, r, s'), ..], ...]
            for traj_iter in range(N):
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
            value_targets = [] # [(s, a, v), ...]
            for trajectory in trajectories:
                for transition in trajectory:
                    state, action = transition[0], transition[1]
                    value = self.lang_values.mc_estimate(state, action, N_ESTIMATE_SAMPLES)
                    value_targets.append((train_idx, (state, action, value)))
            value_buffer.append(value_targets)
            #
            # Update the value function using the value targets
            #
            sample_idxs = np.random.choice(range(len(value_buffer)), size=VALUE_BATCH_SIZE, replace=False)
            value_targets_batch = [value_buffer[idx][1] for idx in sample_idxs]
            self.lang_values.update(value_targets_batch)
            #
            # Use the updated value function to improve the policy
            #
            policy_buffer = []
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
                    sampled_actions = [self.lang_policy.get_action(state) for _ in range(N_ACTION_SAMPLES)]
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
    def rollout(self):
        #
        # Store the observed transitions
        #
        trajectory = []
        #
        # Continue to act until the game terminates.
        #
        while not self.env.is_terminal():
            #
            # Get the current state
            #
            current_state = self.env
            #
            # Get an action from our policy given the current state
            #
            action = self.lang_policy.get_action(current_state)
            #
            # Apply the action to the environment to collect a
            # reward and the next state.
            #
            next_state, reward = self.env.act(action)
            #
            # Store the transition
            #
            trajectory.append((current_state, action, reward, next_state))
        #
        # Return the observed trajectory
        #
        return trajectory