# External imports
from collections import Counter
from copy import deepcopy
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
        # Core components
        #
        self.lang_policy = LanguagePolicy(llm, 
                                          self.config['policy_config'],
                                          self.config['throw_formatting_errors']
        )
        self.lang_values = LanguageValueFunction(llm, 
                                                 self.config['values_config'],
                                                 self.config['throw_formatting_errors']
        )
        self.improvement_op = ImprovementOperator(llm, 
                                                  self.config['improvement_config'],
                                                  self.config['throw_formatting_errors']
        )
    
    #
    # Main actor-critic training loop
    #
    def train(self, T,  # Number of training iterations 
                    N,  # Number of trajectories per iteration
                    K): # Number of Monte-Carlo trajectories
        #
        # Training hyperparameters
        #
        #   - N_ACTION_SAMPLES - num. of actions to sample to estimate policy action probability
        #
        #   - TOP_N_ACTIONS - select top N most probable actions to perform the policy update.
        #                   - 'all' selects all possible actions in the given state.
        #
        #   - VALUE_BATCH_SIZE - num. of value targets to use per training iteration
        #
        #   - POLICY_BATCH_SIZE - num. of policy targets to use per training iteration
        #
        #   - KEEP_N_ITER_HISTORY - num. of training iterations until a target is evicted from its buffer.
        #
        #
        N_ACTION_SAMPLES = 'all'
        TOP_N_ACTIONS = 4
        VALUE_BATCH_SIZE = 'all'
        POLICY_BATCH_SIZE = 'all'
        KEEP_N_ITER_HISTORY = 3
        #
        # Store value targets and policy targets
        #
        value_buffer = []  # [(train_idx, (s, a, v)), ...]
        policy_buffer = [] # [((train_idx, (s, policy target, strategic reasoning)), ...]
        #
        # Main training loop
        #
        for train_idx in range(T): # Main training loop
            #
            # Collect trajectories
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 1: COLLECT TRAJECTORIES')
            #
            # Reset game to its starting position
            #
            self.env.reset()
            #
            # Create N copies of the environment
            #
            envs = [deepcopy(self.env) for _ in range(N)]
            #
            # Rollout each enironment and collect the trajectories.
            #
            trajectories = self.rollout(envs) # [[(s, a, r), ..], ...]
            #
            # Log the average number of states per trajectory.
            #
            print(f'Avg. trajectory length:{np.mean([len(trajectory) for trajectory in trajectories])}')
            #
            # Build value estimation targets
            #
            #    Compute value estimates for each state-action
            #    pair that was observed during rollouts.
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 2: COMPUTE VALUE TARGETS')
            value_targets = [] # [(s, a, v), ...]
            #
            # For Monte-Carlo estimates, we evaluate the state-action
            # pair using a set of sample trajectories.
            #
            for trajectory in trajectories:
                #
                # Sample K sample trajectories per state-action pair in this trajectory.
                #
                sample_trajectories = [[]] * len(trajectory)
                for _ in range(K):
                    #
                    # Start with a fresh list of environments
                    #
                    self.env.reset()
                    envs = [deepcopy(self.env) for _ in range(len(trajectory))]
                    #
                    # Set each environment to a state visited in the trajectory
                    #
                    # Progress each environment's state using the associated action
                    # that was taken at that step.
                    #
                    # Record the reward recieved from this action.
                    #
                    sa_pairs, trajectory_seeds, action_sets = [], [], []
                    for step, env in enumerate(envs):
                        #
                        # Get the state and action that was taken at this step.
                        #
                        state, action, _ = trajectory[step]
                        #
                        # Save the state action pair for this transition.
                        #
                        sa_pairs.append((state, action))
                        #
                        # Set the state
                        #
                        env.set_state(state)
                        #
                        # Save the set of valid actions in this state
                        #
                        action_sets.append(env.actions())
                        #
                        # Take the associated action and get the reward
                        #
                        _, reward = env.act(action)
                        #
                        # Add this transition (state, action, reward)
                        # as the first transition for each sampled trajectory.
                        #
                        trajectory_seeds.append([(state, action, reward)])
                    #
                    # Rollout the environments
                    #
                    rollouts = self.rollout(envs)
                    #
                    # Add the first transition to the front of these rollouts to
                    # get the full sample trajectory.
                    #
                    # Then add the full sampled trajectory to the sample trajectory list.
                    #
                    for sa_idx in range(len(trajectory)):
                        sample_trajectories[sa_idx].append(trajectory_seeds[sa_idx] + rollouts[sa_idx])
                #
                # Given the example trajectories, evaluate how good or bad
                # taking each action is in the given states.
                #
                values = self.lang_values.mc_estimate(sa_pairs,
                                                      action_sets, 
                                                      sample_trajectories)
                #
                # Save the result.
                #
                for step in range(len(trajectory)):
                    value_targets.append(
                        # (train idx, (state, action, value))
                        (train_idx, (sa_pairs[step][0], sa_pairs[step][1], values[step]))
                    )
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
            self.lang_values.update(value_targets_batch, self.env.actions())
            #
            # Use the updated value function to improve the policy
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 4: COMPUTE POLICY TARGETS')
            for trajectory in trajectories:
                for transition in trajectory:
                    state = transition[0]
                    self.env.set_state(state)
                    all_actions = self.env.actions()
                    #
                    # Get a set of actions for the policy improvement operator to consider
                    # for this state.
                    #
                    if N_ACTION_SAMPLES == 'all':
                        #
                        # Consider all available actions
                        #
                        actions = all_actions.keys()
                    else:
                        #
                        # Sample actions from the policy to estimate
                        # action probabilities.
                        #
                        # Recall - the language policy does not directly assign a probability 
                        #          distribution over the action space.
                        #
                        sampled_actions = [self.lang_policy.get_action(state, all_actions)[0] for _ in range(N_ACTION_SAMPLES)]
                        #
                        # Get the top N most frequent actions
                        #
                        actions = [action for action, _ in Counter(sampled_actions).most_common(TOP_N_ACTIONS)]
                    #
                    # Get the value estimates for each action in this state
                    #
                    states = [state] * len(actions)
                    action_sets = [self.env.actions()] * len(actions)
                    values = self.lang_values.get_value(states, actions, action_sets)
                    #
                    # Query the language improvement operator to get strategic reasoning text and
                    # a policy target.
                    #
                    policy_target = self.improvement_op.reason(state, actions, values)
                    #
                    # Store the policy target triplet in the policy buffer.
                    #
                    policy_buffer.append((train_idx, (state, all_actions, policy_target)))
            #
            # Update the policy using the policy targets
            #
            print('+++++++++++++++++++++++++++++')
            print('STEP 5: TRAIN POLICY MODEL')
            if POLICY_BATCH_SIZE == 'all':
                policy_targets_batch = [policy_buffer[idx][1] for idx in range(len(policy_buffer))]
            else:
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
    def rollout(self, envs: list[Environment], max_trajectory_length=5) -> list[tuple[str, int, str]]:
        #
        # Store the observed transitions
        #
        trajectories = [[] for _ in range(len(envs))]
        #
        # Number of steps taken thus far.
        #
        length = 0
        #
        # Continue to act until the game terminates.
        # Or we hit the maximum allowed trajectory length.
        #
        while not all(env.is_terminal() for env in envs) and length < max_trajectory_length:
            #
            # Get indicies of active environments
            #
            active_idxs = [idx for idx, env in enumerate(envs) if not env.is_terminal()]
            #
            # Get the set of actions available in the current states
            #
            action_sets = [envs[i].actions() for i in active_idxs]
            #
            # Select an action for each active environment
            #
            current_states = [envs[i].state for i in active_idxs]
            #
            # Get an action from our policy given the current state
            #
            actions, _ = self.lang_policy.get_action(current_states, action_sets)
            print(actions)
            print(active_idxs)
            #
            # Apply the actions to the environments to collect the
            # rewards and the next states.
            #
            rewards = [envs[env_idx].act(actions[idx])[1] 
                            for idx, env_idx in enumerate(active_idxs)] 
            #
            # Add each transition to its trajectory
            #
            for idx, env_idx in enumerate(active_idxs):
                trajectories[env_idx].append((current_states[idx], actions[idx], rewards[idx]))
            #
            # Increment the length counter
            #
            length += 1
        #
        # Return the observed trajectory
        #
        return trajectories