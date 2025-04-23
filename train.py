# External imports
import argparse
import json

# Internal imports
from agents.actorcritic import ActorCriticAgent
from envs.maze.maze import MazeEnv
from models.mistral import Mistral

if __name__ == "__main__":
    #
    # Parse CLI arguements
    #
    parser = argparse.ArgumentParser(prog='ActorCriticTraining')
    parser.add_argument('train_config')
    args = parser.parse_args()
    #
    # Open the train configuration file
    #
    with open(args.train_config, 'r') as file:
        train_config = json.load(file)
    #
    # Initialize the environment
    #
    env = MazeEnv(train_config['initial_board'])
    #
    # Open the model configuration file and initialize the model object.
    #
    with open(train_config['model_config'], 'r') as file:
        model_config = json.load(file)
    if model_config['type'] == 'Mistral':
        llm = Mistral(train_config['model_config'])
    else:
        raise ValueError(f"Unrecognized model type: {model_config['type']}")
    #
    # Initialize the agent
    #
    agent = ActorCriticAgent(env, llm, train_config['agent_config'])
    #
    # Run training loop
    #
    agent.train(T=1, N=1, K=1)
