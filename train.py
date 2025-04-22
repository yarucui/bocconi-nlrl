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
    parser.add_argument('initial_board')
    parser.add_argument('model_config')
    parser.add_argument('agent_config')
    args = parser.parse_args()
    #
    # Initialize the environment
    #
    env = MazeEnv(args.initial_board)
    #
    # Initialize the model
    #
    with open(args.model_config, 'r') as file:
        model_config = json.load(file)
    if model_config['type'] == 'Mistral':
        llm = Mistral(args.model_config)
    else:
        raise ValueError(f"Unrecognized model type: {model_config['type']}")
    #
    # Initialize the agent
    #
    agent = ActorCriticAgent(env, llm, args.agent_config)
    #
    # Run training loop
    #
    agent.train(T=1, N=1, K=1)
