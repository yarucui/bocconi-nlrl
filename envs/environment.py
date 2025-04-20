

class Environment:
    def __init__(self):
        self.state = None

    #
    # Return a list of all valid actions in the environment
    #
    def actions(self) -> list:
        pass
    
    #
    # Return - True if the game has terminated.
    #        - False otherwise.
    #
    def is_terminal(self):
        pass

    #
    # Apply the given action in the environment,
    # return the resulting next state and the reward gained.
    #
    def act(action):
        pass