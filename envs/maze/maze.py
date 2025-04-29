# External imports
from pathlib import Path

# Internal imports
from envs.environment import Environment

class MazeEnv(Environment):
    #
    # board_file - text file containing the starting state for the maze
    #
    # Example:
    #
    #    #####
    #    #o  #
    #    #   #
    #    #  x#
    #    #####
    #
    #     - The player 'o' starts in square (1, 1).
    #
    #     - The goal 'x' is in the square (3, 3).
    #
    #     - The walls '#' define the boundaries.
    #
    def __init__(self, board_file : str):
        #
        # Save the board file location
        #
        self.board_file = board_file
        #
        # Initialize the game as finished.
        #
        self.terminated = True
        #
        # Set of actions
        #
        self.action_set = {0: 'move up', 1: 'move down', 2: 'move left', 3: 'move right'}
        #
        # Movements associated with each action
        #
        self.movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        #
        # Set the characters that represent the player, goal, and walls.
        #
        self.player_char = 'o'
        self.goal_char = 'x'
        self.wall_char = '#'
        self.space_char = ' '
        #
        # Initialize the player and goal positions
        #
        self.player_position = None
        self.goal_position = None
        #
        # Initialize the environment.
        #
        self.reset()
    
    #
    # Reset the environment to its initial state
    #
    def reset(self):
        #
        # Read the initial game state from the board file.
        #
        starting_state = Path(self.board_file).read_text(encoding='utf-8')
        #
        # Set the environment to its starting state
        #
        self.set_state(starting_state)

    #
    # Given a state, set the environment to that state.
    #
    def set_state(self, state : str) -> None:
        #
        # Set the environment state
        #
        self.state = state
        #
        # Build the grid
        #
        self.state_to_grid()
        #
        # Reset the player and goal positions
        #
        self.player_position = None
        self.goal_position = None
        #
        # Verify the grid is well-formatted and extract the player and
        # goal positions.
        #
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                #
                # Check for the player
                #
                if self.grid[i][j] == self.player_char:
                    assert self.player_position is None, "More than one player found in the input board."
                    self.player_position = (i, j)
                #
                # Check for the goal
                #
                elif self.grid[i][j] == self.goal_char:
                    assert self.goal_position is None, "More than one goal found in the input board."
                    self.goal_position = (i, j)
                #
                # Check for invalid chars.
                #
                elif self.grid[i][j] != self.space_char and self.grid[i][j] != self.wall_char:
                    raise ValueError(f"Invalid character in input board: {self.grid[i][j]}")
        #
        # Check that a player and goal position were found
        #
        assert self.player_position is not None, "Player not found in input board."
        assert self.goal_char is not None, "Goal not found in input board."
        #
        # Maze is not over since we are reseting to a state.
        #
        self.terminated = False

    #
    # Return a list of all valid actions in the environment
    #
    def actions(self) -> list:
        return self.action_set
    
    #
    # Return - True if the game has terminated.
    #        - False otherwise.
    #
    def is_terminal(self):
        return self.terminated
    
    #
    # Apply the given action in the environment,
    # return the resulting next state and the reward gained.
    #
    # Note: If the given action moves the player into a wall or out-of-bounds, then
    #       it is treated as no action and the player stays in place.
    #
    def act(self, action_id : int) -> tuple[str, int]:
        #
        # Check that the game is not over.
        #
        assert not self.terminated, "Trying to act in a terminated game."
        #
        # Check that the action id is valid
        #
        assert action_id in self.action_set.keys(), f"Unrecognized action id: {action_id}"
        #
        # Get the position the player is trying to move into.
        #
        new_player_position = (self.player_position[0]+self.movements[action_id][0],
                               self.player_position[1]+self.movements[action_id][1])
        #
        # Check that the new position is valid.
        #
        if (
            (new_player_position[0] < 0 or new_player_position[0] >= len(self.grid)) or
            (new_player_position[1] < 0 or new_player_position[1] >= len(self.grid[new_player_position[0]])) or
            (self.grid[new_player_position[0]][new_player_position[1]] == self.wall_char)
        ):
            new_player_position = self.player_position # Default to a null action
        #
        # Update the grid
        #
        self.grid[new_player_position[0]][new_player_position[1]] = self.player_char
        if new_player_position != self.player_position:
            self.grid[self.player_position[0]][self.player_position[1]] = self.space_char
        self.player_position = new_player_position
        #
        # Update the state
        #
        self.grid_to_state()
        #
        # Check if the player reached the goal
        #
        reward = int(self.player_position == self.goal_position)
        if reward == 1:
            print('REWARD REACHED')
            self.terminated = True
        #
        # Return the new state and the observed reward
        #
        return self.state, reward

    #
    # Use the current state representation to form the grid. str -> list[list[str]].
    #
    def state_to_grid(self):
        #
        # Build a grid using a 2d list.
        #
        # Example:
        #
        #   self.grid = [['#', '#', '#', '#', '#'],
        #                ['#', 'o', ' ', ' ', '#'],
        #                ['#', ' ', ' ', ' ', '#'],
        #                ['#', ' ', ' ', 'x', '#'],
        #                ['#', '#', '#', '#', '#']]
        #
        self.grid = [list(line) for line in self.state.split('\n')]

    #
    # Use the current grid to form the state. list[list[str]] -> str.
    #
    def grid_to_state(self):
        self.state = ''.join([''.join(line)+'\n' for line in self.grid])
        self.state = self.state[:-1]


