

class LanguageValueFunction:
    def __init__(self):
        pass

    #
    # Given a state-action pair, compute the
    # Monte-Carlo value estimation.
    #
    # K - number of transition samples to form the estimate.
    #
    def mc_estimate(self, state, action, K=1):
        pass

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