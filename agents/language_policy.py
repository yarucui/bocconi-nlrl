


class LanguagePolicy:
    def __init__(self):
        pass

    #
    # Given an state, select an action.
    #
    def get_action(self, state):
        pass

    #
    # Given a batch of policy targets, update the policy.
    #
    def update(self, policy_targets):
        pass
