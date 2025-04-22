# Internal import
from models.model import LanguageModel

class ImprovementOperator:
    def __init__(self, llm : LanguageModel, config : str):
        pass

    #
    # Given the state-action pairs for a state and a description
    # of the task to be accomplished, perform chain-of-thought reasoning
    # to determine the best policy.
    #
    # Return this policy as well as the strategic reasoning behind it.
    #
    def reason(self, values, task_instruction):
        pass
