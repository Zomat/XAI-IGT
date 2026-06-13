from abc import ABC, abstractmethod
import numpy as np


class BehaviorStrategy(ABC):
    @abstractmethod
    def classify(self, actions) -> str:
        pass

    def get_counts(self, actions):
        actions_arr = np.array(actions).astype(int)
        return np.bincount(actions_arr, minlength=4)  # A, B, C, D


class SteingroeverStrategy(BehaviorStrategy):
    """Classification scheme from Steingroever et al. (2013)."""

    def __init__(self, restricted=False):
        self.restricted = restricted
        self.threshold = 65 if restricted else 51
        self.name = "Restricted" if restricted else "Broad"

    def classify(self, actions) -> str:
        A, B, C, D = self.get_counts(actions)

        if (C + D) >= self.threshold:
            return "GOB"        # Good-Over-Bad
        elif (A + B) >= self.threshold:
            return "BOG"        # Bad-Over-Good
        elif (B + D) >= self.threshold:
            return "IOF"        # Infrequent-Over-Frequent
        elif (A + C) >= self.threshold:
            return "FOI"        # Frequent-Over-Infrequent
        else:
            return "Remaining"
