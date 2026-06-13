from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def act(self) -> int:
        """Sample an action (deck index 0-3)."""
        ...

    @abstractmethod
    def update(self, deck_index: int, outcome) -> None:
        """Update internal state after observing an outcome."""
        ...

    @abstractmethod
    def get_action_probs(self) -> np.ndarray:
        """Return softmax probabilities over 4 decks given current state.
        Used for likelihood computation during model fitting.
        """
        ...
