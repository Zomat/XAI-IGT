from dataclasses import dataclass, field
import numpy as np
from agents.base import BaseAgent
from environment.igt_env import CardOutcome


@dataclass
class PVLDeltaState:
    Ev: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class PVLDeltaAgent(BaseAgent):
    A: float  # Shape/Sensitivity ∈ [0, 1]
    w: float  # Loss Aversion ∈ [0, 5]
    a: float  # Learning Rate ∈ [0, 1]
    c: float  # Choice Consistency ∈ [0, 5]

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    state: PVLDeltaState = field(default_factory=PVLDeltaState)
    total_trials_done: int = 0

    def _utility(self, outcome: CardOutcome) -> float:
        gain = float(outcome.gain)
        loss = float(outcome.loss)
        u_gain = gain ** self.A
        u_loss = self.w * (abs(loss) ** self.A)
        return u_gain - u_loss

    def get_action_probs(self) -> np.ndarray:
        scaled_Ev = self.c * self.state.Ev
        max_Ev = np.max(scaled_Ev)
        exp_v = np.exp(scaled_Ev - max_Ev)
        return exp_v / np.sum(exp_v)

    def act(self) -> int:
        probs = self.get_action_probs()
        return self.rng.choice([0, 1, 2, 3], p=probs)

    def update(self, deck_index: int, outcome: CardOutcome) -> None:
        utility = self._utility(outcome)
        self.total_trials_done += 1
        prediction_error = utility - self.state.Ev[deck_index]
        self.state.Ev[deck_index] += self.a * prediction_error
