from dataclasses import dataclass, field
import numpy as np
from agents.base import BaseAgent
from environment.igt_env import CardOutcome

# Number of unchosen decks in IGT (4 total - 1 chosen = 3)
_C = 3


@dataclass
class ORLState:
    EV: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Expected value
    EF: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Win frequency
    PS: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Choice perseverance


@dataclass
class ORLAgent(BaseAgent):
    """Outcome Representation Learning model — Haines et al. (2018).

    Parameters
    ----------
    A_rew   : Reward (positive) learning rate        ∈ (0, 1)
    A_pun   : Punishment (negative) learning rate    ∈ (0, 1)
    K_prime : Perseverance decay (K = 3^K' − 1)     ∈ [0, 5]
    beta_F  : Outcome-frequency weight               ∈ (−∞, ∞)
    beta_P  : Perseverance weight                    ∈ (−∞, ∞)

    Value signal:
        V_j = EV_j + EF_j · β_F + PS_j · β_P
    Action selection: softmax(V)
    """

    A_rew: float
    A_pun: float
    K_prime: float
    beta_F: float
    beta_P: float

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    state: ORLState = field(default_factory=ORLState)
    total_trials_done: int = 0

    def _value_signal(self) -> np.ndarray:
        s = self.state
        return s.EV + s.EF * self.beta_F + s.PS * self.beta_P

    def get_action_probs(self) -> np.ndarray:
        V = self._value_signal()
        exp_v = np.exp(V - np.max(V))
        return exp_v / np.sum(exp_v)

    def act(self) -> int:
        return self.rng.choice([0, 1, 2, 3], p=self.get_action_probs())

    def update(self, deck_index: int, outcome: CardOutcome) -> None:
        # KLUCZOWE: Skalowanie nagród (Haines et al. dzielą przez 100, aby EV było w skali ~1)
        x = float(outcome.gain + outcome.loss) / 100.0   
        sgn_x = np.sign(x)
        
        # Współczynnik dla wybranej talii
        lr_chosen = self.A_rew if x >= 0 else self.A_pun
        # Współczynnik dla NIEwybranych talii (odwrotny)
        lr_unchosen = self.A_pun if x >= 0 else self.A_rew

        K = 3.0 ** self.K_prime - 1.0            # K = 3^K' − 1
        decay = 1.0 / (1.0 + K)

        s = self.state

        # --- Expected value (chosen deck only) ---
        s.EV[deck_index] += lr_chosen * (x - s.EV[deck_index])

        # --- Win frequency ---
        # Wybrana talia
        s.EF[deck_index] += lr_chosen * (sgn_x - s.EF[deck_index])
        
        # Niewybrane talie — sygnał kontrfaktyczny -sgn(x)/3
        for j in range(4):
            if j != deck_index:
                s.EF[j] += lr_unchosen * (-sgn_x / _C - s.EF[j])

        # --- Choice perseverance ---
        # Wszystkie talie ulegają rozpadowi, wybrana dostaje dodatkowy impuls
        for j in range(4):
            if j == deck_index:
                s.PS[j] = 1.0 / (1.0 + K)
            else:
                s.PS[j] *= decay

        self.total_trials_done += 1
