from dataclasses import dataclass, field
import numpy as np
from agents.base import BaseAgent
from environment.igt_env import CardOutcome


@dataclass
class AgentState:
    nominalP: float = 100.0
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)
    N: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class Agent(BaseAgent):
    loss_aversion: float
    perception_shape: float
    learning_rate: float
    forgetting_factor: float
    exploration_weight: float
    horizon: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    state: AgentState = field(default_factory=AgentState)
    total_trials_done: int = 0

    def _perception(self, outcome: CardOutcome) -> float:
        netto = float(outcome.gain + outcome.loss)
        if netto >= 0:
            return netto ** self.perception_shape
        else:
            return -self.loss_aversion * (abs(netto) ** self.perception_shape)

    def _psi_function(self, P_val: float, N_val: float) -> float:
        numerator = P_val * self.state.nominalP
        denominator = P_val + self.state.nominalP
        return numerator / denominator

    # def _score_simulation(self, deck_index: int, x_vec: np.ndarray, P_mat: np.ndarray) -> float:
    #     p_val = max(0.0, P_mat[deck_index, deck_index])
    #     return x_vec[deck_index] + (self.exploration_weight * p_val)

    # def _simulate_state_update(self, P: np.ndarray, N: np.ndarray, action: int):
    #     next_P = P.copy()
    #     next_N = N.copy()
    #     next_N[action] += 1
    #     curr_P = next_P[action, action]
    #     next_P[action, action] = self._psi_function(curr_P, 0)
    #     for i in range(4):
    #         if i != action:
    #             next_P[i, i] += self.forgetting_factor
    #     return next_P, next_N

    def get_dynamic_values(self) -> np.ndarray:
        current_scores = np.array([
            self._score_simulation(a, self.state.x, self.state.P) for a in range(4)
        ])

        if self.horizon == 0:
            return current_scores

        GAMMA, TAU = 0.7, 1.0

        # Baseline: soft value of the CURRENT state (before any action).
        # Using this as reference makes the lookahead term a delta — how much
        # better/worse is the future state after choosing each action.
        # This keeps the lookahead term on the same scale as current_scores.
        max_cur = np.max(current_scores)
        soft_baseline = max_cur + TAU * np.log(np.sum(np.exp((current_scores - max_cur) / TAU)))

        final_values = np.zeros(4)

        for action in range(4):
            sim_P, sim_N = self._simulate_state_update(self.state.P, self.state.N, action)
            future_scores = np.array([
                self._score_simulation(a1, self.state.x, sim_P) for a1 in range(4)
            ])
            max_fut = np.max(future_scores)
            soft_future_val = max_fut + TAU * np.log(np.sum(np.exp((future_scores - max_fut) / TAU)))

            # Delta: incremental change in future opportunity after this action.
            delta_future = soft_future_val - soft_baseline
            final_values[action] = current_scores[action] + GAMMA * delta_future

        return final_values

    def get_action_probs(self) -> np.ndarray:
        values = self.get_dynamic_values()
        tau = 1.0
        exp_v = np.exp((values - np.max(values)) / tau)
        return exp_v / np.sum(exp_v)

    def act(self) -> int:
        probs = self.get_action_probs()
        return self.rng.choice([0, 1, 2, 3], p=probs)

    def _simulate_state_update(self, P: np.ndarray, N: np.ndarray, action: int):
        next_P = P.copy()
        next_N = N.copy()
        next_N[action] += 1
        
        # 1. Spadek niepewności dla wybranej talii
        curr_P = next_P[action, action]
        next_P[action, action] = self._psi_function(curr_P, 0)
        
        # 2. Narastanie znudzenia (powrót do nominalP) dla pozostałych
        for i in range(4):
            if i != action:
                diff = self.state.nominalP - next_P[i, i]
                next_P[i, i] += self.forgetting_factor * diff 
                
        return next_P, next_N

    def _score_simulation(self, deck_index: int, x_vec: np.ndarray, P_mat: np.ndarray) -> float:
        p_val = max(0.0, P_mat[deck_index, deck_index])
        # Zmieniamy log na sqrt dla lepszej dynamiki sygnału
        uncertainty_bonus = self.exploration_weight * np.sqrt(p_val) 
        return x_vec[deck_index] + uncertainty_bonus

    def update(self, deck_index: int, outcome: CardOutcome) -> None:
        self.state.N[deck_index] += 1
        n_k = self.state.N[deck_index]
        P_k = self.state.P[deck_index, deck_index]
        est_x = self.state.x[deck_index]

        utility = self._perception(outcome)
        self.total_trials_done += 1

        # Aktualizacja wartości oczekiwanej (learning rate)
        prediction_error = utility - est_x
        self.state.x[deck_index] = est_x + (self.learning_rate * prediction_error)

        # 1. Aktualizacja niepewności talii wybranej
        self.state.P[deck_index, deck_index] = self._psi_function(P_k, n_k)

        # 2. SPÓJNE Z SYMULACJĄ: Narastanie znudzenia dla pozostałych
        for i in range(4):
            if i != deck_index:
                diff = self.state.nominalP - self.state.P[i, i]
                self.state.P[i, i] += self.forgetting_factor * diff

    # def update(self, deck_index: int, outcome: CardOutcome) -> None:
    #     self.state.N[deck_index] += 1
    #     n_k = self.state.N[deck_index]
    #     P_k = self.state.P[deck_index, deck_index]
    #     est_x = self.state.x[deck_index]

    #     utility = self._perception(outcome)
    #     self.total_trials_done += 1

    #     prediction_error = utility - est_x
    #     delta = self.learning_rate * prediction_error

    #     self.state.x[deck_index] = est_x + delta
    #     self.state.P[deck_index, deck_index] = self._psi_function(P_k, n_k)
    #     for i in range(4):
    #         if i != deck_index:
    #             self.state.P[i, i] += self.forgetting_factor
