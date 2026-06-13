from dataclasses import dataclass, field
import numpy as np
from agents.base import BaseAgent
from environment.igt_env import CardOutcome


@dataclass
class AIFState:
    """Gaussian beliefs q(μ_k) = N(m_k, 1/λ_k) over the mean outcome of each deck."""
    m: np.ndarray = field(default_factory=lambda: np.zeros(4))   # posterior mean
    lam: np.ndarray = field(default_factory=lambda: np.ones(4))  # posterior precision


@dataclass
class AIFAgent(BaseAgent):
    """Active Inference agent for the IGT (Friston et al. formulation).

    The agent maintains Gaussian beliefs q(μ_k) = N(m_k, 1/λ_k) about the mean
    net outcome of each deck k.  Beliefs are updated via exact Bayesian inference
    (Normal-Normal conjugate model) rather than a delta rule.

    Action selection minimises *expected free energy* G(a):

        -G(a) = pragmatic_value(a) + β_info · epistemic_value(a)

    Components
    ----------
    pragmatic_value(a)  = m_a
        Expected outcome under current beliefs; drives exploitation of
        known-good decks.

    epistemic_value(a)  = ½ · log(1 + η / λ_a)
        Expected information gain (mutual information between the next
        observation and the belief about μ_a).  Large when λ_a is small
        (deck uncertain) → drives exploration.

    Action probabilities: P(a) ∝ exp(γ · (−G(a)))

    Belief update (Normal-Normal conjugate)
    ----------------------------------------
    After observing x from deck k:
        λ_k ← λ_k + η
        m_k ← (λ_k_old · m_k + η · x) / λ_k_new

    Outcomes are normalised by /100 (same convention as ORLAgent).

    Parameters
    ----------
    lambda_0      : Prior precision for each deck      ∈ [0.01, 5.0]
                    Low  → agent starts very uncertain → strong initial exploration.
                    High → agent starts confident      → exploits prior quickly.
    obs_precision : Observation / likelihood precision  ∈ [0.01, 10.0]
                    Controls how much each observation updates the belief.
                    Analogous to learning rate in delta-rule models.
    gamma         : Action selection precision          ∈ [0.01, 20.0]
                    Softmax inverse temperature.  High → near-deterministic;
                    low → near-random.
    beta_info     : Weight on epistemic (information-seeking) value ∈ [0.0, 5.0]
                    0 → purely pragmatic (exploitation only).
                    High → strong curiosity / exploration drive.
    """

    lambda_0: float       # prior precision
    obs_precision: float  # observation / likelihood precision
    gamma: float          # action selection precision
    beta_info: float      # epistemic weight

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    state: AIFState = field(default_factory=AIFState)
    total_trials_done: int = 0

    def __post_init__(self) -> None:
        # Initialise posterior precision to the prior value.
        self.state.lam = np.full(4, self.lambda_0)

    # ------------------------------------------------------------------
    # Core Active Inference quantities
    # ------------------------------------------------------------------

    def _neg_efe(self) -> np.ndarray:
        """Return −G(a) for each deck.

        −G(a) = pragmatic_value(a) + β_info · epistemic_value(a)

        pragmatic  : m_a            — expected outcome under current beliefs
        epistemic  : ½·log(1+η/λ_a) — expected information gain (MI)
        """
        m = self.state.m
        lam = self.state.lam
        pragmatic = m
        epistemic = 0.5 * np.log1p(self.obs_precision / lam)
        return pragmatic + self.beta_info * epistemic

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def get_action_probs(self) -> np.ndarray:
        scores = self.gamma * self._neg_efe()
        exp_s = np.exp(scores - np.max(scores))
        return exp_s / np.sum(exp_s)

    def act(self) -> int:
        return self.rng.choice([0, 1, 2, 3], p=self.get_action_probs())

    def update(self, deck_index: int, outcome: CardOutcome) -> None:
        x = float(outcome.gain + outcome.loss) / 100.0  # normalise (same as ORL)

        # Bayesian update: Normal-Normal conjugate
        lam_old = self.state.lam[deck_index]
        m_old = self.state.m[deck_index]

        lam_new = lam_old + self.obs_precision
        m_new = (lam_old * m_old + self.obs_precision * x) / lam_new

        self.state.m[deck_index] = m_new
        self.state.lam[deck_index] = lam_new
        self.total_trials_done += 1
