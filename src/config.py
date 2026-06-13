from dataclasses import dataclass, field
import numpy as np


@dataclass
class AgentConfig:
    """Configuration for an agent model: class, parameter space, and metadata."""
    name: str
    agent_class: type
    param_names: list
    bounds: list
    supports_horizon: bool = True
    default_test_params: list = field(default_factory=list)

    def create_agent(self, params, horizon=0, rng=None):
        """Instantiate the agent with given parameters."""
        if rng is None:
            rng = np.random.default_rng()
        if self.supports_horizon:
            return self.agent_class(*params, horizon=horizon, rng=rng)
        else:
            return self.agent_class(*params, rng=rng)


# --- Agent registry ---
# Import here to avoid circular imports in agents/__init__.py
from agents import Agent, PVLDeltaAgent, ORLAgent, AIFAgent
from test_vectors import GOOD_VECTORS, BAD_VECTORS, ALL_VECTORS
AGENT_CONFIGS = {
    "Agent": AgentConfig(
        name="KPVL", # autorski model
        agent_class=Agent,
        param_names=[
            r"$\lambda$ (Loss Aversion)",
            r"$\rho$ (Perception Shape)",
            r"$\alpha$ (Learning Rate)",
            r"$f$ (Boredom Factor)",
            r"$\beta$ (Exploration Weight)"
        ],
        bounds=[(0.01, 5.0), (0.01, 0.99), (0.01, 0.99), (0.01, 0.5), (0.01, 5.0)],
        supports_horizon=True,
        default_test_params=BAD_VECTORS["worst_case"] #GOOD_VECTORS["ideal"] #[0.15, 0.75, 0.5, 0.08, 4.0],
    ),
    "PVLDeltaAgent": AgentConfig(
        name="PVL-Delta",
        agent_class=PVLDeltaAgent,
        param_names=[
            "A (Shape)",
            "w (Loss Aversion)",
            "a (Learning Rate)",
            "c (Consistency)"
        ],
        bounds=[(0.01, 1.0), (0.01, 5.0), (0.01, 1.0), (0.01, 5.0)],
        supports_horizon=False,
        default_test_params=[0.75, 0.15, 0.5, 0.1],
    ),
    "ORLAgent": AgentConfig(
        name="ORL",
        agent_class=ORLAgent,
        param_names=[
            r"$A_{rew}$ (Reward L-Rate)", 
            r"$A_{pun}$ (Punish L-Rate)", 
            r"$K'$ (Decay Rate)",
            r"$\beta_F$ (Freq Weight)", 
            r"$\beta_P$ (Perseverance)"
        ],
        bounds=[
            (0.001, 0.999),   # A_rew
            (0.001, 0.999),   # A_pun
            (0.0,   5.0),     # K' (K = 3^K' − 1, so K ∈ [0, 242])
            (-10.0, 10.0),    # beta_F
            (-10.0, 10.0),    # beta_P
        ],
        supports_horizon=False,
        default_test_params=[0.3, 0.5, 1.0, 1.5, 1.0],
    )
}

# --- Run settings ---
# Change ACTIVE_AGENT to switch models; all analyses adapt automatically.
# ACTIVE_AGENT is ignored in COMPARISON mode — all models are used.
ACTIVE_AGENT = "Agent"

MODE = "RELIABILITY_SCAN"  # which analysis to run; see valid modes below
REPLOT = False
# Valid modes:
#   RECOVERY          — parameter recovery for ACTIVE_AGENT
#   DIST_RECOVERY     — distributional recovery for ACTIVE_AGENT
#   PSP               — parameter space partitioning for ACTIVE_AGENT
#   FULL_LANDSCAPE    — NLL landscape for ACTIVE_AGENT
#   DIAGNOSTICS       — repeated MLE diagnostics for ACTIVE_AGENT
#   RELIABILITY_SCAN  — MLE-based reliability map across parameter space for ACTIVE_AGENT
#   MCMC              — Bayesian posterior sampling for a single point (ACTIVE_AGENT)
#   MCMC_SCAN         — MCMC-based reliability scan across parameter space (ACTIVE_AGENT)
#   COMPARISON        — cross-fit comparison across ALL models (ignores ACTIVE_AGENT)

N_AGENTS        = 300
HORIZON_TO_TEST = 0
N_TRIALS        = 100
SEED            = 42

# RELIABILITY_SCAN settings
# Total fits = SCAN_N_POINTS × SCAN_N_REPEATS
SCAN_N_POINTS  = 500
SCAN_N_REPEATS = 100

# MCMC settings (used by both MCMC and MCMC_SCAN modes)
# n_walkers must be ≥ 2 × n_params:
#   Agent        (5 params) → min 10,  use 32
#   ORLAgent     (5 params) → min 10,  use 32
#   PVLDeltaAgent(4 params) → min  8,  use 32
MCMC_N_WALKERS = 32
MCMC_N_BURNIN  = 500   # burn-in for single MCMC run
MCMC_N_STEPS   = 2000  # production steps for single MCMC run

# MCMC_SCAN settings — MCMC-based reliability scan across parameter space
# Total likelihood evaluations ≈ MCMC_SCAN_N_POINTS × MCMC_N_WALKERS × (burnin + steps)
#   Test run  (10 pts):  10 × 32 × 1300 =    416 000  evals  (~5 min)
#   Full run (100 pts): 100 × 32 × 1300 =  4 160 000  evals  (~1-2h)
MCMC_SCAN_N_POINTS = 100    # set to 100 for a full run
MCMC_SCAN_N_STEPS  = 1000  # shorter than single MCMC; sufficient for scan
MCMC_SCAN_N_BURNIN = 300

# COMPARISON settings
# Total fits = COMPARISON_N_AGENTS × len(AGENT_CONFIGS)^2 × COMPARISON_N_REPEATS
# Example: 200 agents × 3 models^2 × 3 repeats = 5400 fits
# Start with N_AGENTS=50 to verify correctness before a full run.
COMPARISON_N_AGENTS = 200
COMPARISON_N_REPEATS = 3
#COMPARISON_N_AGENTS   = 200   # synthetic participants per data-generating model
#COMPARISON_N_REPEATS  = 3     # independent fits per (agent × fit_model) pair
COMPARISON_N_STARTS   = 10    # L-BFGS-B starting points per fit (>= 10 recommended)
# Which models to include — defaults to all; restrict to subset if needed:
# COMPARISON_MODELS = ["PVLDeltaAgent", "Agent"]
COMPARISON_MODELS = list(AGENT_CONFIGS.keys())
