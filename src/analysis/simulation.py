import numpy as np
from environment.igt_env import EnvironmentBechara, get_lhs_samples  # noqa: re-exported


def generate_data(params, config, n_trials=100, horizon=0, seed=None):
    """Simulate one agent playing the IGT for n_trials steps.

    Args:
        params: parameter vector matching config.param_names
        config: AgentConfig instance
        n_trials: number of card draws
        horizon: look-ahead depth (ignored when config.supports_horizon=False)
        seed: RNG seed for reproducibility

    Returns:
        (actions, outcomes) lists of length n_trials
    """
    rng = np.random.default_rng(seed)
    env = EnvironmentBechara()
    decks = env.create_live_game_decks()

    agent = config.create_agent(params, horizon=horizon, rng=rng)

    actions, outcomes = [], []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    d_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for _ in range(n_trials):
        a = agent.act()
        d_char = d_map[a]
        card = decks[d_char][counts[a] % len(decks[d_char])]
        counts[a] += 1
        agent.update(a, card)
        actions.append(a)
        outcomes.append(card)

    return actions, outcomes
