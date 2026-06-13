import numpy as np
from scipy.optimize import minimize, differential_evolution
from analysis.simulation import generate_data


def nll(params, actions, outcomes, config, horizon=0):
    """Negative log-likelihood of observed actions under the model.

    Uses get_action_probs() — works for any agent that implements BaseAgent.
    """
    try:
        agent = config.create_agent(params, horizon=horizon)
    except Exception:
        return 1e9

    neg_log_like = 0.0
    epsilon = 1e-10

    for t, a in enumerate(actions):
        probs = agent.get_action_probs()
        neg_log_like -= np.log(max(probs[a], epsilon))
        agent.update(a, outcomes[t])

    return neg_log_like


def fit_model(actions, outcomes, config, horizon=0, method="MLE", seed=None):
    """Fit model parameters to observed action sequences.

    Args:
        actions, outcomes: game data from generate_data()
        config: AgentConfig with bounds and default_test_params
        horizon: look-ahead depth
        method: 'MLE' (L-BFGS-B with multiple starts) or 'DE' (differential evolution)
        seed: RNG seed for random starting points

    Returns:
        Best-fit parameter array, or None if all optimizations failed.
    """
    rng = np.random.default_rng(seed)
    bounds = config.bounds

    best_res = None
    best_fun = float("inf")

    if method == "MLE":
        starts = [list(config.default_test_params)] if config.default_test_params else []
        for _ in range(3):
            starts.append([rng.uniform(b[0], b[1]) for b in bounds])

        for x0 in starts:
            res = minimize(
                nll, x0,
                args=(actions, outcomes, config, horizon),
                bounds=bounds,
                method="L-BFGS-B",
            )
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res

    elif method == "DE":
        best_res = differential_evolution(
            nll, bounds=bounds,
            args=(actions, outcomes, config, horizon),
            polish=True,
        )

    return best_res.x if (best_res is not None and best_res.success) else None


def compute_landscape_data(true_params, mle_params, actions, outcomes, config, horizon,
                           grid_res=25, n_profile=50):
    """Pre-compute NLL profiles and 2D grids for the identification matrix.

    Returns a dict for np.savez with keys:
      true_params, mle_params,
      profile_x   (n_profile, n_params),  profile_nll (n_profile, n_params),
      grid_axes_x (n_params, n_params, grid_res),
      grid_axes_y (n_params, n_params, grid_res),
      grid_Z      (n_params, n_params, grid_res, grid_res).
    """
    bounds = config.bounds
    n_params = len(config.param_names)
    base = list(mle_params)

    profile_x   = np.zeros((n_profile, n_params))
    profile_nll = np.zeros((n_profile, n_params))
    for i in range(n_params):
        x_vals = np.linspace(bounds[i][0], bounds[i][1], n_profile)
        profile_x[:, i] = x_vals
        for k, v in enumerate(x_vals):
            p = base.copy()
            p[i] = v
            profile_nll[k, i] = nll(p, actions, outcomes, config, horizon)

    grid_axes_x = np.zeros((n_params, n_params, grid_res))
    grid_axes_y = np.zeros((n_params, n_params, grid_res))
    grid_Z      = np.zeros((n_params, n_params, grid_res, grid_res))
    for i in range(n_params):
        for j in range(i):
            x_range = np.linspace(bounds[j][0], bounds[j][1], grid_res)
            y_range = np.linspace(bounds[i][0], bounds[i][1], grid_res)
            grid_axes_x[i, j] = x_range
            grid_axes_y[i, j] = y_range
            Z = np.zeros((grid_res, grid_res))
            for xi in range(grid_res):
                for yi in range(grid_res):
                    temp = base.copy()
                    temp[j] = x_range[xi]
                    temp[i] = y_range[yi]
                    Z[yi, xi] = nll(temp, actions, outcomes, config, horizon)
            grid_Z[i, j] = Z

    return dict(
        true_params=np.array(true_params),
        mle_params=np.array(mle_params),
        profile_x=profile_x,
        profile_nll=profile_nll,
        grid_axes_x=grid_axes_x,
        grid_axes_y=grid_axes_y,
        grid_Z=grid_Z,
    )


def run_optimization(true_params, config, horizon=0, seed=None, gen_config=None):
    """Generate synthetic data and recover parameters. Used for parallel recovery.

    Args:
        gen_config: AgentConfig used to *generate* data. Defaults to config (self-recovery).
                    Pass a different AgentConfig to enable cross-model fitting.
    """
    data_config = gen_config if gen_config is not None else config
    actions, outcomes = generate_data(
        true_params, config=data_config, n_trials=100, horizon=horizon, seed=seed
    )
    recovered_params = fit_model(
        actions, outcomes, config=config, horizon=horizon, method="MLE", seed=seed
    )
    return true_params, recovered_params
