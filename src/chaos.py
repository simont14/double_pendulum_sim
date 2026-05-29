import numpy as np


def trajectory_divergence(result1, result2):
    """
    Compute the Euclidean distance in state space between two trajectories.

    Parameters
    ----------
    result1, result2 : dict
        Outputs from DoublePendulum.solve(). Must share the same time grid.

    Returns
    -------
    divergence : np.ndarray of shape (N,)
        ‖state1[:, i] − state2[:, i]‖₂ at each time step.
    """
    diff = result1["state"] - result2["state"]
    return np.linalg.norm(diff, axis=0)


def lyapunov_exponent(result1, result2, epsilon):
    """
    Estimate the maximal Lyapunov exponent from two nearby trajectories.

    Parameters
    ----------
    result1, result2 : dict
        Outputs from DoublePendulum.solve(). Must share the same time grid.
        result2 should start with a perturbation of magnitude epsilon.
    epsilon : float
        Initial separation between the two trajectories.

    Returns
    -------
    lambda_t : np.ndarray of shape (N-1,)
        Running MLE estimate: (1/t) * ln(δ(t) / ε) at each time step.
    lambda_mean : float
        MLE averaged over the last 20 % of the trajectory.
    """
    div = trajectory_divergence(result1, result2)
    t = result1["t"]
    lambda_t = np.log(div[1:] / epsilon) / t[1:]
    cutoff = int(0.8 * len(lambda_t))
    lambda_mean = float(np.mean(lambda_t[cutoff:]))
    return lambda_t, lambda_mean
