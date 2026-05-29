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
