import numpy as np


class DoublePendulum:
    """Double pendulum system."""

    def __init__(
        self,
        m1=1.0,
        m2=1.0,
        l1=1.0,
        l2=1.0,
        g=9.81,
        theta1_0=np.pi / 2,
        theta2_0=np.pi / 2,
        omega1_0=0.0,
        omega2_0=0.0,
    ):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.state0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0])
    
    def equations_of_motion(self, t, state):
        """
        Compute the derivatives of the state vector.

        Parameters
        ----------
        t : float
            Current time (not used, but required by solve_ivp).
        state : array-like, shape (4,)
            Current state [theta1, theta2, omega1, omega2].

        Returns
        -------
        list of float, shape (4,)
            Derivatives [omega1, omega2, alpha1, alpha2].
        """
        theta1, theta2, omega1, omega2 = state

        delta = theta1 - theta2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        denom = self.m1 + self.m2 * sin_delta**2

        alpha1 = (
            -self.m2 * self.l2 * omega2**2 * sin_delta
            - (self.m1 + self.m2) * self.g * np.sin(theta1)
            - self.m2 * self.l1 * omega1**2 * sin_delta * cos_delta
            + self.m2 * self.g * np.sin(theta2) * cos_delta
        ) / (self.l1 * denom)

        alpha2 = (
            (self.m1 + self.m2) * (
                self.l1 * omega1**2 * sin_delta
                - self.g * np.sin(theta2)
                + self.g * np.sin(theta1) * cos_delta
            )
            + self.m2 * self.l2 * omega2**2 * sin_delta * cos_delta
        ) / (self.l2 * denom)

        return [omega1, omega2, alpha1, alpha2]