import numpy as np
from scipy.integrate import solve_ivp

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


    def solve(self, t_max=10.0, dt=0.01):
        """
        Integrate the equations of motion over time.

        Parameters
        ----------
        t_max : float
            Total simulation time in seconds.
        dt : float
            Time step for output (not the integration step).

        Returns
        -------
        dict with keys:
            "t"      : array, shape (N,) — time points
            "theta1" : array, shape (N,) — angle of first pendulum
            "theta2" : array, shape (N,) — angle of second pendulum
            "omega1" : array, shape (N,) — angular velocity of first pendulum
            "omega2" : array, shape (N,) — angular velocity of second pendulum
        """
        t_span = (0.0, t_max)
        t_eval = np.arange(0.0, t_max, dt)

        sol = solve_ivp(
            fun=self.equations_of_motion,
            t_span=t_span,
            y0=self.state0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return {
            "t":      sol.t,
            "theta1": sol.y[0],
            "theta2": sol.y[1],
            "omega1": sol.y[2],
            "omega2": sol.y[3],
            "energy": self.energy(sol.y),
        }


    def energy(self, state):
        """
        Compute total mechanical energy E = T + V.

        Parameters
        ----------
        state : array-like, shape (4,) or (4, N)
            State vector(s) [theta1, theta2, omega1, omega2].

        Returns
        -------
        float or array
            Total energy at the given state(s).
        """
        theta1, theta2, omega1, omega2 = state

        # Énergie cinétique
        T = (
            0.5 * (self.m1 + self.m2) * self.l1**2 * omega1**2
            + 0.5 * self.m2 * self.l2**2 * omega2**2
            + self.m2 * self.l1 * self.l2 * omega1 * omega2 * np.cos(theta1 - theta2)
        )

        # Énergie potentielle
        V = (
            -self.m1 * self.g * self.l1 * np.cos(theta1)
            - self.m2 * self.g * (self.l1 * np.cos(theta1) + self.l2 * np.cos(theta2))
        )

        return T + V