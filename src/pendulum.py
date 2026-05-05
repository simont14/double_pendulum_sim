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