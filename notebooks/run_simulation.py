import sys
sys.path.insert(0, ".")

from src.pendulum import DoublePendulum
from src.visualization import plot_angles, animate_pendulum

dp = DoublePendulum(
    theta1_0=2.0,
    theta2_0=2.0,
    omega1_0=0.0,
    omega2_0=0.0,
)

result = dp.solve(t_max=20.0, dt=0.01)
plot_angles(result)

animate_pendulum(dp, result)