import sys
sys.path.insert(0, ".")

from src.pendulum import DoublePendulum
from src.visualization import plot_angles, animate_pendulum, plot_phase_space, plot_sensitivity, plot_lyapunov, plot_chaos_divergence
from src.chaos import trajectory_divergence, lyapunov_exponent

dp = DoublePendulum(
    theta1_0=2.0,
    theta2_0=2.0,
    omega1_0=0.0,
    omega2_0=0.0,
)

result = dp.solve(t_max=20.0, dt=0.01)
plot_angles(result)

animate_pendulum(dp, result)
plot_phase_space(result)

epsilon = 1e-8
dp2 = DoublePendulum(theta1_0=2.0 + epsilon, theta2_0=2.0, omega1_0=0.0, omega2_0=0.0)
result2 = dp2.solve(t_max=20.0, dt=0.01)
div = trajectory_divergence(result, result2)
plot_sensitivity(result["t"], div, epsilon)

lambda_t, lambda_mean = lyapunov_exponent(result, result2, epsilon)
plot_lyapunov(result["t"][1:], lambda_t, lambda_mean)

plot_chaos_divergence(dp, result, result2, epsilon)