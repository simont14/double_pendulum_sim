from src.pendulum import DoublePendulum
from src.visualization import plot_phase_space, plot_chaos_divergence
from src.chaos import trajectory_divergence, lyapunov_exponent
import numpy as np


def test_default_params():
    dp = DoublePendulum()
    assert dp.m1 == 1.0
    assert dp.l1 == 1.0
    assert dp.g == 9.81
    assert dp.state0.shape == (4,)


def test_custom_params():
    dp = DoublePendulum(m1=2.0, l2=0.5, theta1_0=0.0)
    assert dp.m1 == 2.0
    assert dp.l2 == 0.5
    assert dp.state0[0] == 0.0  # theta1_0


def test_equilibrium_at_rest():
    dp = DoublePendulum(theta1_0=0.0, theta2_0=0.0, omega1_0=0.0, omega2_0=0.0)
    state = [0.0, 0.0, 0.0, 0.0]
    derivs = dp.equations_of_motion(0.0, state)

    assert abs(derivs[2]) < 1e-10  # alpha1 doit être ~0
    assert abs(derivs[3]) < 1e-10  # alpha2 doit être ~0


def test_solve_returns_correct_shape():
    dp = DoublePendulum()
    result = dp.solve(t_max=5.0, dt=0.01)

    assert "t"      in result
    assert "state"  in result
    assert "energy" in result
    assert len(result["t"]) == 500           # 5.0 / 0.01 = 500 points
    assert result["state"].shape == (4, 500)


def test_energy_at_rest_position():
    """At rest at the bottom (theta=0, omega=0), E should equal -(m1+m2)*g*l1 - m2*g*l2."""
    dp = DoublePendulum(theta1_0=0.0, theta2_0=0.0, omega1_0=0.0, omega2_0=0.0)
    state = [0.0, 0.0, 0.0, 0.0]
    E = dp.energy(state)
    expected = -(dp.m1 + dp.m2) * dp.g * dp.l1 - dp.m2 * dp.g * dp.l2
    assert abs(E - expected) < 1e-10


def test_solve_includes_energy():
    dp = DoublePendulum()
    result = dp.solve(t_max=5.0, dt=0.01)
    assert "energy" in result
    assert len(result["energy"]) == len(result["t"])


def test_energy_conservation():
    dp = DoublePendulum(
        theta1_0=np.pi / 4,
        theta2_0=np.pi / 6,
        omega1_0=0.0,
        omega2_0=0.0,
    )
    result = dp.solve(t_max=10.0, dt=0.01)

    states = result["state"]

    E0 = dp.energy(states[:, 0])
    E  = dp.energy(states)

    relative_error = np.max(np.abs(E - E0) / np.abs(E0))
    assert relative_error < 1e-6


def test_phase_space_plot_runs(tmp_path):
    dp = DoublePendulum()
    result = dp.solve(t_max=5.0, dt=0.01)
    out = tmp_path / "phase_space.png"
    plot_phase_space(result, filename=str(out))
    assert out.exists()


def test_divergence_zero_for_identical_trajectories():
    dp = DoublePendulum()
    r = dp.solve(t_max=5.0, dt=0.01)
    div = trajectory_divergence(r, r)
    assert np.all(div == 0.0)


def test_divergence_grows_in_chaotic_regime():
    epsilon = 1e-8
    dp1 = DoublePendulum(theta1_0=2.0, theta2_0=2.0)
    dp2 = DoublePendulum(theta1_0=2.0 + epsilon, theta2_0=2.0)
    r1 = dp1.solve(t_max=20.0, dt=0.01)
    r2 = dp2.solve(t_max=20.0, dt=0.01)
    div = trajectory_divergence(r1, r2)
    assert div[-1] / div[1] > 100


def test_lyapunov_positive_in_chaotic_regime():
    epsilon = 1e-8
    dp1 = DoublePendulum(theta1_0=2.0, theta2_0=2.0)
    dp2 = DoublePendulum(theta1_0=2.0 + epsilon, theta2_0=2.0)
    r1 = dp1.solve(t_max=20.0, dt=0.01)
    r2 = dp2.solve(t_max=20.0, dt=0.01)
    _, lam = lyapunov_exponent(r1, r2, epsilon)
    assert lam > 0


def test_chaos_divergence_plot_runs(tmp_path):
    epsilon = 1e-8
    dp1 = DoublePendulum(theta1_0=2.0, theta2_0=2.0)
    dp2 = DoublePendulum(theta1_0=2.0 + epsilon, theta2_0=2.0)
    r1 = dp1.solve(t_max=5.0, dt=0.01)
    r2 = dp2.solve(t_max=5.0, dt=0.01)
    out = tmp_path / "chaos_divergence.png"
    plot_chaos_divergence(dp1, r1, r2, epsilon, filename=str(out))
    assert out.exists()


def test_lyapunov_shape():
    epsilon = 1e-8
    dp1 = DoublePendulum(theta1_0=2.0, theta2_0=2.0)
    dp2 = DoublePendulum(theta1_0=2.0 + epsilon, theta2_0=2.0)
    r1 = dp1.solve(t_max=5.0, dt=0.01)
    r2 = dp2.solve(t_max=5.0, dt=0.01)
    lambda_t, _ = lyapunov_exponent(r1, r2, epsilon)
    assert lambda_t.shape == (len(r1["t"]) - 1,)