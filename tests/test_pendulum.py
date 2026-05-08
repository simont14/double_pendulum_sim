from src.pendulum import DoublePendulum
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
    assert "theta1" in result
    assert "theta2" in result
    assert "omega1" in result
    assert "omega2" in result
    assert len(result["t"]) == 500  # 5.0 / 0.01 = 500 points


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

    states = np.array([
        result["theta1"],
        result["theta2"],
        result["omega1"],
        result["omega2"],
    ])

    E0 = dp.energy(states[:, 0])
    E  = dp.energy(states)

    relative_error = np.max(np.abs(E - E0) / np.abs(E0))
    assert relative_error < 1e-6