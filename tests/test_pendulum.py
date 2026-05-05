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