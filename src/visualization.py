import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def plot_angles(result, filename="figures/angles.png"):
    """
    Plot theta1(t) and theta2(t) over time.

    Parameters
    ----------
    result : dict
        Output from DoublePendulum.solve().
    filename : str
        Path to save the figure.
    """
    t      = result["t"]
    theta1 = result["state"][0]
    theta2 = result["state"][1]

    # Normalisation entre -π et π
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, np.degrees(theta1), color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("θ₁ (degrés)")
    axes[0].set_title("Trajectoires angulaires du double pendule")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, np.degrees(theta2), color="tomato", linewidth=0.8)
    axes[1].set_ylabel("θ₂ (degrés)")
    axes[1].set_xlabel("Temps (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {filename}")

def animate_pendulum(dp, result, filename="figures/animation.gif", fps=30):
    """
    Create an animation of the double pendulum.

    Parameters
    ----------
    dp : DoublePendulum
        The pendulum object (for l1, l2).
    result : dict
        Output from DoublePendulum.solve().
    filename : str
        Path to save the GIF.
    fps : int
        Frames per second.
    """
    theta1 = result["state"][0]
    theta2 = result["state"][1]

    # Positions cartésiennes des deux masses
    x1 =  dp.l1 * np.sin(theta1)
    y1 = -dp.l1 * np.cos(theta1)
    x2 = x1 + dp.l2 * np.sin(theta2)
    y2 = y1 - dp.l2 * np.cos(theta2)

    # Setup de la figure
    L = dp.l1 + dp.l2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-L * 1.1, L * 1.1)
    ax.set_ylim(-L * 1.1, L * 1.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double pendule")

    # Éléments graphiques qu'on va animer
    line,  = ax.plot([], [], "o-", color="steelblue", linewidth=2, markersize=8)
    trace, = ax.plot([], [], "-", color="tomato", linewidth=0.5, alpha=0.5)

    trace_x, trace_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(frame):
        # Position des tiges
        line.set_data(
            [0, x1[frame], x2[frame]],
            [0, y1[frame], y2[frame]],
        )

        # Trace de la trajectoire de la masse 2
        trace_x.append(x2[frame])
        trace_y.append(y2[frame])
        trace.set_data(trace_x, trace_y)

        return line, trace

    # On prend 1 frame sur 3 pour alléger le GIF
    frames = range(0, len(theta1), 3)
    anim = FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=True
    )

    anim.save(filename, writer="pillow", fps=fps)
    plt.close()
    print(f"Animation sauvegardée : {filename}")


def plot_phase_space(result, filename="figures/phase_space.png"):
    """
    Plot phase space portraits (ω vs θ) for both pendulum arms.

    Parameters
    ----------
    result : dict
        Output from DoublePendulum.solve().
    filename : str
        Path to save the figure.
    """
    t      = result["t"]
    theta1 = (result["state"][0] + np.pi) % (2 * np.pi) - np.pi
    omega1 = result["state"][2]
    theta2 = (result["state"][1] + np.pi) % (2 * np.pi) - np.pi
    omega2 = result["state"][3]

    norm = plt.Normalize(t[0], t[-1])
    cmap = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Espace des phases du double pendule")

    for ax, theta, omega, label in [
        (axes[0], theta1, omega1, "1"),
        (axes[1], theta2, omega2, "2"),
    ]:
        # Construire les segments pour LineCollection
        points = np.array([theta, omega]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.6, alpha=0.8)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel(f"θ{label} (rad)")
        ax.set_ylabel(f"ω{label} (rad/s)")
        ax.set_title(f"Pendule {label}")
        ax.grid(True, alpha=0.3)

    fig.colorbar(lc, ax=axes, label="Temps (s)", fraction=0.02, pad=0.04)

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {filename}")


def plot_sensitivity(t, divergence, epsilon, filename="figures/sensitivity.png"):
    """
    Plot trajectory divergence over time (log and linear scales).

    Parameters
    ----------
    t : np.ndarray
        Time array from DoublePendulum.solve().
    divergence : np.ndarray
        Distance between two trajectories at each time step.
    epsilon : float
        Initial perturbation magnitude (shown in title).
    filename : str
        Path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Sensibilité aux conditions initiales (ε = {epsilon:.2e})")

    axes[0].semilogy(t, divergence, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Δ(t) — échelle log")
    axes[0].set_title("Divergence des trajectoires")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, divergence, color="tomato", linewidth=0.8)
    axes[1].set_ylabel("Δ(t) — échelle linéaire")
    axes[1].set_xlabel("Temps (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {filename}")