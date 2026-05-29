import matplotlib.pyplot as plt
import numpy as np


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