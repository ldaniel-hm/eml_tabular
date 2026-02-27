"""
Module: plotting/regacu.py
Description: Función para graficar el Regret Acumulado.

Author: Tu Grupo
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from algorithms import Algorithm


def plot_cumulative_regret(
    steps: int,
    cumulative_regret: np.ndarray,
    algorithms: List[Algorithm],
):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.

    Parameters
    ----------
    steps : int
        Número de pasos de tiempo.
    cumulative_regret : np.ndarray
        Matriz (n_algorithms x steps) con el regret acumulado.
    algorithms : List[Algorithm]
        Lista de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))

    for idx, algo in enumerate(algorithms):
        label = type(algo).__name__

        if hasattr(algo, "epsilon"):
            label += f" (epsilon={algo.epsilon})"
        if hasattr(algo, "temperature"):
            label += f" (temperature={algo.temperature})"

        plt.plot(range(steps), cumulative_regret[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Regret Acumulado", fontsize=14)
    plt.title("Regret Acumulado vs Pasos de Tiempo", fontsize=16)
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()
