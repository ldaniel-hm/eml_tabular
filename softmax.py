"""Simulación ejemplo para Softmax y visualización de regret por brazo."""

from typing import List

import numpy as np

from algorithms import Algorithm
from plotting.regacu import plot_cumulative_regret


def plot_regret_for_each_arm(
    steps: int,
    cumulative_regret_by_arm: np.ndarray,
    algorithms: List[Algorithm],
):
    """Grafica el regret acumulado para cada brazo usando `plot_cumulative_regret`.

    Parameters
    ----------
    steps : int
        Número de pasos de tiempo.
    cumulative_regret_by_arm : np.ndarray
        Tensor con forma (n_arms, n_algorithms, steps).
    algorithms : List[Algorithm]
        Lista de algoritmos comparados.
    """
    n_arms = cumulative_regret_by_arm.shape[0]

    for arm_idx in range(n_arms):
        print(f"Graficando regret acumulado del brazo {arm_idx}")
        plot_cumulative_regret(
            steps=steps,
            cumulative_regret=cumulative_regret_by_arm[arm_idx],
            algorithms=algorithms,
        )
