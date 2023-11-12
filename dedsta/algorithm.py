__author__ = "Tomasz Rybotycki"

import numpy as np
from KDEpy.kernel_funcs import Kernel, gaussian
from KDEpy import FFTKDE
from numpy.typing import NDArray
from typing import List
import abc


class DEDSTADataPoint:
    """
    Class representing a single data point in DEDSTA algorithm. The value can either
    be
    """
    def __init__(self, value: NDArray[float]):
        self.value: NDArray[float] = value
        self.weight: float = 1.0


class DEDSTAReservoir(abc.ABC):
    """
    Class representing a reservoir used in DEDSTA algorithm.
    """
    def __init__(self, min_size: int, max_size: int) -> None:
        self.min_size: int = min_size
        self.max_size: int = max_size

        self.data_points: List[DEDSTADataPoint] = list()

    @abc.abstractmethod
    def add(self, data_point: NDArray[float]) -> None:
        """
        Adds a new data point to the reservoir.

        :param data_point:
            New data point.

        :return:
            None.
        """
        raise NotImplementedError

    def get_points(self) -> List[NDArray[float]]:
        """
        Returns all points from the reservoir.

        :return:
            All points from the reservoir.
        """
        return [point.value for point in self.data_points]

    def get_weights(self) -> List[float]:
        """
        Returns all weights from the reservoir.

        :return:
            All weights from the reservoir.
        """
        return [point.weight for point in self.data_points]

    def size(self) -> int:
        return len(self.data_points)


class DEDSTASlidingWindowReservoir(DEDSTAReservoir):
    """
    Class representing a sliding window variant of the reservoir used in DEDSTA
    algorithm.
    """
    def __init__(self, min_size: int, max_size:int) -> None:
        super().__init__(min_size, max_size)

    def add(self, data_point: NDArray[float]) -> None:
        """
        Adds a new data point to the reservoir.

        :param data_point:
            New data point.

        :return:
            None.
        """
        if self.size() >= self.max_size:
            self.data_points.pop(0)

        self.data_points.append(DEDSTADataPoint(data_point))


class DEDSTA:
    """
    DEDSTA stands for (D)ensity (E)stitmation for (D)ata (S)treams with (T)tends
    (A)lgorithm. As the name suggests, it does exactly that.

    The implementation is based on KDEpy library.
    """
    def __init__(self, reservoir: DEDSTAReservoir, kernel: str = "gaussian") -> None:
        self.reservoir: DEDSTAReservoir = reservoir
        self.kernel: str = kernel

    def update(self, data_point: NDArray[float]) -> None:
        """
        Updates the estimator with a new data point.

        :param data_point:
            New data point.

        :return:
            None.
        """
        self.reservoir.add(data_point)

    def evaluate(self, grid_points: NDArray[float]) -> NDArray[float]:
        """
        Evaluates the estimator at given grid points.

        :note:
            The points have to be equidistant!

        :param grid_points:
            Grid points at which the estimator should be evaluated.

        :return:
            Estimated density values at given grid points.
        """

        kde: FFTKDE = FFTKDE(bw=1, kernel=self.kernel).fit(
            self.reservoir.get_points(),
            self.reservoir.get_weights()
        )

        return kde.evaluate(grid_points)
