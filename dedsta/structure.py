__author__ = "Tomasz Rybotycki"

import numpy as np
from KDEpy.kernel_funcs import Kernel, gaussian
from typing import List
import abc
from numpy.typing import NDArray


class DEDSTADataPoint:
    """
    Class representing a single data point in DEDSTA algorithm. The value can either
    be a single number or a vector.
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
