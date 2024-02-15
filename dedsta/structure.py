__author__ = "Tomasz Rybotycki"

from typing import List
import abc
from math import floor
from numpy.typing import ArrayLike


class DEDSTADataPoint:
    """
    Class representing a single data point in DEDSTA algorithm. The value can either
    be a single number or a vector.
    """
    def __init__(self, value: ArrayLike):
        """
        Initializes the data point.

        :param value:
            New data point sampled from the stream.
        """
        self.value: ArrayLike = value
        self.weight: float = 1.0


class DEDSTAReservoir(abc.ABC):
    """
    Class representing a reservoir used in DEDSTA algorithm.
    """
    def __init__(self, min_size: int, max_size: int) -> None:
        """
        Initializes the reservoir.

        :param min_size:
            Minimum size of the reservoir.
        :param max_size:
            Maximum size of the reservoir.
        """
        self.min_size: int = min_size
        self.max_size: int = max_size

        self.data_points: List[DEDSTADataPoint] = list()

    def add(self, data_point: ArrayLike) -> None:
        """
        Adds a new data point to the reservoir.

        :param data_point:
            New data point sampled from the stream.
        """
        if self.size() >= self.max_size:
            self.remove()

        self.data_points.insert(0, DEDSTADataPoint(data_point))

    @abc.abstractmethod
    def remove(self) -> None:
        """
        Removes a data point from the reservoir. The choice of the data point depends
        on the implementation.
        """
        raise NotImplementedError

    def get_points(self) -> List[ArrayLike]:
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

    def reset_weights(self) -> None:
        """
        Resets all weights in the reservoir.
        """
        for point in self.data_points:
            point.weight = 1.0

    def size(self) -> int:
        """
        Returns the size of the reservoir.

        :return:
            Size of the reservoir.
        """
        return len(self.data_points)


class DEDSTASlidingWindowReservoir(DEDSTAReservoir):
    """
    Class representing a sliding window variant of the reservoir used in DEDSTA
    algorithm.
    """
    def __init__(self, min_size: int, max_size: int) -> None:
        """
        Initializes the reservoir.

        :param min_size:
            Minimum size of the reservoir.
        :param max_size:
            Maximum size of the reservoir.
        """
        super().__init__(min_size, max_size)

    def remove(self) -> None:
        """
        Removes the last data point from the reservoir.
        """
        self.data_points.pop(-1)


class DEDSTAModule(abc.ABC):
    """
    Abstract class representing a module in DEDSTA algorithm.
    """
    def __init__(self, reservoir: DEDSTAReservoir) -> None:
        self.reservoir: DEDSTAReservoir = reservoir

    @abc.abstractmethod
    def apply(self, nonstationarity: float) -> None:
        """
        Applies the module to modify the weights of the data points in the reservoir.

        :param nonstationarity:
            The degree of nonstationarity. It should be a number between 0 and 1.
        """
        raise NotImplementedError


class DEDSTAAgingModule(DEDSTAModule):
    """
    Class representing a module responsible for updating weights in DEDSTA algorithm.
    """
    def __init__(self, reservoir: DEDSTAReservoir) -> None:
        super().__init__(reservoir)

    def apply(self, nonstationarity: float) -> None:
        """
        Updates weights of all data points in the reservoir.

        :note:
            It assumes that the new element is inserted at the beginning of the
            reservoir.

        :param nonstationarity:
            Nonstationarity degree.
        """
        n_elements: int = self.reservoir.size()

        for i in range(n_elements):
            self.reservoir.data_points[i].weight = (
                2 * (1 - i * nonstationarity / n_elements)
            )


class DEDSTAReductionModule(DEDSTAModule):
    """
    Class representing a module responsible for reducing the size of the reservoir in
    DEDSTA algorithm. The reduction is expected in the nonstationarity regime.
    """
    def __init__(self, reservoir: DEDSTAReservoir) -> None:
        """
        Initializes the module.

        :param reservoir:
            Reservoir to which the module should be applied.
        """
        super().__init__(reservoir)

    def apply(self, nonstationarity: float) -> None:
        """
        Reduces the size of the reservoir. The desired size varies depending on the
        minimal and maximal size of the reservoir and on the current degree of
        nonstationarity.

        We first compute the intermediate value :math:`m*` as follows:
        .. math::
            m* = \text{floor}(1.1 m_0 (1 - \nu)),

        where :math:`m_0` is the maximal size of the reservoir, and :math:`\nu` is the
        nonstationarity degree. The desired size of the elements in the current step
        is equal to :math:`m*` or :math:`m_0` (:math:`m_min`) if :math:`m*' is greater
        (smaller) than :math:`m_0` (:math:`m_min`).

        :param nonstationarity:
            Nonstationarity degree.
        """
        m = floor(1.1 * self.reservoir.max_size * (1 - nonstationarity))

        if m < self.reservoir.min_size:
            m = self.reservoir.min_size
        if m > self.reservoir.max_size:
            m = self.reservoir.max_size

        while self.reservoir.size() > m:
            self.reservoir.remove()
