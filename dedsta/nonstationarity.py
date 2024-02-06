from numpy.typing import NDArray, ArrayLike
from abc import ABC, abstractmethod
from numpy import array, delete, vstack
from statsmodels.tsa.stattools import kpss
from typing import List
from math import exp


def sgm(x: float) -> float:
    """
    Sigmoid function.

    :param x:
        Input value.

    :return:
        Value of the sigmoid function.
    """
    return 1 / (1 + exp(-x))


class NonstationarityDegreeEstimator(ABC):
    """
    Abstract class for nonstationarity degree estimation. Since it will be called
    in an online context, it should be able to update itself fast. The evaluation
    should also be done quickly.
    """

    @abstractmethod
    def update(self, data_point: NDArray[float]) -> None:
        """
        Updates the estimator with a new data point.

        :param data_point:
            New data point.

        :return:
            None.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the estimator.

        :return:
            Estimated nonstationarity degree.
        """
        raise NotImplementedError


class KPSSNonstationarityDegreeEstimator(NonstationarityDegreeEstimator):
    """
    Nonstationarity degree estimator based on KPSS test.
    """

    def __init__(self, window_size: int = 600):
        """
        Constructor for the KPSSNonstationarityDegreeEstimator.

        :param window_size:
            Size of the window used for the estimation. It's set to 600 by default,
            as this value was empirically shown to work well.
        """
        super().__init__()
        self._window_size: int = window_size
        self._data: ArrayLike[float] = array([])

    def update(self, data_point: NDArray[float]) -> None:
        """
        Updates the estimator with a new data point.

        :param data_point:
            New data point.
        """
        if len(self._data) == 0:
            self._data = array([data_point])
            return

        if len(self._data) == self._window_size:
            print("Removing first element")
            self._data = delete(self._data, 0, 0)

        self._data = vstack((self._data, data_point))

    def evaluate(self) -> float:
        """
        Evaluates the estimator.

        :return:
            Estimated nonstationarity degree.
        """
        dimension_nonstationarity_degrees: List[float] = []

        for i in range(self._data.shape[1]):
            kpss_value, _, _, _ = kpss(self._data[:, i])
            dimension_nonstationarity_degrees.append(sgm(0.995 * kpss_value - 2.932))

        return max(dimension_nonstationarity_degrees)
