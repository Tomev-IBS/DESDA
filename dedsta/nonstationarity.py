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
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> float:
        """
        Computes the value on nonstationarity degree.

        :return:
            Nonstationarity degree value.
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
        Estimates the value of the nonstationarity degree. The formula for 1d
        nonstationarity degree :math:`\nu` is as follows:

        .. math::
            \nu = sgm(0.995 \cdot KPSS - 2.932),

        where :math:`KPSS` is the value of the KPSS test, and :math:`sgm` is the
        sigmoid function. The values of the coefficients in the formula were selected
        empirically. The whole process was described in
        TODO: add reference to the paper when it comes out.

        For the multidimensional case, the maximum value of the nonstationarity
        degree is taken. This was both empirically proven to be more effective, but
        also makes reasonable sense, as the stream should be considered non-stationary
        even if only one of its dimensions is non-stationary.

        :return:
            Estimated nonstationarity degree in the range [0, 1].
        """
        dimension_nonstationarity_degrees: List[float] = []

        for i in range(self._data.shape[1]):
            kpss_value, _, _, _ = kpss(self._data[:, i])
            dimension_nonstationarity_degrees.append(sgm(0.995 * kpss_value - 2.932))

        return max(dimension_nonstationarity_degrees)
