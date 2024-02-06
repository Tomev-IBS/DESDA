__author__ = "Tomasz Rybotycki"


from KDEpy import FFTKDE
from numpy.typing import NDArray
from dedsta.structure import DEDSTAReservoir


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
