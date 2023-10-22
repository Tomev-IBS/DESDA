__author__ = "Tomasz Rybotycki"


from KDEpy import BaseKDE
from KDEpy.kernel_funcs import Kernel


class DEDSTA:
    """
    DEDSTA stands for (D)ensity (E)stitmation for (D)ata (S)treams with (T)tends
    (A)lgorithm. As the name suggests, it does exactly that.

    The implementation is based on KDEpy library.
    """
    def __init__(self, kernel: Kernel = BaseKDE._kernel_functions["gaussian"]) -> None:
        pass

    def fit(self, data: np.ndarray) -> None:
        pass

