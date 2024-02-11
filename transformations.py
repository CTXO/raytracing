from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Transformation(ABC):
    matrix: npt.ArrayLike
    
    def validate_matrix(self):
        if not self.matrix.shape != (4, 4):
            raise ValueError(f"Matrix should be 4x4 but it is {self.matrix.shape[0]}x{self.matrix.shape[1]} instead. Matrix:\n{self.matrix}")
        last_row_of_matrix = self.matrix[3]
        expected_last_row = np.array([0, 0, 0, 1])
        if not np.array_equal(last_row_of_matrix, expected_last_row):
            raise ValueError(f"Matrix should have {expected_last_row} as last row, but it has {last_row_of_matrix} instead. Matrix:\n{self.matrix}")

    def __init__(self) -> None:
        pass