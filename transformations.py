from abc import ABC, abstractmethod
import math
import numpy as np
import numpy.typing as npt

class Transformation(ABC):
    matrix: npt.ArrayLike
    
    @staticmethod
    def get_sin_cos(angle):
        radians = math.radians(angle)
        return math.sin(radians), math.cos(radians)

    def validate_matrix(self):
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Matrix should be 4x4 but it is {self.matrix.shape[0]}x{self.matrix.shape[1]} instead. Matrix:\n{self.matrix}")
        last_row_of_matrix = self.matrix[3]
        expected_last_row = np.array([0, 0, 0, 1])
        if not np.array_equal(last_row_of_matrix, expected_last_row):
            raise ValueError(f"Matrix should have {expected_last_row} as last row, but it has {last_row_of_matrix} instead. Matrix:\n{self.matrix}")


class Translation(Transformation):
    def __init__(self, x, y, z) -> None:
        self.matrix = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        self.validate_matrix()



class RotationZ(Transformation):
    def __init__(self, angle) -> None:
        sin, cos = self.get_sin_cos(angle)
        self.matrix = np.array([
            [cos, -sin, 0, 0],
            [sin, cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.validate_matrix()

