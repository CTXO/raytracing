from abc import ABC, abstractmethod
import math
import numpy as np
import numpy.typing as npt

from structures import Point, Vector

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
    
    def transform_np_array(self, np_array: npt.ArrayLike) -> npt.ArrayLike:
        if np_array.shape != (3,):
            return ValueError(f"Np array should have 3 coordinates points, it has shape {np_array.shape} instead. Array:\n{np_array}")
        np_array_4d = np.append(np_array, 1)
        transformed_array = self.matrix @ np_array_4d
        transformed_array_3d = transformed_array[:-1]
        return transformed_array_3d
    
    def transform_point(self, point: Point) -> Point:
        point_array_transformed = self.transform_np_array(point.p)
        return Point(point_array_transformed)
    
    def transform_vector(self, vector: Vector) -> Vector:
        vector_array_transformed = self.transform_np_array(vector.v)
        return Vector(vector_array_transformed)
        
        

        



class Translation(Transformation):
    def __init__(self, x, y, z) -> None:
        self.matrix = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        self.validate_matrix()


class RotationX(Transformation):
    def __init__(self, angle) -> None:
        sin, cos = self.get_sin_cos(angle)
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, cos, -sin, 0],
            [0, sin, cos, 0],
            [0, 0, 0, 1]
        ])
        self.validate_matrix()
        

class RotationY(Transformation):
    def __init__(self, angle) -> None:
        sin, cos = self.get_sin_cos(angle)
        self.matrix = np.array([
            [cos, 0, sin, 0],
            [0, 1, 0, 0],
            [-sin, 0, cos, 0],
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

