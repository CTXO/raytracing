from __future__ import annotations
import numpy as np
import numpy.typing as npt

class Point:
    def __init__(self, p) -> None:
        self.p = np.array(p)

    def __str__(self):
        return f"Point({self.p[0], self.p[1], self.p[2]})"

    def add_vector(self, v: Vector) -> Point:
        return Point(self.p + v.v)
    
    def get_coordinates(self) -> [float]:
        return [self.p[0], self.p[1], self.p[2]]

    def __getitem__(self, item):
        return self.p[item]


class Vector:
    def __init__(self, v: [float] = None, v_np: npt.NDArray = None) -> None:
        if v is not None:
            self.v = np.array(v)
        elif v_np is not None:
            self.v = v_np
        else:
            raise ValueError("v or v_np is missing")

    def __str__(self):
        return f"Vector({self.v[0], self.v[1], self.v[2]})"

    @staticmethod
    def cross(v1: Vector, v2: Vector) -> Vector:
        return Vector(np.cross(v1.v, v2.v))
    
    @staticmethod
    def from_points(initial_p: Point, final_p: Point) -> Vector:
        vector_np = final_p.p - initial_p.p
        return Vector(v_np=vector_np)

    def normalize(self) -> Vector:
        magnitude = self.magnitude()
        if magnitude == 0:
            raise Exception("magnitude cannot be zero")
        return Vector(self.v / self.magnitude())

    def magnitude(self) -> int:
        return np.linalg.norm(self.v)
    
    def get_location(self) -> [float]:
        return [self.p[0], self.p[1], self.p[2]]
    
    def add_vector(self, v: Vector) -> Vector:
        return Vector(self.v + v.v)
    
    @staticmethod
    def dot(v1: Vector, v2: Vector) -> float:
        return np.dot(v1.v, v2.v)
    
    def __mul__(self, other: float) -> Vector:
        new_v = self.v * other
        return Vector(new_v)
    
    def __truediv__(self, other: float) -> Vector:
        return Vector(self.v / other)

    def __floordiv__(self, other: float) -> Vector:
        return Vector(self.v // other)

    def __neg__(self) -> Vector:
        return Vector(self.v * -1)

    def __getitem__(self, item):
        return self.v[item]


class BoundingBox:
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
