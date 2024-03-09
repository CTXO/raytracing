from __future__ import annotations
import numpy as np
import numpy.typing as npt

import colors


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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Vector:
        return Vector(self.v / other)

    def __floordiv__(self, other: float) -> Vector:
        return Vector(self.v // other)

    def __rtruediv__(self, other) -> Vector:
        return Vector(other / self.v)

    def __rfloordiv__(self, other):
        return Vector(other // self.v)

    def __neg__(self) -> Vector:
        return Vector(self.v * -1)

    def __getitem__(self, item):
        return self.v[item]


class BoundingBox:

    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
        self.edges = self.compute_edges()
        self.color = colors.GREEN

    def compute_edges(self):
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point

        # Compute the 12 edges of the bounding box
        edges = [
            ((min_x, min_y, min_z), (max_x, min_y, min_z)),
            ((min_x, min_y, min_z), (min_x, max_y, min_z)),
            ((min_x, min_y, min_z), (min_x, min_y, max_z)),
            ((max_x, min_y, min_z), (max_x, max_y, min_z)),
            ((max_x, min_y, min_z), (max_x, min_y, max_z)),
            ((min_x, max_y, min_z), (max_x, max_y, min_z)),
            ((min_x, max_y, min_z), (min_x, max_y, max_z)),
            ((max_x, max_y, min_z), (max_x, max_y, max_z)),
            ((min_x, min_y, max_z), (max_x, min_y, max_z)),
            ((min_x, min_y, max_z), (min_x, max_y, max_z)),
            ((max_x, min_y, max_z), (max_x, max_y, max_z)),
            ((min_x, max_y, max_z), (max_x, max_y, max_z))
        ]
        return edges

    def intersect_edges(self, ray_origin, ray_direction):
        closest_intersection = None
        closest_distance = float('inf')
        for edge in self.edges:
            if self.intersect_segment(ray_origin, ray_direction, edge[0], edge[1]):
                intersection_point = self.compute_intersection_point(ray_origin, ray_direction, edge[0], edge[1])
                distance = self.compute_distance(ray_origin, intersection_point)
                if distance < closest_distance:
                    closest_intersection = edge
                    closest_distance = distance

        return closest_distance

    def intersect(self, ray):
        ray_inverse = 1 / ray.direction

        tx_min = (self.min_point[0] - ray.origin[0]) * ray_inverse[0] if ray.direction[0] != 0 else -inf
        tx_max = (self.max_point[0] - ray.origin[0]) * ray_inverse[0] if ray.direction[0] != 0 else inf
        ty_min = (self.min_point[1] - ray.origin[1]) * ray_inverse[1] if ray.direction[1] != 0 else -inf
        ty_max = (self.max_point[1] - ray.origin[1]) * ray_inverse[1] if ray.direction[1] != 0 else inf
        tz_min = (self.min_point[2] - ray.origin[2]) * ray_inverse[2] if ray.direction[2] != 0 else -inf
        tz_max = (self.max_point[2] - ray.origin[2]) * ray_inverse[2] if ray.direction[2] != 0 else inf

        if tx_min > tx_max:
            tx_min, tx_max = tx_max, tx_min
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        if max(tx_max, ty_max, tz_max) < 0:
            return {}

        if ray.direction[0] == 0 and ray.origin[0] < self.min_point[0] or ray.origin[0] > self.max_point[0]:
            return {}

        if ray.direction[1] == 0 and ray.origin[1] < self.min_point[1] or ray.origin[1] > self.max_point[1]:
            return {}

        if ray.direction[2] == 0 and ray.origin[2] < self.min_point[2] or ray.origin[2] > self.max_point[2]:
            return {}

        if tx_min > ty_max or ty_min > tx_max:
            return {}

        if ty_min > tz_max or tz_min > ty_max:
            return {}

        if tx_min > tz_max or tz_min > tx_max:
            return {}


        return {'t': max(tx_min, ty_min, tz_min)}