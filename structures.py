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
        t = self.intersect_edges(ray.origin, ray.direction)
        return {'t': t}

    def compute_intersection_point(self, ray_origin, ray_direction, edge_start, edge_end):
        t = ((edge_start[0] - ray_origin[0]) * (edge_end[0] - edge_start[0]) +
             (edge_start[1] - ray_origin[1]) * (edge_end[1] - edge_start[1]) +
             (edge_start[2] - ray_origin[2]) * (edge_end[2] - edge_start[2])) / \
            ((ray_direction[0] * (edge_end[0] - edge_start[0])) +
             (ray_direction[1] * (edge_end[1] - edge_start[1])) +
             (ray_direction[2] * (edge_end[2] - edge_start[2])))

        intersection_point = (ray_origin[0] + t * ray_direction[0],
                              ray_origin[1] + t * ray_direction[1],
                              ray_origin[2] + t * ray_direction[2])

        return intersection_point

    def compute_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    @staticmethod
    def intersect_segment(ray_origin, ray_direction, edge_start, edge_end) -> bool:

        epsilon = 1e-3
        e1 = edge_end[0] - edge_start[0]
        e2 = edge_end[1] - edge_start[1]
        e3 = edge_end[2] - edge_start[2]
        p = ray_direction[1] * e3 - ray_direction[2] * e2
        q = ray_direction[2] * (ray_origin[1] - edge_start[1]) - ray_direction[1] * (ray_origin[2] - edge_start[2])
        r = e2 * (ray_origin[2] - edge_start[2]) - e3 * (ray_origin[1] - edge_start[1])
        det = e1 * p + ray_direction[0] * (e2 * ray_direction[2] - e3 * ray_direction[1])

        if det == 0:
            return False

        t = (e1 * q + ray_direction[0] * r) / det
        if t < -epsilon or t > 1 + epsilon:
            return False

        u = (q + p * t) / det
        if u < 0 or u > 1:
            return False

        v = (e2 * r - e3 * q) / det
        if v < 0 or v > 1:
            return False

        return True


