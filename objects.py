from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List, Tuple
from structures import Vector, Point
from scene import Ray

import numpy as np
import numpy.typing as npt


class ScreenObject(ABC):
    @abstractmethod
    def intersect(self, ray: Ray) -> dict:
        pass


class Sphere(ScreenObject):
    def __init__(self, center: Point, radius: int, color: npt.ArrayLike):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray: Ray) -> dict:
        oc = Vector.from_points(self.center, ray.origin)
        a = Vector.dot(ray.direction, ray.direction)
        b = Vector.dot(oc, ray.direction) * 2.0
        c = Vector.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            return {'t': min(t1, t2)}
        else:
            return {}


class Plane(ScreenObject):
    def __init__(self, point: Point, normal: Vector, color: npt.ArrayLike):
        self.point = point
        self.normal = normal.normalize()
        self.color = color

    def intersect(self, ray: Ray) -> dict:
        denom = Vector.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  
            v = Vector.from_points(ray.origin, self.point)
            t = Vector.dot(v, self.normal) / denom

            if t <= 0:
                return {}
            else:
                return {'t': t}
        else:
            return {}


class Triangle(ScreenObject):
    def __init__(self, points: Tuple[Point, Point, Point], color: npt.ArrayLike,
                 normal: Vector = None):
        self.points = points
        self.color = np.array(color)

        v1 = Vector.from_points(self.points[0], self.points[1])
        v2 = Vector.from_points(self.points[0], self.points[2])
        if normal is None:
            normal = Vector.cross(v1, v2)

        self.raw_normal = normal
        self.normal = normal.normalize()

    def area(self):
        return self.raw_normal.magnitude() / 2
    
    def intersect(self, ray: Ray) -> dict:
        triangle_plane = Plane(self.points[0], self.normal, self.color)  # Make plane and other objects accept Vector instead of [float]
        plane_intersect_t = triangle_plane.intersect(ray).get('t')
        if not plane_intersect_t or plane_intersect_t < 0:
            return None
        
        point_intersect = ray.origin.add_vector(ray.direction * plane_intersect_t)
        
        t1 = Triangle(points=[point_intersect, self.points[0], self.points[1]], color=self.color)
        t2 = Triangle(points=[point_intersect, self.points[0], self.points[2]], color=self.color)
        t3 = Triangle(points=[point_intersect, self.points[1], self.points[2]], color=self.color)
        
        at1 = t1.area()
        at2 = t2.area()
        at3 = t3.area()
        at = self.area()
        
        alpha = at1 / at
        beta = at2 / at
        gamma = at3 / at
        
        if abs(alpha + beta + gamma - 1) > 1e-6:
            return {}
        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return {"t": plane_intersect_t, "normal": self.normal, "color": self.color}


class TMesh:
    def __init__(self, triangle_count: int, vertex_count: int, vertices: List[Point],
                  vertices_indexes: List[Tuple[int, int, int]], colors: List[Tuple[float, float, float]],
                  triangle_normals: List[Vector] = None, vertices_normals: List[Vector] = None):
        self.triangle_count = triangle_count
        self.vertex_count = vertex_count
        
        if len(vertices) != self.vertex_count:
            raise ValueError(f"Expected {self.vertex_count} vertices. Found {len(vertices)} instead.")
        self.vertices = vertices
        
        if len(vertices_indexes) != self.triangle_count:
            raise(f"Expected {self.triangle_count} vertices_indexes. Found {len(vertices_indexes)} instead.")
        self.vertices_indexes = vertices_indexes

        if len(colors) != self.triangle_count:
            raise(f"Expected {self.triangle_count} colors. Found {len(colors)} instead.")
        self.colors = colors
        
        if triangle_normals is None:
            # Todo
            pass
        elif len(triangle_normals) != self.triangle_count:
            raise(f"Expected {self.triangle_count} triangle_normals. Found {len(triangle_normals)} instead.")
        self.triangle_normals = triangle_normals
        
        if vertices_normals is None:
            # Todo
            pass
        elif len(vertices_normals) != self.vertex_count:
            raise(f"Expected {self.vertex_count} vertices_normals. Found {len(vertices_normals)} instead.")
        self.vertices_normals = vertices_normals
        
        
                



