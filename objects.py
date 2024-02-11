from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List, Tuple
from structures import Vector, Point
from scene import Ray

import numpy as np
import numpy.typing as npt

from transformations import Transformation


class ScreenObject(ABC):
    @abstractmethod
    def intersect(self, ray: Ray) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, transf: Transformation) -> ScreenObject:
        raise NotImplementedError


class Sphere(ScreenObject):
    def __init__(self, center: Point, radius: int, color: npt.ArrayLike):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray: Ray) -> dict:
        oc = Vector.from_points(self.center, ray.origin)
        a = Vector.dot(ray.direction, ray.direction)
        b = Vector.dot(oc, ray.direction) * 2.0
        c = Vector.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            return {'t': min(t1, t2), 'color': self.color}
        else:
            return {}
    
    def transform(self, transf: Transformation) -> ScreenObject:
        self.center = transf.transform_point(self.center)
        return self


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
                return {'t': t, 'color': self.color}
        else:
            return {}
        
    def transform(self, transf: Transformation) -> ScreenObject:
        self.normal = transf.transform_vector(self.normal)
        self.point = transf.transform_point(self.point)
        return self
    
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
        try:
            self.normal = normal.normalize()
        except:
            self.normal = self.raw_normal

    def area(self):
        return self.raw_normal.magnitude() / 2

    def intersect(self, ray: Ray) -> dict:
        triangle_plane = Plane(self.points[0], self.normal,
                               self.color)  # Make plane and other objects accept Vector instead of [float]
        plane_intersect_t = triangle_plane.intersect(ray).get('t')
        if not plane_intersect_t or plane_intersect_t < 0:
            return {}

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
        return {}
    
    def transform(self, transf: Transformation) -> ScreenObject:
        transformed_points = []
        for p in self.points:
            transformed_points.append(transf.transform_point(p))
        self.points = transformed_points
        self.normal = transf.transform_vector(self.normal)
        return self

    def __str__(self):
        return f"""
        p1: {self.points[0]}
        p2: {self.points[1]}
        p3: {self.points[2]}
        normal: {self.normal}
        """


class TMesh(ScreenObject):
    triangles: List[Triangle]
    triangle_normals: List[Vector]

    def __init__(self, triangle_count: int, vertex_count: int, vertices: List[Point],
                 vertices_indexes: List[Tuple[int, int, int]], colors: List[npt.ArrayLike],
                 triangle_normals: List[Vector] = None, vertices_normals: List[Vector] = None):
        self.triangle_count = triangle_count
        self.vertex_count = vertex_count

        if len(vertices) != self.vertex_count:
            raise ValueError(f"Expected {self.vertex_count} vertices. Found {len(vertices)} instead.")
        self.vertices = vertices

        if len(vertices_indexes) != self.triangle_count:
            raise ValueError(f"Expected {self.triangle_count} vertices_indexes. Found {len(vertices_indexes)} instead.")
        self.vertices_indexes = vertices_indexes

        if len(colors) != self.triangle_count:
            raise ValueError(f"Expected {self.triangle_count} colors. Found {len(colors)} instead.")
        self.colors = colors

        self.triangles = []
        if triangle_normals is None:
            triangle_normals = []
            for t_i in range(triangle_count):
                indexes = vertices_indexes[t_i]
                points = [self.vertices[indexes[0]], self.vertices[indexes[1]], self.vertices[indexes[2]]]
                triangle = Triangle(points, self.colors[t_i])
                self.triangles.append(triangle)
                triangle_normals.append(triangle.normal)

        if len(triangle_normals) != self.triangle_count:
            raise ValueError(f"Expected {self.triangle_count} triangle_normals. Found {len(triangle_normals)} instead.")

        self.triangle_normals = triangle_normals

        if vertices_normals is None:
            pass
        elif len(vertices_normals) != self.vertex_count:
            raise ValueError(f"Expected {self.vertex_count} vertices_normals. Found {len(vertices_normals)} instead.")

        self.vertices_normals = vertices_normals

    def intersect(self, ray: Ray) -> dict:
        intersect_triangle = None
        triangle_id = None
        for i, triangle in enumerate(self.triangles):
            intersect_triangle = triangle.intersect(ray)
            if intersect_triangle.get('t'):
                triangle_id = i
                break
        if intersect_triangle.get('t'):
            intersect_triangle.update({'triangle_id': triangle_id})
            return intersect_triangle

        return {}
    

    def transform(self, transf: Transformation) -> ScreenObject:
        transformed_triangles = []
        for triangle in self.triangles:
            transformed_triangles.append(triangle.transform(transf))
        
        self.triangles = transformed_triangles
        
        if self.triangle_normals:
            tranformed_normals = []
            for normal in self.triangle_normals:
                tranformed_normals.append(transf.transform_vector(normal))
            
            self.triangle_normals = tranformed_normals
        
        if self.vertices_normals:
            transformed_vertices_normals = []
            for vertice_normal in self.vertices_normals:
                transformed_vertices_normals.append(transf.transform_vector(vertice_normal))
            self.vertices_normals = transformed_vertices_normals
        return self
            
