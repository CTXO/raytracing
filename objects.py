from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scene import Ray

from abc import ABC
from abc import abstractmethod
from typing import List, Tuple
from math import inf

from structures import BoundingBox, Vector, Point

import numpy as np
import numpy.typing as npt

import colors
from transformations import Transformation

from intersectable import IntersectableMixin


class ScreenObject(IntersectableMixin):
    k_diffusion: float = 0
    k_specular: float = 0
    k_ambient: float = 0
    k_reflection: float = 0
    k_refraction: float = 0
    n_refraction: float = 1
    shininess: float = 0
    color: npt.ArrayLike
    normal_always_positive = False
    debug_object = False
    bounding_box: BoundingBox = None
    real_object = True

    @abstractmethod
    def transform(self, transf: Transformation) -> ScreenObject:
        raise NotImplementedError
    
    @abstractmethod
    def normal_of(self, point: Point, **kwargs) -> Vector:
        raise NotImplementedError
    
    @abstractmethod
    def create_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    @staticmethod
    def validate_coefficient(coefficient, min_value=0, max_value=1):
        if min_value and coefficient < min_value:
            return min_value
        if max_value and coefficient > max_value:
            return max_value
        return coefficient

    @staticmethod
    def get_intersect_point(ray, t):
        vector = ray.direction * t
        return ray.origin.add_vector(vector)

    def set_coefficients(self, k_diffusion=0, k_specular=0, k_ambient=0, k_reflection=0, k_refraction=0, n_refraction=1, shininess=0):
        self.k_diffusion = self.validate_coefficient(k_diffusion)
        self.k_specular = self.validate_coefficient(k_specular)
        self.k_ambient = self.validate_coefficient(k_ambient)
        self.k_reflection = self.validate_coefficient(k_reflection)
        self.k_refraction = self.validate_coefficient(k_refraction)
        self.n_refraction = self.validate_coefficient(n_refraction, min_value=1, max_value=None)
        self.shininess = self.validate_coefficient(shininess)
        return self


class Sphere(ScreenObject):
    def __init__(self, center: Point, radius: int, color: npt.ArrayLike):
        self.center = center
        self.radius = radius
        self.color = color
        self.bounding_box = self.create_bounding_box()

    def intersect(self, ray: Ray) -> dict:
        oc = Vector.from_points(self.center, ray.origin)
        a = Vector.dot(ray.direction, ray.direction)
        b = Vector.dot(oc, ray.direction) * 2.0
        c = Vector.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            t = min(t1, t2)
            intersect_point = self.get_intersect_point(ray, t)
            return {'t': t, 'color': self.color, 'point': intersect_point}
        else:
            return {}
    
    def transform(self, transf: Transformation) -> ScreenObject:
        self.center = transf.transform_point(self.center)
        return self

    def normal_of(self, point: Point, **kwargs) -> Vector:
        return Vector.from_points(self.center, point)

    def create_bounding_box(self) -> BoundingBox:
        min_point = self.center.add_vector(Vector([-self.radius, -self.radius, -self.radius]))
        max_point = self.center.add_vector(Vector([self.radius, self.radius, self.radius]))
        return BoundingBox(min_point, max_point)
    

class Plane(ScreenObject):
    def __init__(self, point: Point, normal: Vector, color: npt.ArrayLike):
        self.point = point
        self.normal = normal.normalize()
        self.color = color
        self.normal_always_positive = True
        self.bounding_box = self.create_bounding_box()

    def intersect(self, ray: Ray) -> dict:
        denom = Vector.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:
            v = Vector.from_points(ray.origin, self.point)
            t = Vector.dot(v, self.normal) / denom

            if t <= 1e-6:
                return {}
            else:
                return {'t': t, 'color': self.color, 'point': self.get_intersect_point(ray, t)}
        else:
            return {}
        
    def transform(self, transf: Transformation) -> ScreenObject:
        self.normal = transf.transform_vector(self.normal)
        self.point = transf.transform_point(self.point)
        return self

    def normal_of(self, point: Point, **kwargs) -> Vector:
        return self.normal
    
    def create_bounding_box(self) -> BoundingBox:
        return BoundingBox(min_point=Point([0,0,0]), max_point=Point([0,0,0])) # Todo


class Triangle(ScreenObject):
    def __init__(self, points: Tuple[Point, Point, Point], color: npt.ArrayLike,
                 normal: Vector = None):
        self.points: Tuple[Point, Point, Point] = points
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
        self.normal_always_positive = True

        self.bounding_box = self.create_bounding_box()

    def area(self):
        return self.raw_normal.magnitude() / 2

    def get_baricentric_coordinates(self, point: Point):
        t1 = Triangle(points=[point, self.points[0], self.points[1]], color=self.color)
        t2 = Triangle(points=[point, self.points[0], self.points[2]], color=self.color)
        t3 = Triangle(points=[point, self.points[1], self.points[2]], color=self.color)

        at1 = t1.area()
        at2 = t2.area()
        at3 = t3.area()
        at = self.area()

        alpha = at1 / at
        beta = at2 / at
        gamma = at3 / at

        return alpha, beta, gamma


    def intersect(self, ray: Ray) -> dict:
        triangle_plane = Plane(self.points[0], self.normal,
                               self.color)  # Make plane and other objects accept Vector instead of [float]
        plane_intersect_t = triangle_plane.intersect(ray).get('t')
        if not plane_intersect_t or plane_intersect_t < 0:
            return {}

        point_intersect = self.get_intersect_point(ray, plane_intersect_t)

        alpha, beta, gamma = self.get_baricentric_coordinates(point_intersect)

        if abs(alpha + beta + gamma - 1) > 1e-6:
            return {}
        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return {"t": plane_intersect_t, "normal": self.normal, "color": self.color, "point": point_intersect}
        return {}
    
    def transform(self, transf: Transformation) -> ScreenObject:
        transformed_points = []
        for p in self.points:
            transformed_points.append(transf.transform_point(p))
        self.points = transformed_points
        self.normal = transf.transform_vector(self.normal)
        return self

    def normal_of(self, point: Point, **kwargs) -> Vector:
        return self.normal

    def create_bounding_box(self) -> BoundingBox:
        min_point = Point([min(self.points[0][0], self.points[1][0], self.points[2][0]),
                           min(self.points[0][1], self.points[1][1], self.points[2][1]),
                           min(self.points[0][2], self.points[1][2], self.points[2][2])])
        max_point = Point([max(self.points[0][0], self.points[1][0], self.points[2][0]),
                           max(self.points[0][1], self.points[1][1], self.points[2][1]),
                           max(self.points[0][2], self.points[1][2], self.points[2][2])])
        return BoundingBox(min_point, max_point)

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
    normal_always_positive = True

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
            vertices_normals = []
            for i, vertex in enumerate(self.vertices): # vertex is a point
                vertex_normal = Vector([0, 0, 0])
                for triangle_tuple in vertices_indexes: # triangle_tuple is a tuple of 3 indexes
                    if i in triangle_tuple:
                        triangle_index = vertices_indexes.index(triangle_tuple)
                        triangle = self.triangles[triangle_index]
                        vertex_normal = vertex_normal.add_vector(triangle.normal)
                vertices_normals.append(vertex_normal.normalize())

        elif len(vertices_normals) != self.vertex_count:
            raise ValueError(f"Expected {self.vertex_count} vertices_normals. Found {len(vertices_normals)} instead.")

        self.vertices_normals = vertices_normals

        self.bounding_box = self.create_bounding_box()

    def set_coefficients(self, k_diffusion=0, k_specular=0, k_ambient=0, k_reflection=0, k_refraction=0, n_refraction=1, shininess=0):
        super().set_coefficients(k_diffusion, k_specular, k_ambient, k_reflection, k_refraction, n_refraction, shininess)
        for triangle in self.triangles:
            triangle.set_coefficients(k_diffusion, k_specular, k_ambient, k_reflection, k_refraction, n_refraction, shininess)
        return self

    def intersect(self, ray: Ray) -> dict:
        intersect_triangle = None
        triangle_id = None
        min_t = float('inf')
        for i, triangle in enumerate(self.triangles):
            intersect_temp = triangle.intersect(ray)
            t_temp = intersect_temp.get('t')
            if t_temp and t_temp < min_t:
                triangle_id = i
                min_t = t_temp
                intersect_triangle = intersect_temp

        if intersect_triangle and intersect_triangle.get('t'):
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

    def normal_of(self, point: Point, **kwargs) -> Vector:
        triangle_id = kwargs.get('triangle_id')
        if triangle_id is None:
            raise ValueError("Expected triangle_id")
        triangle = self.triangles[triangle_id]
        triangle_vertices_ids = self.vertices_indexes[triangle_id]
        vertices_normal = []
        for triangle_vertice_id in triangle_vertices_ids:
            vertices_normal.append(self.vertices_normals[triangle_vertice_id])

        alpha, beta, gamma = triangle.get_baricentric_coordinates(point)
        normal = (vertices_normal[0] * alpha).add_vector(vertices_normal[1] * beta).add_vector(vertices_normal[2] * gamma)
        return normal

    def create_bounding_box(self) -> BoundingBox:
        min_point = Point([inf, inf, inf])
        max_point = Point([-inf, -inf, -inf])
        for vertex in self.vertices:
            for i in range(3):
                if vertex[i] < min_point[i]:
                    min_point[i] = vertex[i]
                if vertex[i] > max_point[i]:
                    max_point[i] = vertex[i]
        return BoundingBox(min_point, max_point)


class OctreeNode(IntersectableMixin):
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
        self.box = BoundingBox(min_point=min_point, max_point=max_point)
        self.children = []

    def create_children(self):
        mid_point = []
        for i in range(3):
            min_coord = self.min_point[i] +  + (self.max_point[i] - self.min_point[i]) / 2
            mid_point.append(min_coord)

        children_bounds = [
                [self.min_point, mid_point],
                [Point([mid_point[0], self.min_point[1], self.min_point[2]]), Point([self.max_point[0], mid_point[1], mid_point[2]])], 
                [Point([self.min_point[0], mid_point[1], self.min_point[2]]), Point([mid_point[0], self.max_point[1], mid_point[2]])],
                [Point([self.min_point[0], self.min_point[1], mid_point[2]]), Point([mid_point[0], mid_point[1], self.max_point[2]])],

                [Point([self.min_point[0], mid_point[1], mid_point[2]]), Point([mid_point[0], self.max_point[1], self.max_point[2]])],
                [Point([mid_point[0], self.min_point[1], mid_point[2]]), Point([self.max_point[0], mid_point[1], self.max_point[2]])],
                [Point([mid_point[0], mid_point[1], self.min_point[2]]), Point([self.max_point[0], self.max_point[1], mid_point[2]])],
                [mid_point, self.max_point],
        ]

        self.children = [OctreeNode(min_point=bound[0], max_point=bound[1]) for bound in children_bounds]

    def intersect(self, ray: Ray) -> dict:
        min_t = float('inf')
        if self.children:
            for children in self.children:
                intersect = children.intersect(ray)
                t = intersect.get('t')
                if t and t < min_t:
                    min_t = t
        else:
            return self.box.intersect(ray)
        if min_t == float('inf'):
            return {}
        return {'t': min_t}

class Octree(IntersectableMixin):
    def __init__(self, objs: List[ScreenObject]) -> None:
        """Creates an Octree object

        Args:
            node (OctreeNode): The root node of the octree
            debug (int, optional): Debug level. If 0, the octree will not be drawn.
                If 1, the octree's edges will be drawn.
                If 2, the whole octree will be drawn as a solyd.
                Lighting will not be applied to the octree.
        """

        min_point = Point([inf, inf, inf])
        max_point = Point([-inf, -inf, -inf])
        objs = list(filter(lambda obj: not isinstance(obj, Plane), objs)) # remove plan objects for now
        for obj in objs:
            if not obj.bounding_box:
                raise ValueError("Bounding box not found in object")
            for i in range(3):
                if obj.bounding_box.min_point[i] < min_point[i]:
                    min_point[i] = obj.bounding_box.min_point[i]
                if obj.bounding_box.max_point[i] > max_point[i]:
                    max_point[i] = obj.bounding_box.max_point[i]

        x_dif = max_point[0] - min_point[0]
        y_dif = max_point[1] - min_point[1]
        z_dif = max_point[2] - min_point[2]

        max_dif = max(x_dif, y_dif, z_dif)

        max_point[0] = min_point[0] + max_dif 
        max_point[1] = min_point[1] + max_dif
        max_point[2] = min_point[2] + max_dif

        self.root = OctreeNode(min_point, max_point)
        self.root.create_children()
        self.root.children[0].create_children()
        self.color = colors.GREEN

    def intersect(self, ray: Ray) -> dict:
        return self.root.intersect(ray)




