from structures import Vector, Point
from scene import Ray

import numpy as np

class Sphere:
    def __init__(self, center: [float], radius: int, color: [float]):
        self.center = Point(center)
        self.radius = radius
        self.color = np.array(color)

    def intersect(self, ray: Ray):
        oc = Vector.from_points(self.center, ray.origin)
        a = Vector.dot(ray.direction, ray.direction)
        b = Vector.dot(oc, ray.direction) * 2.0
        c = Vector.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            return min(t1, t2)
        else:
            return None
        
class Plane:
    def __init__(self, point: [float], normal: [float], color: [float]):
        self.point = Point(point)
        self.normal = Vector(normal).normalize()
        self.color = np.array(color)

    def intersect(self, ray: Ray):
        denom = Vector.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  
            v = Vector.from_points(ray.origin, self.point)
            t = Vector.dot(v, self.normal) / denom
            return t if t > 0 else None
        else:
            return None
