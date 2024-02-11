import colors
from objects import Plane
from objects import Sphere
from objects import TMesh
from objects import Triangle
from structures import Point
from structures import Vector
from transformations import Transalation

def sphere():
    sphere3 = Sphere(Point((1.5, 0, 3)), 1, colors.BLUE).transform(Transalation(0,3,0))
    return [sphere3]


def spheres_and_plane():
    sphere1 = Sphere(Point((0, 0, 8)), 1, colors.RED)
    sphere2 = Sphere(Point((0, 0, 12)), 3, colors.BLUE)
    plane = Plane(Point((0, 1, 0)), Vector((0, 1, 0)), colors.GREEN)
    return [sphere1, sphere2, plane]


def triangle():
    p1 = Point((0, 1, 3))
    p2 = Point((0.5, 0, 3))
    p3 = Point((-0.5, 0, 3))
    triangle_ret = Triangle((p3, p2, p1), colors.RED)
    return [triangle_ret]


def pentagon():
    vertices_pentagon=[
        Point((1, 0, 5)),
        Point((0, 1, 5)),
        Point((-1, 0, 5)),
        Point((-.5, -1, 5)),
        Point((.5, -1, 5))
    ]

    vertices_indexes_pentagon = [
        (0, 1, 4),
        (2, 3, 4),
        (1, 2, 4),
    ]
    pentagon_ret = TMesh(triangle_count=3, vertex_count=5, vertices=vertices_pentagon, vertices_indexes=vertices_indexes_pentagon,
                  colors=[colors.RED, colors.GREEN, colors.BLUE])
    return [pentagon_ret]


def star():
    vertices_star = [
        Point((0, 3, 5)),
        Point((-4, 1, 5)),
        Point((-3, -3, 5)),
        Point((3, -3, 5)),
        Point((4, 1, 5)),
        Point((1, 1, 5)),
        Point((2, -1, 5)),
        Point((0, -2, 5)),
        Point((-2, -1, 5)),
        Point((-1, 1, 5)),
    ]

    indexes_star = [
        (0, 5, 9),
        (4, 5, 6),
        (1, 8, 9),
        (2, 7, 8),
        (3, 6, 7),
        (5, 6, 7),
        (7, 8, 9),
        (5, 7, 9),
    ]

    star_ret = TMesh(triangle_count=8, vertex_count=10, vertices=vertices_star, vertices_indexes=indexes_star,
                 colors=[colors.BLUE] * 8)
    return [star_ret]


def face():
    head = Sphere(center=Point((0, 0, 20)), radius=10, color=colors.RED)
    left_eye = Sphere(center=Point((2,1,10)), radius=1, color=colors.BLUE)
    right_eye = Sphere(center=Point((-2,1,10)), radius=1, color=colors.BLUE)

    vertices_mouth = [
        Point((1.5, -2, 10)),
        Point((-1.5, -2, 10)),
        Point((1.5, -3, 10)),
        Point((-1.5, -3, 10)),
    ]

    vertices_index_mouth = [
        (0, 1, 3),
        (0, 2, 3),
    ]

    mouth = TMesh(triangle_count=2, vertex_count=4, vertices=vertices_mouth, vertices_indexes=vertices_index_mouth,
                  colors=[colors.GREEN]*2)

    return [head, left_eye, right_eye, mouth]


def angry_face():
    vertices_l_brow = [
        Point((1, 2.3, 10)),
        Point((0.1, 2.8, 10)),
        Point((1.5, 3.5, 10)),
        Point((2.5, 3, 10)),
    ]

    vertices_index_l_brow = [
        (1, 3, 0),
        (1, 3, 2)
    ]

    l_brow = TMesh(triangle_count=2, vertex_count=4, vertices=vertices_l_brow, vertices_indexes=vertices_index_l_brow,
                   colors=[colors.BLUE]*2)


    vertices_r_brow = [
        Point((-1, 2.3, 10)),
        Point((-0.1, 2.8, 10)),
        Point((-1.5, 3.5, 10)),
        Point((-2.5, 3, 10)),
    ]

    vertices_index_r_brow = [
        (1, 3, 0),
        (1, 3, 2)
    ]

    r_brow = TMesh(triangle_count=2, vertex_count=4, vertices=vertices_r_brow, vertices_indexes=vertices_index_r_brow,
                   colors=[colors.BLUE]*2)
    angry_face = face()
    angry_face.extend([r_brow, l_brow])
    return angry_face


def pyramid():
    p0 = Point((100, 0, 0))
    p1 = Point((0, 100, 0))
    p2 = Point((-100, 0, 0))
    p3 = Point((0, -100, 0))
    p4 = Point((0, 0, 100))

    t_count = 4

    vertices = [p0, p1, p2, p3, p4]

    indexes = [
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4)
    ]

    color_list = [
        colors.RED,
        colors.GREEN,
        colors.BLUE,
        colors.GREEN
    ]

    t_mesh = TMesh(triangle_count=t_count, vertex_count=5, vertices=vertices, vertices_indexes=indexes, colors=color_list)
    return [t_mesh]





