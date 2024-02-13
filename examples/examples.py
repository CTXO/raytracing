import colors
from objects import Plane
from objects import Sphere
from objects import TMesh
from objects import Triangle
from scene import Camera
from scene import Light
from scene import Screen
from structures import Point
from structures import Vector
from transformations import RotationX, RotationY, RotationZ, Translation


points_close = {
    'origin_point': Point((0, 0, 0)),
    'target_point': Point((0, 0, 0.5)),
    'up_vector': Vector((0, 1, 0)),
}
points_close_diagonal = {
    'origin_point': Point((0, 4, 0)),
    'target_point': Point((0, 3.5, 0.5)),
    'up_vector': Vector((0, 1, 0)),
}
points_realistic = {
    'origin_point': Point((-300, 300, 300)),
    'target_point': Point((-299, 299, 299)),
    'up_vector': Point((0, 100, 0)),
}

point_above_circle = Point((0, 3, 3))
point_diagonal_to_circle = Point((0, 1, 1))

def get_camera(origin_point=None, target_point=None, up_vector=None, lights=None):
    return Camera(origin_point, target_point, up_vector, Screen(), lights=lights)


def sphere():
    translation = Translation(-1.5,0,0)
    rotationZ = RotationZ(180)
    rotationY = RotationY(-45)
    rotationX = RotationX(-45)
    sphere3 = Sphere(Point((1.5, 0, 3)), 1, colors.BLUE).transform(translation)
    sphere3.set_coefficients(k_diffusion=1, k_ambient=0, k_specular=0.6, shininess=10000)
    c = get_camera(**points_close_diagonal, lights=[Light(point_diagonal_to_circle)])
    c.render([sphere3])


def spheres_and_plane():
    sphere1 = Sphere(Point((0, 0, 8)), 1, colors.RED)
    sphere2 = Sphere(Point((0, 0, 12)), 3, colors.BLUE)
    plane = Plane(Point((0, 1, 0)), Vector((0, 1, 0)), colors.GREEN)

    params = {
        'k_diffusion': 1,
        'k_ambient': 0.1,
        'k_specular': 0.4,
        'shininess': 10
    }
    sphere1.set_coefficients(**params)
    sphere2.set_coefficients(**params)
    plane.set_coefficients(**params)

    point_above_plane = Point([0,4,0])
    point_bellow_plane = Point([4, -2, 6])
    c = get_camera(**points_close, lights=[Light(point_bellow_plane)])
    translation = Translation(2, 0, 0)
    rotationZ = RotationZ(90)
    objs = [sphere1, sphere2, plane]
    c.render(objs)


def triangle():

    params = {
        'k_diffusion': 1,
        'k_ambient': 0.1,
        'k_specular': 0.9,
        'shininess': 10
    }
    p1 = Point((0, 4, 3))
    p2 = Point((2, 0, 3))
    p3 = Point((-2, 0, 3))
    translation = Translation(1,0, 0)
    rotationZ = RotationZ(45)
    triangle_ret = Triangle((p3, p2, p1), colors.RED)
    triangle_ret.set_coefficients(**params)
    c = get_camera(**points_close, lights=[Light(Point([3, -3, 0]))])
    c.render([triangle_ret])


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
    rotationZ = RotationZ(180)
    translation = Translation(2,0,0)
    return [pentagon_ret.transform(rotationZ).transform(translation)]


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

    params = {
        'k_diffusion': 1,
        'k_ambient': 0.1,
        'k_specular': 0.2,
        'shininess': 1
    }

    star_ret = TMesh(triangle_count=8, vertex_count=10, vertices=vertices_star, vertices_indexes=indexes_star,
                 colors=[colors.BLUE] * 8)
    star_ret.set_coefficients(**params)
    c = get_camera(**points_close, lights=[Light(Point([-5, -3, 4]))])

    c.render([star_ret])


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





