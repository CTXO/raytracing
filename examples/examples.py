import colors
from objects import Octree, OctreeNode, Plane
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

points_octree = {
    'origin_point': Point((0, 1.5, -5)),
    'target_point': Point((0, 1.5, -4.5)),
    'up_vector': Vector((0, 1, 0)),
}


points_octree_above = {
    'origin_point': Point((0, 11, 5)),
    'target_point': Point((0, 10, 5)),
    'up_vector': Vector((1, 0, 0)),
}

points_close_diagonal = {
    'origin_point': Point((0, 4, 0)),
    'target_point': Point((0, 3.5, 0.5)),
    'up_vector': Vector((0, 1, 0)),
}
points_close_diagonal_down = {
    'origin_point': Point((0, -4, 0)),
    'target_point': Point((0, -3.5, 0.5)),
    'up_vector': Vector((0, 1, 0)),
}
points_realistic = {
    'origin_point': Point((-250, 300, 300)),
    'target_point': Point((-249, 299, 299)),
    'up_vector': Vector((0, 100, 0)),
}

point_above_circle = Point((0, 3, 3))
point_diagonal_to_circle = Point((0, 1, 1))

def get_camera(origin_point=None, target_point=None, up_vector=None, lights=None, show_bb=False, show_octree=False, res=400, screen_size=800):
    return Camera(origin_point, target_point, up_vector, Screen(h_res=res, v_res=res, screen_size=screen_size), lights=lights, show_bb=show_bb, show_octree=show_octree)


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
    sphere_translation = Translation(0, -2, 0)
    objs = [sphere1.transform(sphere_translation), sphere2, plane.transform(rotationZ)]
    c.render_from_file(load_file='./examples/spheres_and_plane_rotated_and_translated.npy')
    # c.render(objs, save_file='./examples/spheres_and_plane_rotated_and_translated.npy')


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
    rotationX = RotationY(45)
    triangle_ret = Triangle((p3, p2, p1), colors.RED)
    triangle_ret.set_coefficients(**params)
    triangle_ret.transform(rotationX)
    c = get_camera(**points_close, lights=[Light(Point([3, -3, 0]))])
    c.render_from_file(load_file='./examples/triangle_rotated.npy')
    # c.render([triangle_ret], save_file='./examples/triangle_rotated.npy')


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

    c.render([star_ret], save_file='./examples/star_original.npy')


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

    params = {
        'k_diffusion': 1,
        'k_ambient': 0.1,
        'k_specular': 0.9,
        'shininess': 10
    }

    color_list = [
        colors.RED,
        colors.GREEN,
        colors.BLUE,
        colors.GREEN
    ]

    c = get_camera(**points_realistic, lights=[Light(Point([0, 500, 0]))])


    t_mesh = TMesh(triangle_count=t_count, vertex_count=5, vertices=vertices, vertices_indexes=indexes, colors=color_list)
    t_mesh.set_coefficients(**params)
    c.render([t_mesh])
    return [t_mesh]


def simple_scenario():
    sphere1 = Sphere(Point([10,0,5]), 2, colors.RED)
    sphere2 = Sphere(Point([-10,0,5]), 2, colors.BLUE)
    sphere3 = Sphere(Point([-1.5,4,5]), 2, colors.BLUE)
    sphere4 = Sphere(Point([-2,4,5]), 2, colors.RED)
    plane = Plane(Point([0,0,0]), Vector([0,1,0]), colors.GREEN)

    pyramid_points = [
        Point([1, 0, 1]),
        Point([1, 0, 3]),
        Point([-1, 0, 1]),
        Point([-1, 0, 3]),
        Point([0, 2, 2]),
    ]

    indexes = [
        (0, 1, 4),
        (1, 3, 4),
        (3, 2, 4),
        (2, 0, 4),
        (0, 1, 3),
        (1, 3, 2),
    ]

    color_list = [
        colors.BLUE,
        colors.YELLOW,
        colors.RED,
        colors.CYAN,
        colors.BLUE,
        colors.RED
    ]
    pyramid = TMesh(triangle_count=len(indexes), vertex_count=len(pyramid_points), vertices=pyramid_points,
                    vertices_indexes=indexes, colors=color_list)

    params = {
        'k_diffusion': 0.6,
        'k_ambient': 0.1,
        'k_specular': 0.1,
        'shininess': 10,
    }

    sphere1.set_coefficients(k_diffusion=0.8, k_specular=0.3, k_ambient=0.1, shininess=10)
    sphere2.set_coefficients(k_diffusion=0.8, k_specular=0.3, k_ambient=0.1, shininess=10)
    sphere3.set_coefficients(k_diffusion=0.8, k_specular=0.3, k_ambient=0.1, shininess=10)
    sphere4.set_coefficients(k_diffusion=0.8, k_specular=0.3, k_ambient=0.1, shininess=10)
    plane.set_coefficients(**params, k_refraction=1, n_refraction=1.5)

    pyramid.set_coefficients(**params)

    c = get_camera(**points_close_diagonal, lights=[Light(Point([0, 5, 0])), Light(Point([0,3,6])), Light(Point([0,5, 5]))], show_octree=False, res=300, screen_size=300)

    # c.render_from_file(load_file='./examples/scenario-bounding-boxes-cubed.npy')
    # c.render([sphere1, pyramid, sphere2], save_file='./examples/scenario-bounding-boxes-octree-recursive-front.npy')
    c.render([pyramid], save_file='./examples/debug.npy')



def bounding_box():
    points = {
        'origin_point': Point([-0.5, 0.5, -0.5]),
        'target_point': Point([0, 1, 0]),
        'up_vector': Vector([0, 1, 0]),
    }

    points_parallel = {
        'origin_point': Point([4, 4, 0]),
        'target_point': Point([4, 4, 0.5]),
        'up_vector': Vector([0, 1, 0]),
    }

    params = {
        'k_diffusion': 0.6,
        'k_ambient': 0.1,
        'k_specular': 0.1,
        'shininess': 10,
    }

    c = get_camera(**points, lights=[Light(Point([0, 0, 5]))], show_octree=True)
    sphere = Sphere(center=Point([4,4,4]), radius=1, color=colors.RED)
    oct = Octree(objs=[sphere])
    sphere.set_coefficients(**params, k_reflection=1)
    c.render([oct], save_file='./examples/octree-test.npy')


def bunch_of_spheres():
    c = get_camera(**points_realistic, lights=[Light(Point([0, 500, 0])), Light(Point([200, 100, 200]))], show_octree=False)
    spheres = []
    import random

    for x in range(10):
        for z in range(10):
            for y in range(10):
                if x in [0, 9] or z in [0, 9] or y == 0:
                    base = 50
                    spheres.append(Sphere(Point([base + x*10, y*10, base + z*10]), 5, random.choice([colors.RED])))
            

    for sphere in spheres:
        sphere.set_coefficients(k_diffusion=0.6, k_ambient=0.1, k_specular=0.1, shininess=10)

    # c.render_from_file(load_file='./examples/bunch_of_spheres_slow.npy')
    c.render(spheres, save_file='./examples/bunch_of_spheres_slow.npy')


def create_tmesh_from_obj(obj_file_path):
    # Load the mesh from the OBJ file
    vertices = []
    faces = []
    with open(obj_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                faces.append(face)

    # Create a list of Point objects from the vertices
    points = [Point(vertex) for vertex in vertices]

    # Create a list of indexes for the faces
    indexes = [(face[0], face[1], face[2]) for face in faces]

    # Assuming you have a function to get colors based on the number of vertices
    color_list = [colors.RED for i in range(len(faces))]


    # Create a TMesh object using the extracted data
    t_mesh = TMesh(triangle_count=len(faces), vertex_count=len(vertices), vertices=points, vertices_indexes=indexes, colors=color_list)

    return t_mesh

def teapot():
    params = {
        'k_diffusion': 0.6,
        'k_ambient': 0.1,
        'k_specular': 0.1,
        'shininess': 10,
    }
    t_mesh = create_tmesh_from_obj('./examples/objs/teapot.obj')
    t_mesh.set_coefficients(**params)
    c = get_camera(**points_octree, lights=[Light(Point([0, 500, 0]))])
    c.render([t_mesh], save_file='./examples/teapot.npy')




