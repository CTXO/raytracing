from examples.examples import *
from scene import Camera, Screen
from structures import Point
from structures import Vector


# Camera for simple objects
origin_point = Point((0, 0, 0))
target_point = Point((0, 0, 0.5))
up_vector = Vector((0,1,0))
c1 = Camera(origin_point, target_point, up_vector, Screen())
c1.render(star())


# More realistic camera
initial_p = Point((-300, 300, 300))
target_p = Point((-299, 299, 299))
normal = Vector((0, 100, 0))

c2 = Camera(initial_p=initial_p, target_p=target_p, up_input_v=normal, scene=Screen())
# c2.render(pyramid())
# c2.render_from_file(load_file='./examples/pyramid-far.npy')
