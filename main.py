from examples.examples import *
from scene import Camera, Light, Screen
from structures import Point
from structures import Vector


# Camera for simple objects
origin_point = Point((0, 0, 0))
target_point = Point((0, 0, 0.5))
up_vector = Vector((0,1,0))
l = Light(Point([0,3,3]), [255,255,255])

# c1 = Camera(origin_point, target_point, up_vector, Screen(), lights=[l])
# simple_scenario()
# spheres_and_plane()
# More realistic camera
initial_p = Point((-300, 300, 300))
target_p = Point((-299, 299, 299))
normal = Vector((0, 100, 0))

p0 = Point([0,0,0])
p1 = Point([1,1,1])
p2 = Point([2,2,2])
p3 = Point([3,3,3])
p4 = Point([4,4,4])
p5 = Point([5,5,5])
# ni = OctreeNode(min_point=p2, max_point=p4)
# no = OctreeNode(min_point=p1, max_point=p4)
# if no.contains(ni.box):
#     print('should be true')
# else:
#     print('should not be false')

simple_scenario()
# bunch_of_spheres()

# c2 = Camera(initial_p=initial_p, target_p=target_p, up_input_v=normal, scene=Screen())
# c2.render(pyramid())
# c2.render_from_file(load_file='./examples/pyramid-far.npy')
