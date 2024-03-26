from __future__ import annotations
from typing import TYPE_CHECKING

from objects import Octree
if TYPE_CHECKING:
    from objects import ScreenObject
    from structures import BoundingBox

import time
from math import sqrt
from typing import List

from structures import Vector, Point
import numpy as np
import cv2


class Ray:
    def __init__(self, origin: Point, direction: Vector):
        self.origin = origin
        self.direction = direction.normalize()


class Screen:
    def __init__(self, h_res=400, v_res=400, screen_size=800) -> None:
        self.h_res = h_res
        self.v_res = v_res
        self.screen_size = screen_size

        self.pixel_size_h = 2 / self.h_res
        self.pixel_size_v = 2 / self.v_res

        self.real_pixel_size_h = screen_size // self.h_res
        self.real_pixel_size_v = screen_size // self.v_res

        self.real_grid = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)
    
    
    def draw_pixel(self, x, y, rgb):
        self.real_grid[y*self.real_pixel_size_v: (y+1)*self.real_pixel_size_v, x*self.real_pixel_size_h: (x+1)*self.real_pixel_size_h] = rgb


class Light:
    def __init__(self, point: Point, color: List[int] = None):
        if color is None:
            color = [255, 255, 255]
        self.point = point
        self.color = np.array(color)
    
    
class Camera:
    ambient_light = np.array([255,255,255])
    
    def __init__(self, initial_p: Point, target_p: Point, up_input_v: Vector, scene: Screen, lights: List[Light]):
        """
        Class that represents the camera that will render the scene.

        Args:
            initial_p: Initial position of the camera.
            target_p: Position where the camera is looking at.
            up_input_v: Vector that represents the up direction of the camera.
            scene: Scene object that represents the screen where the camera will render the scene.
            lights: List of Light objects that will illuminate the scene.
            show_bb: If True, bounding boxes of the objects will be shown in the scene.
            show_octree: If True, the octree will be shown in the scene.
        """

        self.initial_p = initial_p
        self.target_p = target_p
        
        self.front_v = Vector.from_points(initial_p=self.initial_p, final_p=self.target_p)
        self.distance = self.front_v.magnitude()
        
        up_v_temp = up_input_v
        self.right_v = Vector.cross(self.front_v, up_v_temp)
        self.up_v = Vector.cross(self.right_v, self.front_v)
        self.right_v = self.right_v.normalize()
        self.up_v = self.up_v.normalize()

        self.s = scene
    
        right_displacement = self.right_v * (self.s.pixel_size_h/2)
        up_displacement = self.up_v * (self.s.pixel_size_v/2)
        self.current_spot = self.target_p.add_vector(up_displacement).add_vector(right_displacement)
        
        self.lights = lights

    def go_horizontal(self, units=1) -> Point:
        hor_displacement = self.right_v * 2 / self.s.h_res * units
        next_spot = self.current_spot.add_vector(hor_displacement) 

        return next_spot
    
    def go_vertical(self, units=1):
        ver_displacement = self.up_v * 2 / self.s.v_res * units
        next_spot = self.current_spot.add_vector(ver_displacement)
        
        return next_spot

    def set_position(self, pos):
        self.current_spot = pos

    def get_sin_or_cos(self, sin_or_cos):
        return sqrt(1 - sin_or_cos**2)

    def calculate_color(self, obj, intersect_point: Point, octree, triangle_id=None, recursion_depth=0):
        """ Calculates the color of the object at the intersect point using Phong's model.

        Args:
            obj: Object that was intersected.
            intersect_point: Point where the intersection happened.
            octree: Octree object that contains the objects in the scene.
            triangle_id: Id of the triangle that was intersected in case the object intersected is a Mesh.
            recursion_depth: Current recursion depth.
        """

        recursion_limit = 3
        partial_result = np.array([0,0,0])
        reflected_vector = None
        obj_normal = obj.normal_of(intersect_point, triangle_id=triangle_id).normalize()
        for light in self.lights:
            light_vector = Vector.from_points(intersect_point, light.point)
            light_vector_inverted = -light_vector
            ray_from_light = Ray(origin=light.point, direction=light_vector_inverted)
            objects_to_intersect = octree.get_objects_to_intersect(ray_from_light)
            light_object, _ = self.calculate_intersection(ray_from_light, objects_to_intersect)
            if light_object and light_object != obj:
                continue

            cos_lv_normal = Vector.dot(obj_normal, light_vector.normalize())
            if obj.normal_always_positive:
                cos_lv_normal = abs(cos_lv_normal)
            else:
                cos_lv_normal = max(cos_lv_normal, 0)

            if triangle_id is None:
                obj_color = obj.color
            else:
                obj_color = obj.colors[triangle_id]
            diffusion = light.color * obj_color * obj.k_diffusion * cos_lv_normal

            camera_vector = Vector.from_points(intersect_point, self.initial_p)
            reflected_vector: Vector = (obj_normal * 2 * cos_lv_normal).add_vector(-light_vector.normalize())
            cos_rv_camera = Vector.dot(reflected_vector.normalize(), camera_vector.normalize())
            specular = light.color * obj.k_specular * (max(cos_rv_camera, 0) ** obj.shininess)
            partial_result = partial_result + diffusion + specular

        result = obj.k_ambient * self.ambient_light + partial_result
        v_vector = Vector.from_points(self.initial_p, intersect_point)
        reflected_v_vector = (obj_normal * 2 * Vector.dot(obj_normal, v_vector)).add_vector(-v_vector)
        if reflected_vector and recursion_depth < recursion_limit and obj.k_reflection > 0:
            reflected_ray = Ray(origin=intersect_point, direction=reflected_v_vector)
            reversed_reflected_ray = Ray(origin=intersect_point, direction=-reflected_v_vector)
            objs_to_intersect = octree.get_objects_to_intersect(reversed_reflected_ray, render_all_in_root_node=True)
            chosen_obj, chosen_intersect = self.calculate_intersection(reflected_ray, objs_to_intersect, current_obj=obj)
            if chosen_obj and chosen_obj != obj:
                point = chosen_intersect.get('point')
                triangle_id = chosen_intersect.get('triangle_id')
                reflected_color = self.calculate_color(chosen_obj, point, octree, triangle_id,
                                                       recursion_depth=recursion_depth+1)
                result = result + obj.k_reflection*reflected_color

        if recursion_depth < recursion_limit and obj.k_refraction > 0:
            n = obj.n_refraction
            cos_theta = Vector.dot(obj_normal, v_vector.normalize())
            sin_theta = self.get_sin_or_cos(cos_theta)
            sin_theta_t = sin_theta / n
            cos_theta_t = self.get_sin_or_cos(sin_theta_t)
            t_vector = (v_vector*(1/n)).add_vector(-obj_normal*(cos_theta_t - (1/n)*cos_theta))
            t_ray = Ray(origin=intersect_point, direction=t_vector)
            reversed_t_ray = Ray(origin=intersect_point, direction=-t_vector)
            objs_to_intersect = octree.get_objects_to_intersect(reversed_t_ray, render_all_in_root_node=True)
            chosen_obj, chosen_intersect = self.calculate_intersection(t_ray, objs_to_intersect, current_obj=obj)
            if chosen_intersect:
                chosen_triangle_id = chosen_intersect.get('triangle_id')
                different_objects = chosen_obj != obj or chosen_triangle_id is not None and chosen_triangle_id != triangle_id
                if different_objects:
                    point = chosen_intersect.get('point')
                    triangle_id = chosen_intersect.get('triangle_id')
                    refracted_color = self.calculate_color(chosen_obj, point, octree, triangle_id,
                                                           recursion_depth=recursion_depth+1)
                    result = result + obj.k_refraction*refracted_color

        result = np.clip(result, 0, 255)
        return result

    def calculate_intersection(self, ray: Ray, objs: List[ScreenObject], current_obj=None, ignore_debug=True):
        from objects import TMesh
        min_t = float('inf')
        chosen_obj = None
        chosen_intersect = None

        for obj in objs:
            if ignore_debug and not obj.real_object:
                continue
            intersect = obj.intersect(ray)
            t = intersect.get('t')

            if t is not None and t < min_t and (current_obj != obj or not isinstance(current_obj, TMesh)):
                min_t = t
                chosen_obj = obj
                chosen_intersect = intersect

        return chosen_obj, chosen_intersect

    def get_test_coords(self, *args, res=300):
        coords = []
        for coord in args:
            x, y = coord
            real_y = (res-1) - y
            if real_y % 2 != 0:
                real_x = (res-1) - x
            else:
                real_x = x
            coords.append([real_x, real_y])
        return coords


    def render(self, objs, use_octree=True, save_file=None, partial_render=False, show_bb=False, show_octree=False):
        """That traverses the pixels in the screen and calculates the color of each one.

        Args:
            objs: List of objects in the scene.
            use_octree: If True, the octree will be used to accelerate the intersection calculations.
            save_file: If not None, the rendered image will be saved in the file.
            partial_render: If True, the rendered image will be saved in the file after each 5% of progress.
            show_bb: If True, bounding boxes of the objects will be shown in the scene.
            show_octree: If True, the octree will be shown in the scene.
        """

        start_time = time.time()

        # Go to the left corner of the screen
        full_left_iter = -self.s.h_res // 2
        next_pos = self.go_horizontal(units=full_left_iter)
        self.set_position(next_pos)

        # Go to the bottom corner of the screen
        full_down_iter = -self.s.v_res // 2
        next_pos = self.go_vertical(units=full_down_iter)
        self.set_position(next_pos)

        go_left = False
        p_v = self.s.v_res - 1
        total_iterations = self.s.h_res * self.s.v_res
        counter = 0

        octree = Octree(objs)
        octree.active = use_octree

        if show_bb:
            bounding_boxes = [obj.bounding_box for obj in objs if obj.real_object and obj.bounding_box]
            objs.extend(bounding_boxes)


        for i in range(self.s.v_res):
            if go_left:
                units = -1
                p_h = self.s.h_res - 1
            else:
                units = 1
                p_h = 0
            for j in range(self.s.h_res):
                ray_dir = Vector.from_points(self.initial_p, self.current_spot)
                ray = Ray(self.current_spot, ray_dir)

                objects_to_intersect = octree.get_objects_to_intersect(ray, render_all_in_root_node=False, show_bb=show_bb)
                if show_octree:
                    objects_to_intersect.append(octree)

                color = np.array([0, 0, 0])  # black

                chosen_obj, chosen_intersect = self.calculate_intersection(ray, objects_to_intersect,
                                                                           ignore_debug=False)
                if chosen_obj:
                    point: Point = chosen_intersect.get('point')

                    if not chosen_obj.real_object:
                        color = chosen_obj.color * 255
                    else:
                        color = self.calculate_color(chosen_obj, point, octree, chosen_intersect.get('triangle_id'))

                self.s.draw_pixel(p_h, p_v, color)
                last_h = j == self.s.h_res - 1
                if not last_h:
                    self.set_position(self.go_horizontal(units=units))
                    if go_left:
                        p_h -= 1
                    else:
                        p_h += 1
                counter += 1
            self.set_position(self.go_vertical(1))
            p_v -= 1
            go_left = not go_left
            progress = counter / total_iterations * 100
            if progress % 5 == 0:
                print(f'Progress: {counter / total_iterations * 100:.2f}%')
                if save_file and partial_render:
                    np.save(save_file, self.s.real_grid)

        end_time = time.time()

        time_difference = end_time - start_time
        print(f"Rendered in {time_difference :2f} seconds")

        if save_file:
            np.save(save_file, self.s.real_grid)

        self.render_grid()

    def render_from_file(self, load_file):
        self.s.real_grid = np.load(load_file)
        self.render_grid()

    def render_grid(self):
        cv2.imshow("Ray tracing", self.s.real_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __str__(self):
        s = f"""
                initial_p: {self.initial_p}
                target_p: {self.target_p}
                front_v: {self.front_v.v}
                right_v: {self.right_v.v}
                up_v: {self.up_v.v}
            """
        return s
        

