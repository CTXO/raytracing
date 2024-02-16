import time
from typing import List

from structures import Vector, Point
import numpy as np
import cv2


class Screen:
    h_res = 400
    v_res = 400
    screen_size = 800
    pixel_size_h = 2 / h_res
    pixel_size_v = 2 / v_res
    
    real_pixel_size_h = screen_size // h_res
    real_pixel_size_v = screen_size // v_res
    
    real_grid = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)
    
    
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
        
    def calculate_color(self, obj, intersect_point: Point, triangle_id=None):
        partial_result = np.array([0,0,0])
        for light in self.lights:
            light_vector = Vector.from_points(intersect_point, light.point)
            obj_normal = obj.normal_of(intersect_point, triangle_id).normalize()
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
        result = np.clip(result, 0, 255)
        return result

    def render(self, objs, save_file=None):
        start_time = time.time()

        full_left_iter = -self.s.h_res // 2
        next_pos = self.go_horizontal(units=full_left_iter)
        self.set_position(next_pos)

        full_down_iter = -self.s.v_res // 2
        next_pos = self.go_vertical(units=full_down_iter)
        self.set_position(next_pos)

        go_left = False
        p_v = self.s.v_res - 1
        total_iterations = self.s.h_res * self.s.v_res
        counter = 0

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

                min_t = float('inf')
                chosen_obj = None
                for obj in objs:
                    intersect = obj.intersect(ray)
                    t = intersect.get('t')
                    if t and t < min_t:
                        min_t = t
                        chosen_obj = obj
                        chosen_intersect = intersect

                if chosen_obj:
                    point = chosen_intersect.get('point')
                    color = self.calculate_color(chosen_obj, point, chosen_intersect.get('triangle_id'))
                    # color = chosen_intersect.get('color')
                else:
                    color = np.array([0,0,0])  # black

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
        

class Ray:
    def __init__(self, origin: Point, direction: Vector):
        self.origin = origin
        self.direction = direction.normalize()

