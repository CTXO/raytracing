import time

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
    

class Camera:
    def __init__(self, initial_p: Point, target_p: Point, up_input_v: Vector, scene: Screen):
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
        
        self.grid_bound_tr = self.target_p.add_vector(self.up_v).add_vector(self.right_v)
        self.grid_bound_bl = self.target_p.add_vector(-self.up_v).add_vector(-self.right_v)    

    def valid_position(self, pos: Point) -> bool:
        x_right, y_top, z_top = self.grid_bound_tr.p 
        x_left, y_bottom, z_bottom = self.grid_bound_bl.p
        
        x,y,z = pos.get_coordinates()
        if x > max(x_right, x_left) or x < min(x_right, x_left):
            return False
        if y > max(y_top, y_bottom) or y < min(y_top, y_bottom):
            return False
        if z > max(z_top, z_bottom) or z < min(z_top, z_bottom):
            return False
        
        return True
    
    def go_horizontal(self, units=1) -> Point:
        hor_displacement = self.right_v * 2 / self.s.h_res * units
        next_spot = self.current_spot.add_vector(hor_displacement) 

        if not self.valid_position(next_spot):
            return False

        return next_spot
    
    def go_vertical(self, units=1):
        ver_displacement = self.up_v * 2 / self.s.v_res * units
        next_spot = self.current_spot.add_vector(ver_displacement)
        
        if not self.valid_position(next_spot):
            return False

        return next_spot

    def set_position(self, pos):
        self.current_spot = pos

    def render(self, objs):
        start_time = time.time()
        next_pos = self.go_horizontal(units=-1)
        while next_pos is not False:
            self.set_position(next_pos)
            next_pos = self.go_horizontal(units=-1)
        
        
        next_pos = self.go_vertical(units=-1)
        while next_pos is not False:
            self.set_position(next_pos)
            next_pos = self.go_vertical(units=-1)
            
        go_left = False
        p_v = self.s.v_res - 1
        total_iterations = self.s.h_res * self.s.v_res
        counter = 0
        
        next_pos_v = self.current_spot
        while next_pos_v is not False:
            self.set_position(next_pos_v)
            if go_left:
                units = -1
                p_h = self.s.h_res - 1
            else:
                units = 1
                p_h = 0
            next_pos_h = self.current_spot
            while next_pos_h is not False:
                self.set_position(next_pos_h)
                
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
                    color = chosen_intersect.get('color')
                else:
                    color = np.array([0,0,0])  # black
                
                self.s.draw_pixel(p_h, p_v, color * 255)
                next_pos_h = self.go_horizontal(units=units)
                if go_left:
                    p_h -= 1
                else:
                    p_h += 1
                counter += 1
            next_pos_v = self.go_vertical(1)
            p_v -= 1
            go_left = not go_left
            print(f'Progress: {counter / total_iterations * 100}%')
            
        end_time = time.time()

        time_difference = end_time - start_time
        print(f"Rendered in {time_difference :2f} seconds")
        cv2.imshow("Ray casting", self.s.real_grid)
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