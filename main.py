import cv2
import numpy as np


class Vector:
    def __init__(self, v) -> None:
        self.v = np.array(v)
        
    @staticmethod
    def cross(v1, v2):
        return Vector(np.cross(v1, v2))


    def normalize(self):
        return Vector(self.v / self.magnitude())

    def magnitude(self):
        return np.linalg.norm(self.v)
    
    def get_location(self):
        return [self.p[0], self.p[1], self.p[2]]
    

    

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
    def __init__(self, initial_p, target_p, up_input_v, scene):
        up_input_v = np.array(up_input_v)
        self.initial_p = np.array(initial_p)
        self.target_p = np.array(target_p)
        self.front_v = Vector(self.target_p - self.initial_p)
        self.distance = self.front_v.magnitude()
        self.right_v = Vector.cross(self.front_v.v, up_input_v)
        self.up_v = Vector.cross(self.right_v.v, self.front_v.v)
        
        self.right_v = self.right_v.normalize()
        self.up_v = self.up_v.normalize()
    
        self.s = scene
        self.current_spot = self.initial_p + self.front_v.v + (self.right_v.v*self.s.pixel_size_h/2) + self.up_v.v*(self.s.pixel_size_v/2)
        
        self.grid_bound_tr = self.initial_p + (self.front_v.v + self.right_v.v + self.up_v.v)
        self.grid_bound_bl = self.initial_p + (self.front_v.v - self.right_v.v - self.up_v.v)
    
    def valid_position(self, pos):
        x_right, y_top, z_top = self.grid_bound_tr
        x_left, y_bottom, z_bottom = self.grid_bound_bl
        
        if pos[0] > max(x_right, x_left) or pos[0] < min(x_right, x_left):
            return False
        if pos[1] > max(y_top, y_bottom) or pos[1] < min(y_top, y_bottom):
            return False
        if pos[2] > max(z_top, z_bottom) or pos[2] < min(z_top, z_bottom):
            return False
        
        
        return True
    
    def go_horizontal(self, units=1):
        next_spot = self.current_spot + (self.right_v.v*2 / self.s.h_res)*units

        if not self.valid_position(next_spot):
            return False

        return next_spot
    
    def go_vertical(self, units=1):
        next_spot = self.current_spot + (self.up_v.v*2 / self.s.v_res)*units
        
        if not self.valid_position(next_spot):
            return False

        return next_spot

    def set_position(self, pos):
        self.current_spot = pos


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
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = direction / np.linalg.norm(direction)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            return min(t1, t2)
        else:
            return None
        
class Plane:
    def __init__(self, point, normal, color):
        self.point = np.array(point)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.color = np.array(color)

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  
            t = np.dot(self.point - ray.origin, self.normal) / denom
            return t if t > 0 else None
        else:
            return None

        

def render(c, objs):
    next_pos = c.go_horizontal(units=-1)
    while next_pos is not False:
        c.set_position(next_pos)
        next_pos = c.go_horizontal(units=-1)
    
    
    next_pos = c.go_vertical(units=-1)
    while next_pos is not False:
        c.set_position(next_pos)
        next_pos = c.go_vertical(units=-1)
        
    go_left = False
    p_v = c.s.v_res - 1
    p_h = 0
    
    next_pos_v = c.current_spot
    while (next_pos_v is not False):
        c.set_position(next_pos_v)
        if go_left:
            units = -1
            p_h = c.s.h_res - 1
        else:
            units = 1
            p_h = 0
        next_pos_h = c.current_spot
        while (next_pos_h is not False):
            c.set_position(next_pos_h)
            
            ray_dir = c.current_spot - c.initial_p
            ray = Ray(c.current_spot, ray_dir)
            
            min_t = float('inf')
            chosen_obj = None
            for obj in objs:
                t = obj.intersect(ray)
                if t and t < min_t:
                    min_t = t
                    chosen_obj = obj
            
            if chosen_obj:
                color = chosen_obj.color
            else:
                color = np.array([0,0,0]) # black
            
            c.s.draw_pixel(p_h, p_v, color * 255)
            next_pos_h = c.go_horizontal(units=units)
            if go_left:
                p_h -= 1
            else:
                p_h += 1
        next_pos_v = c.go_vertical(1)
        p_v -= 1
        go_left = not go_left
        

    cv2.imshow("Grid", c.s.real_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

c = Camera((0,7,8), (0,6,8), (1,0,0), Screen())
sphere1 = Sphere((0, 0, 8), 1, (0,0,1))
sphere2 = Sphere((0, 0, 12), 3, (1,0,0))
plane = Plane((0,2.9,0), (0, 1, 0), (0, 1, 0))
objs = [sphere1, sphere2, plane]
render(c, objs)
def main():
    po_input = input("Digite as coordenadas do ponto de origem separado por espaços: ").split(" ")
    po = tuple(map(float, po_input))

    pf_input = input("Digite o ponto de direção da câmera separado por espaços: ").split(" ")
    pf = tuple(map(float, pf_input))

    vector_up_input = input("Digite as coordenadas do vetor up separado por espaços: : ").split(" ")
    vector_up = tuple(map(float, vector_up_input))
    
    s = Screen()
    c = Camera(po, pf, vector_up, s)

    objs = []
    circle_count = input("Quantos circulos você deseja inserir na cena? - ")
    try:
        circle_count = int(circle_count)
    except:
        circle_count = 0
    for i in range(int(circle_count)):
        center_circle_input  = input(f"Informe as coordenadas do centro do círculo {i+1} separado por espaços: ").split(" ")
        center_circle = tuple(map(float, center_circle_input))

        radius_circle_input = int(input(f"Informe o raio do círculo: "))
        radius_circle = int(radius_circle_input)
        
        color_circle_input = input(f"Informe a cor do círculo em formato RGB normalizado, separado por espaços: ").split(" ")
        color_circle = tuple(map(float, color_circle_input))


        
        sphere = Sphere(center_circle, radius_circle, color_circle)
        print(sphere)
        objs.append(sphere)
    
    plane_count = input("Quantos planos você deseja inserir na cena? - ")
    try:
        plane_count = int(plane_count)
    except:
        plane_count = 0
    
    for i in range(int(plane_count)):
        point_plane_input = input(f"Informe as coordenadas de um ponto do plano {i+1} separadas por espaços: ").split(" ")
        point_plane = tuple(map(float, point_plane_input))

        normal_plane_input = input(f"Informe as coordenadas do vetor normal do plano {i+1}, separadas por espaço: ").split(" ")
        normal_plane = tuple(map(float, normal_plane_input))

        color_plane_input = input(f"Digite a cor do plano em escala RGB normalizada, separada por espaços: ").split(" ")
        color_plane = tuple(map(float, color_plane_input))
        
        plane = Plane(point_plane, normal_plane, color_plane)
        objs.append(plane)
        print(plane)
    
    render(c, objs)
        


# main()
