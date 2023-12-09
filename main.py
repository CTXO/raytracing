from examples.examples import *
from scene import Camera, Screen
from structures import Point
from structures import Vector

origin_point = Point((0, 0, 0))
target_point = Point((0, 0, 0.5))
up_vector = Vector((0,1,0))
c = Camera(origin_point, target_point, up_vector, Screen())


c.render_from_file('./examples/angry_face.npy')

# def main():
#     po_input = input("Digite as coordenadas do ponto de origem separado por espaços: ").split(" ")
#     po = tuple(map(float, po_input))

#     pf_input = input("Digite o ponto de direção da câmera separado por espaços: ").split(" ")
#     pf = tuple(map(float, pf_input))

#     vector_up_input = input("Digite as coordenadas do vetor up separado por espaços: : ").split(" ")
#     vector_up = tuple(map(float, vector_up_input))
    
#     s = Screen()
#     c = Camera(po, pf, vector_up, s)

#     objs = []
#     circle_count = input("Quantos circulos você deseja inserir na cena? - ")
#     try:
#         circle_count = int(circle_count)
#     except:
#         circle_count = 0
#     for i in range(int(circle_count)):
#         center_circle_input  = input(f"Informe as coordenadas do centro do círculo {i+1} separado por espaços: ").split(" ")
#         center_circle = tuple(map(float, center_circle_input))

#         radius_circle_input = int(input(f"Informe o raio do círculo: "))
#         radius_circle = int(radius_circle_input)
        
#         color_circle_input = input(f"Informe a cor do círculo em formato RGB normalizado, separado por espaços: ").split(" ")
#         color_circle = tuple(map(float, color_circle_input))


        
#         sphere = Sphere(center_circle, radius_circle, color_circle)
#         print(sphere)
#         objs.append(sphere)
    
#     plane_count = input("Quantos planos você deseja inserir na cena? - ")
#     try:
#         plane_count = int(plane_count)
#     except:
#         plane_count = 0
    
#     for i in range(int(plane_count)):
#         point_plane_input = input(f"Informe as coordenadas de um ponto do plano {i+1} separadas por espaços: ").split(" ")
#         point_plane = tuple(map(float, point_plane_input))

#         normal_plane_input = input(f"Informe as coordenadas do vetor normal do plano {i+1}, separadas por espaço: ").split(" ")
#         normal_plane = tuple(map(float, normal_plane_input))

#         color_plane_input = input(f"Digite a cor do plano em escala RGB normalizada, separada por espaços: ").split(" ")
#         color_plane = tuple(map(float, color_plane_input))
        
#         plane = Plane(point_plane, normal_plane, color_plane)
#         objs.append(plane)
#         print(plane)
    
#     render(c, objs)
        


# main()
