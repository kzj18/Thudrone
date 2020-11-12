#!/usr/bin/python
#-*- encoding: utf8 -*-

fire_coordinate = [
    1.75, 4.25, 6.75
]

ball_coordinate = [
    [6.5, 7, 1.72],
    [3.5, 7.5, 0.72],
    [5, 9.5, 1],
    [4, 11, 1.72],
    [1, 14.5, 0.2]
]

if __name__ == "__main__":
    with open('//home//kzj18//thudrone//2020//arena//kzj//world//arena_demo.world', 'r') as input_file:
        text = input_file.read()
        points_fire = list(range(3))
        points = {}
        current_points = {}
        for point_fire in points_fire:
            points['red'] = list(range(5))
            for point_red in points['red']:
                current_points['red'] = point_red
                points['yellow'] = points['red'] * 1
                points['yellow'].remove(point_red)
                for point_yellow in points['yellow']:
                    current_points['yellow'] = point_yellow
                    points['blue'] = points['yellow'] * 1
                    points['blue'].remove(point_yellow)
                    for point_blue in points['blue']:
                        current_points['blue'] = point_blue
                        result = text * 1
                        result = result.replace('fire', str(fire_coordinate[point_fire]))
                        for color in ['red', 'yellow', 'blue']:
                            for index, dimension in enumerate(['x', 'y', 'z']):
                                result = result.replace('ball_'+color+'_'+dimension, str(ball_coordinate[current_points[color]][index]))
                        with open('//home//kzj18//thudrone//2020//arena//kzj//world//arena_f%d_r%d_y%d_b%d.world'%(point_fire, point_red, point_yellow, point_blue), 'w') as output_file:
                            output_file.write(result)
        