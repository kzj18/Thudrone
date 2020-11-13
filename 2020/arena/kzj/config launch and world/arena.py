#!/usr/bin/python
#-*- encoding: utf8 -*-

'''
you need to put this file under UAV_SIM
and put target_demo.yaml under config
and put arena_demo.world under world
and modify the file name of world in arena.launch
the correct name is 'arena.world' not 'arena_1.world'
'''

import os

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

python_file = os.path.abspath('').replace('/', '//')
yaml = {
    'demo': python_file + '//config//target_demo.yaml',
    'file': python_file + '//config//target.yaml'
}
world = {
    'demo': python_file + '//world//arena_demo.world',
    'file': python_file + '//world//arena.world'
}

if __name__ == "__main__":
    points = {
        'fire': list(range(3)),
        'ball': list(range(5))
    }
    point_fire = -1
    while not point_fire in points['fire']:
        point_fire = input('fire point in ' + str(points['fire']) +':')
    points['fire'].remove(point_fire)
    current_points = {}
    current_points['red'] = -1
    current_points['yellow'] = -1
    current_points['blue'] = -1
    while not current_points['red'] in points['ball']:
        current_points['red'] = input('red point in ' + str(points['ball']) + ':')
    points['ball'].remove(current_points['red'])
    while not current_points['yellow'] in points['ball']:
        current_points['yellow'] = input('yellow point in ' + str(points['ball']) + ':')
    points['ball'].remove(current_points['yellow'])
    while not current_points['blue'] in points['ball']:
        current_points['blue'] = input('blue point in ' + str(points['ball']) + ':')
    
    answer = ''
    for _ in range(5):
        if _ == current_points['red']:
            answer += 'r'
        elif _ == current_points['yellow']:
            answer += 'y'
        elif _ == current_points['blue']:
            answer += 'b'
        else:
            answer += 'e'

    print('your answer is:' + answer)
    print('your fire is:' + str(point_fire))

    with open(yaml['demo'], 'r') as input_file:
        text = input_file.read()
        text = text.replace('answer', answer)
        print(text)
        with open(yaml['file'], 'w') as output_file:
            output_file.write(text)

    with open(world['demo'], 'r') as input_file:
        text = input_file.read()
        for index, iterm in enumerate(points['fire']):
            text = text.replace('unfire%d'%index, str(fire_coordinate[iterm]))
        text = text.replace('fire', str(fire_coordinate[point_fire]))
        for color in ['red', 'yellow', 'blue']:
            for index, dimension in enumerate(['x', 'y', 'z']):
                text = text.replace('ball_'+color+'_'+dimension, str(ball_coordinate[current_points[color]][index]))
        with open(world['file'], 'w') as output_file:
            output_file.write(text)