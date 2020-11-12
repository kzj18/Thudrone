#!/usr/bin/python
#-*- encoding: utf8 -*-

import numpy as np

'''
result_format = [
    [['r', 50], ['e', 0], ['e', 0],...]
    [['e', 0], ['e', 0],...]
    [['y', 20], ['e', 0],...]
    [['b', 30], ['e', 0],...]
    [['e', 0], ['e', 0],...]
]
'''
COLOR = ['r', 'y', 'b']

def guess(input_result):
    result = input_result[:]
    for index, iterm in enumerate(result):
        if iterm == []:
            result[index].append(['e', 0])
    for index, iterm1 in enumerate(result):
        counter = {
            'r': [0, 0],
            'y': [0, 0],
            'b': [0, 0],
            'e': [0, 0]
        }
        for iterm2 in iterm1:
            counter[iterm2[0]][0] += 1
            if iterm2[1] > counter[iterm2[0]][1]:
                counter[iterm2[0]][1] = iterm2[1]
        max_color = max(COLOR, key=lambda x: counter[x][0])
        max_value = counter[max_color][0]
        if max_value < counter['e'][0]:
            result[index] = ['e', 0]
        else:
            max_list = []
            for name, value in counter.items():
                if value[0] == max_value:
                    max_list.append(name)
            color = max(max_list, key=lambda y: counter[y][1])
            result[index] = [color, counter[color][1]]
    answer = ''
    counter = {
        'r': [],
        'y': [],
        'b': [],
        'e': []
    }
    for index, point in enumerate(result):
        counter[point[0]].append(index)
    for color in COLOR:
        if len(counter[color]) > 1:
            expectation = max(counter[color], key=lambda x: result[x][1])
            counter[color].remove(expectation)
            for index in counter[color]:
                result[index] = ['e', 0]
                counter['e'].append(index)
            counter[color] = [expectation]
    for color in COLOR:
        if len(counter[color]) == 0:
            guess_answer = counter['e'][np.random.randint(len(counter['e']))]
            result[guess_answer] = [color, -1]
            counter['e'].remove(guess_answer)
    for key in result:
        answer += key[0]
    return answer

def confident(input_result):
    result = input_result[:]
    for index, iterm1 in enumerate(result):
        if not iterm1 == []:
            counter = {
                'r': [0, 0],
                'y': [0, 0],
                'b': [0, 0],
                'e': [0, 0]
            }
            for iterm2 in iterm1:
                counter[iterm2[0]][0] += 1
                if iterm2[1] > counter[iterm2[0]][1]:
                    counter[iterm2[0]][1] = iterm2[1]
            max_color = max(COLOR, key=lambda x: counter[x][0])
            max_value = counter[max_color][0]
            if max_value < counter['e'][0]:
                result[index] = [['e', 0]]
            else:
                max_list = []
                for name, value in counter.items():
                    if value[0] == max_value:
                        max_list.append(name)
                color = max(max_list, key=lambda y: counter[y][1])
                result[index] = [[color, counter[color][1]]]
    counter = {
        'r': [],
        'y': [],
        'b': [],
        'e': []
    }
    for index, point in enumerate(result):
        if not point == []:
            counter[point[0][0]].append(index)
    if len(counter['r']) == 1 and len(counter['y']) == 1 and len(counter['b']) == 1:
        return guess(result)
    elif (len(counter['r']) * len(counter['y']) == 1) or (len(counter['r']) * len(counter['b']) == 1) or (len(counter['b']) * len(counter['y']) == 1) and\
        (len(counter['r']) * len(counter['y']) * len(counter['b']) == 0):
        if len(counter['e']) == 2:
            return guess(result)
    return('unsure')

if __name__ == "__main__":
    size = int(input('size:'))
    output = bool(input('output:'))
    show_confident = bool(input('show confident:'))
    for _ in range(size):
        test = [
            [],
            [],
            [],
            [],
            []
        ]

        for index in range(np.random.randint(5)):
            for _ in range(np.random.randint(5)):
                color = (COLOR+['e'])[np.random.randint(4)]
                if not color == 'e':
                    area = np.random.randint(50)
                else:
                    area = 0
                test[index].append([color, area])
        
        guess_answer = guess(test)
        if guess_answer.count('r') == 1 and guess_answer.count('y') == 1 and guess_answer.count('b') == 1:
            if output:
                confident_answer = confident(test)
                if show_confident:
                    if not confident_answer == 'unsure':
                        print(test)
                        print(confident_answer)
                        print(guess_answer)
                else :
                    print(test)
                    print(confident_answer)
                    print(guess_answer)
        else:
            print(test)
            print(guess_answer)
            print('wrong')
            exit()
    print('success')