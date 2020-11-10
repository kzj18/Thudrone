#!/usr/bin/python
#-*- encoding: utf8 -*-

import numpy as np

'''
result_format = [
    ['r', 50],
    ['e', 0],
    ['g', 20],
    ['b', 30],
    ['e', 0]
]
'''
COLOR = ['r', 'g', 'b']

def guess(result):
    answer = ''
    counter = {
        'r': [],
        'g': [],
        'b': [],
        'e': []
    }
    for index in range(5-len(result)):
        result.append(['e', 0])
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

if __name__ == "__main__":
    size = int(input('size:'))
    output = bool(input('output:'))
    for _ in range(size):
        test = []

        for index in range(5):
            color = (COLOR+['e'])[np.random.randint(4)]
            if not color == 'e':
                area = np.random.randint(50)
            else:
                area = 0
            test.append([color, area])

        guess_answer = guess(test)
        if guess_answer.count('r') == 1 and guess_answer.count('g') == 1 and guess_answer.count('b') == 1:
            if output:
                print(test)
                print(guess_answer)
        else:
            print(test)
            print(guess_answer)
            print('wrong')
            exit()
    print('success')