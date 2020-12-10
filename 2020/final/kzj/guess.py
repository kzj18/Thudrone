#!/usr/bin/python
#-*- encoding: utf8 -*-

import rospy
import random
import copy

def guess(results):
    result_list_1 = [
        [],
        [],
        [],
        [],
        []
    ]
    for result in results:
        if result[1] == 4:
            result_list_1[result[0]-1].append(3)
        elif not result[1] == -1:
            result_list_1[result[0]-1].append(result[1])
    print('STEP 1: result is %s'%result_list_1)
    result_list_2 = []
    for result in result_list_1:
        if result == []:
            result_list_2.append(-1)
        else:
            result_no_3 = []
            for class_num in result:
                if not class_num == 3:
                    result_no_3.append(class_num)
            if not result_no_3 == []:
                result_list_2.append(max(result_no_3, key=lambda a: a.count))
            else:
                result_list_2.append(max(result, key=lambda a: a.count))
    print('STEP 2: result is %s'%result_list_2)
    class_list = [0, 1, 2]
    undo_list = []
    for index, result in enumerate(result_list_2):
        if result in class_list:
            class_list.remove(result)
        elif result == -1:
            undo_list.append(index)
    if len(class_list) > len(undo_list):
        random.shuffle(class_list)
        for index, item in enumerate(undo_list):
            result_list_2[item] = class_list[index]
    else:
        random.shuffle(undo_list)
        undo_copy = copy.deepcopy(undo_list)
        for index, item in enumerate(class_list):
            result_list_2[undo_list[index]] = item
            del undo_copy[index]
        for index in undo_copy:
            result_list_2[index] = 3
    print('STEP 3: result is %s'%result_list_2)
    return result_list_2
