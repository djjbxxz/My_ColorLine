import tensorflow as tf
import numpy as np
import math
import itertools


def make_move_into_matrix(move):
    action = np.zeros((9, 9), dtype=np.uint8)
    for i in range(2):
        action[move[i][0], move[i][1]] = i+1
    return action


def make_move_into_index(actions):
    temp = []
    for i in range(2):
        x, y = np.where(actions == i+1)
        temp.append([x[0], y[0]])
    return temp


def rotate(data):
    new_data = []
    game_map = np.reshape(data[0][0:81], (9, 9))
    next_three = data[0][81:84]
    move = make_move_into_matrix(data[1])
    for i in range(3):
        move = np.rot90(move)
        game_map = np.rot90(game_map)
        new_data.append([np.concatenate(
            (game_map.flatten(), next_three), axis=None), np.array(make_move_into_index(move), dtype=np.uint8)])
    return new_data


def get_train_data(data):
    game_map, action = data[0], data[1]
    action = make_move_into_matrix(action)
    # 翻转

    maps.append(np.flip(maps[0], axis=0))
    actions.append(np.flip(actions[0], axis=0))
    # 旋转
    rotate(maps)
    rotate(actions)

    actions = make_move_into_index(actions)
    return maps, actions


def flip(data):
    move = make_move_into_matrix(data[1])
    move = np.flip(move, axis=0)
    move = make_move_into_index(move)
    return [np.concatenate((np.flip(np.reshape(data[0][0:81], (9, 9)), axis=0), data[0][81:84]), axis=None), np.array(move, dtype=np.uint8)]


def get_color_map(color_num):

    def get_color_C(color_num):
        return list(itertools.combinations(range(1, 8), color_num))

    def get_color_A(colors):
        return list(itertools.permutations(colors, len(colors)))

    temp = []
    for each in get_color_C(color_num):
        temp.extend(get_color_A(each))
    return temp


def get_color_map_by_color_num():
    color_map_by_colornum = []
    for i in range(1, 8):
        color_map_by_colornum.append(get_color_map(i))
    return color_map_by_colornum


class GetMoreData():
    color_map_by_color_num = None

    def __init__(self, data):
        if GetMoreData.color_map_by_color_num == None:
            GetMoreData.color_map_by_colornum = get_color_map_by_color_num()
        self.data = [data]
        self.rotate()
        self.flip()

    def flip(self):
        new_data = []
        for sample in self.data:
            new_data.append(flip(sample))
        self.data.extend(new_data)

    def rotate(self):
        new_data = []
        for sample in self.data:
            new_data.extend(rotate(sample))
        self.data.extend(new_data)

    def get_data(self):
        return self.data

if __name__ == '__main__':
    data = []
    game_map = np.array([[1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [3, 3, 3, 3, 4, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
    next_three = np.array([1, 2, 3], dtype=np.uint8)
    data.append(np.concatenate((game_map, next_three), axis=None))
    data.append(np.asarray([[8, 0], [8, 1]], dtype=np.uint8))
    data = GetMoreData(data).get_data()
    # _map, move = get_train_data([game_map, action])
    color_map_by_colornum = get_color_map_by_color_num()
    temp = get_color_map(7)
    color = range(1, 8)
    # print(list(map(lambda x:''.join(x),itertools.combinations(color, 2)))) #C2X
    # print(list(map(lambda x:''.join(x),itertools.permutations('ABCD', 2)))) #A2X
    # action = np.asarray([[8, 0], [8, 1]], dtype=int)
    # _map, move = get_train_data(game_map, action)
