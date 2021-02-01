import tensorflow as tf
import numpy as np
import math
def make_move_into_matrix(move):
    action = np.zeros((9, 9))
    for i in range(2):
        action[move[i][0], move[i][1]] = i+1
    return action


def make_move_into_index(actions):
    move_index = []
    for i in range(len(actions)):
        temp = []
        for j in  range(2):
            temp.append(np.where(actions[i]==j+1))
        temp = np.array(temp)
        temp = temp.flatten()
        index_sum = 0
        for k in range(4):
            index_sum+= temp[k] * math.pow(9,3-k)
        move_index.append(index_sum)
    return move_index

def get_train_data(game_map, action):
    action = make_move_into_matrix(action)
    maps = []
    maps.append(game_map)
    actions = []
    actions.append(action)
    #翻转
    maps.append(np.flip(maps[0],axis=0))
    actions.append(np.flip(actions[0],axis=0))
    #旋转
    rot(maps)
    rot(actions)

    actions = make_move_into_index(actions)
    return maps,actions


def rot(content):
    length = len(content)
    for i in range(length):
        content.append(np.rot90(content[i]))
        for j in range(2):
            content.append(np.rot90(content[-1]))



if __name__ == '__main__':
    game_map = np.array([[1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [3, 3, 3, 3, 4, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4],
                         [3, 3, 3, 3, 3, 4, 4, 4, 4]])

    action = np.asarray([[8,0],[8,1]],dtype = int)
    _map,move = get_train_data(game_map,action)
    a=3

