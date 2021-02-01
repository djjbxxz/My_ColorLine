import numpy as np
import random


def LayChess(game_map,num):
    index_list = GetEmptyIndex(game_map,num)
    for i in index_list:
        game_map[i] = random.randint(1, 7)
    return game_map


def GetEmptyIndex(game_map,num):
    empty = []
    for i in range(9):
        for j in range(9):
            if game_map[i, j] == 0:
                empty.append((i, j))
    random_empty = []
    for i in range(num):
        rand_index = random.randint(0, len(empty)-1)
        random_empty.append(empty[rand_index])
        empty.pop(rand_index)
    return random_empty


def get_random_start():
    game_map = np.zeros(shape = (9,9),dtype=np.int)
    game_map = LayChess(game_map,5)
    comingcolor = np.array([1,2,3], dtype=np.int)
    for i in range(3):
        comingcolor[i] = random.randint(1,7)
    return game_map,comingcolor



if __name__ == '__main__':
    for i in range(100):
        game = Game(9)
        game.LayChess(5)
    print(game.map)
