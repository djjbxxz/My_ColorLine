from visualization import visualization
from inference import get_move_index, load_model
from PathfindingDll import loadDLL
from GameControlerDLL import load_game_dll
from generate_training_data import get_train_data
import tensorflow as tf
import copy
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_value_softmax(value):
    return value/50


class Node:
    """
    a

    """

    def __init__(self, game_map, last_move, parent_node, p, is_root, next_three, score):
        self.last_move = last_move
        self.next_three = next_three
        self.game_map = game_map
        self.score = score  # 游戏得分
        self.value = 0

        self.parent = parent_node
        self.children = []
        self.is_end = False
        self.p = p  # 先验概率 computed by NN
        self.visit_count = 0  # N
        # self.q = 0  # （子树部分）行动价值均值
        # self.w = 0  # （子树部分）行动价值的总和
        self.is_root = is_root
        if self.last_move is not None:
            temp = judge(self.game_map, self.next_three, self.last_move)
            if temp == -1:
                self.is_end = True  # end of game
            else:
                self.score = score + temp
                self.value = temp

    def is_leaf(self):
        return len(self.children) == 0

    def select(self):
        if not self.is_end:
            if self.is_leaf():
                return self
            else:
                leaf = []
                for each in self.children:
                    t = each.select()
                    if t != None:
                        leaf.append(t)
                # 公式
                if len(leaf) != 0:
                    return max(leaf, key=lambda node: node.value+c_puct*node.p/(1+node.visit_count))
                # return max(leaf, key=lambda node: node.value+c_puct*node.p*np.sqrt(node.parent.visit_count+1)/(1+node.visit_count))

    def evaluate_expand(self,use_NN):
        policy = evaluate(self.game_map, self.next_three,use_NN)
        for i in range(6561):
            if policy[i] > 0:
                # expand it!
                self.children.append(
                    Node(copy.deepcopy(self.game_map), get_move_index(i), self, policy[i], False, self.next_three, self.score))

    def backup(self):
        if not self.is_root:
            # 不是根节点，更新本节点
            for i in range(len(self.children)):
                self.children[i].visit_count += 1
                self.value += self.children[i].value*decay
            self.visit_count += 1
            self.value = self.value/self.visit_count
        else:  # 是根节点
            for i in range(len(self.children)):
                self.children[i].visit_count += 1
            return self

        # 更新祖先节点
        this = self
        while(not this.parent.is_root):
            this.parent.visit_count += 1
            this.parent.value += this.value*decay/this.parent.visit_count
            this = this.parent
        return this.parent  # 返回当前state的node

    def play(self):
        self.real_move = max(
            self.children, key=lambda child: child.visit_count)
        # self.children.clear()
        self.real_move.is_root = True
        return self.real_move


def evaluate(game_map, next_three,use_NN):
    if use_NN:
        inputs = np.ndarray(shape=(1, 9, 9, 4))
        inputs[0, :, :, 0] = game_map
        for i in range(1, 4):
            inputs[0, :, :, i] = next_three[i-1]
        policy = model(inputs, training=False)
        policy = np.array(tf.reshape(tf.cast(policy, np.float32), [6561]))
    else:
        policy = np.ones(shape = (6561),dtype=np.float32)
    estimate(policy, game_map)

    return policy


if __name__ == '__main__':
    c_puct = 0.05  # 探索常数
    decay = 0.7  # 奖赏回传衰减值

    estimate = loadDLL()
    judge = load_game_dll()
    model = load_model('test')

    import time
    from Game import get_random_start
    game_map, comingcolor = get_random_start()
    # game_map = np.array([[1, 1, 2, 2, 4, 5, 6, 7, 3],
    #                      [1, 2, 1, 2, 6, 6, 7, 5, 4],
    #                      [1, 3, 3, 2, 3, 3, 2, 3, 3],
    #                      [1, 1, 2, 2, 1, 1, 6, 6, 3],
    #                      [0, 0, 5, 0, 4, 4, 3, 3, 1],
    #                      [1, 1, 2, 0, 3, 5, 5, 6, 6],
    #                      [7, 1, 4, 0, 3, 3, 4, 4, 3],
    #                      [3, 1, 2, 0, 1, 5, 5, 6, 6],
    #                      [7, 0, 1, 0, 0, 0, 0, 2, 3]])

    game_map = np.array([[1, 1, 2, 2, 4, 5, 6, 7, 3],
                         [1, 2, 1, 2, 6, 6, 7, 5, 4],
                         [2, 3, 3, 5, 3, 3, 2, 3, 3],
                         [1, 1, 2, 2, 1, 1, 6, 6, 3],
                         [0, 0, 5, 0, 4, 4, 3, 3, 1],
                         [1, 1, 2, 0, 3, 5, 5, 6, 6],
                         [7, 1, 4, 0, 3, 3, 4, 4, 3],
                         [3, 1, 2, 0, 1, 5, 5, 6, 6],
                         [7, 0, 1, 0, 0, 0, 0, 2, 3]])
    node = Node(game_map, None, None, None, True,
                comingcolor, 0)  # start of game
    start = node
    iteration = 40
    time_start = time.time()
    for j in range(1):
        for i in range(iteration):
            node = node.select()
            if node == None:
                break
            node.evaluate_expand(False)
            node = node.backup()
            if i % 20 == 0:
                time_end = time.time()
                print(str(i)+"次 "+time.strftime("%M:%S",
                                                time.localtime(time_end-time_start)))
        visualization(start)
        node = node.play()

    time_end = time.time()

    print(time.strftime("%M:%S", time.localtime(time_end-time_start)))
    print(start.real_move.last_move)
    a = 3
