import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl
import matplotlib.patches as mpathes
mpl.rcParams['toolbar'] = 'None'

Index2Color = {0: '#FFFFFF', 1: '#009C08', 2: '#ED1C24', 3: '#000084',
               4: '#B5A518', 5: '#18C6F7', 6: '#C618C6', 7: '#943100'}

scale = 20

point_coor = [[8, 7], [4, 5], [12, 5], [2, 3], [6, 3], [10, 3],
              [14, 3], [1, 1], [3, 1], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1], [15, 1]]
for i in range(len(point_coor)):
    for j in range(2):
        point_coor[i][j] = (point_coor[i][j])/scale
        if j == 1:
            point_coor[i][j] = point_coor[i][j]*2.5  # 抬高


def show_tree(node):
    fig, bTreeNode.Tree_ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    visual_node = bTreeNode(node)
    plt.show()

    return visual_node


def show_Board(node):
    Board(node)


def show_real_moves(node):
    Boards_all(node)


class Board:
    '''
    show一个board状态\n
    ax可选，
    传入node应包括game_map,next_three,score
    '''

    def __init__(self, node, fig=None, ax=None):
        if fig == None and ax == None:
            self.fig, self.ax = self.get_ax()
        else:
            self.fig = fig
            self.ax = ax
        self.pointtoStree = node
        self.show()

    @staticmethod
    def get_ax():
        fig, ax = plt.subplots(figsize=(4.5, 5))
        plt.subplots_adjust(left=0, bottom=0, right=1.11, top=1)
        Board.set_ax(ax)
        return fig, ax

    @staticmethod
    def set_ax(ax):
        ax.invert_yaxis()
        ax.yaxis.set_ticks_position('left')

    def show(self):
        self.drawboardLine()
        self.drawboardChess()
        self.drawmove()
        self.drawnext3()
        self.drawscore()
        plt.show()

    def drawboardLine(self):
        for i in range(0, 10):
            self.plotLine([i/10, 0/10], [i/10, 9/10], "-")
            self.plotLine([0/10, i/10], [9/10, i/10], "-")

    def drawboardChess(self):
        game_map = self.pointtoStree.game_map
        for i in range(9):
            for j in range(9):
                self.ax.add_patch(self.plotFilledCircle(
                    [i+2, j+1], Index2Color[game_map[i][j]]))

    def plotmove(self, start, end, text):
        self.ax.add_patch(self.plotRetangle(start))
        self.ax.add_patch(self.plotRetangle(end))
        self.plotLine_index(start, end, '->', text)

    def drawmove(self):
        if self.pointtoStree.real_move != None:
            start = self.pointtoStree.real_move.last_move[0]
            end = self.pointtoStree.real_move.last_move[1]
            self.plotmove(start, end, '')
        else:
            for i in range(len(self.pointtoStree.children[0:2])):
                start = self.pointtoStree.children[i].last_move[0]
                end = self.pointtoStree.children[i].last_move[1]
                self.plotmove(start, end, str(i+1))

    def plotLine_index(self, start, end, shape, text):
        arrow_args = dict(arrowstyle=shape)
        self.ax.annotate(text, xy=[(end[1]+0.5)/10, (8-end[0]+0.5)/10], xycoords='axes fraction', color='white',
                         xytext=[(start[1]+0.5)/10, (8-start[0]+0.5)/10],  arrowprops=arrow_args)

    def plotLine(self, start, end, shape, text=None):
        arrow_args = dict(arrowstyle=shape)
        self.ax.annotate("", xy=start, xycoords='axes fraction',
                         xytext=end,  arrowprops=arrow_args)
        if text != None:
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            self.ax.text((start[0]+end[0])/2-0.5 /
                         scale, (start[1]+end[1])/2, text, bbox=props)

    def drawnext3(self):
        comingcolor = self.pointtoStree.next_three
        for i in range(3):
            self.ax.add_patch(self.plotFilledCircle(
                [1, i+4], Index2Color[comingcolor[i]]))
        self.plotLine([3/10, 10/10], [6/10, 10/10], "-")
        for i in range(3, 7):
            self.plotLine([i/10, 9/10], [i/10, 10/10], "-")

    def drawscore(self):
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        self.ax.text(0.7, 0.06, 'score:' +
                     str(self.pointtoStree.score), bbox=props)

    @staticmethod
    def plotFilledCircle(xy, color):
        return mpathes.Circle([(xy[1]-0.5)/10, (xy[0]-0.5)/10], 0.8/20, color=color)

    @staticmethod
    def plotRetangle(xy):
        return mpathes.Rectangle([(xy[1]+0.05)/10, (xy[0]+0.05+1)/10], 0.9/10, 0.9/10, color='r', fill=False)


class Boards_all(Board):

    def __init__(self, node, fig=None, ax=None):
        super().__init__(node, fig=fig, ax=ax)

    def show(self):
        self.drawboardLine()
        self.drawboardChess()
        self.drawmove()
        self.drawnext3()
        self.drawscore()
        self.draw_next_Button()
        self.draw_last_Button()
        plt.show()

    def draw_last_Button(self):
        if(self.pointtoStree.parent != None):
            self.last_button = self.draw_button(
                [0.05, 0.92], 'last', self.callback_last)

    def draw_next_Button(self):
        if(self.pointtoStree.real_move != None):
            self.next_button = self.draw_button(
                [0.2, 0.92], 'next', self.callback_next)

    def draw_button(self, position, text, callback):
        button = Button(
            plt.axes(list(position)+[1/10, 0.6/10]), text)
        button.on_clicked(callback)
        return button

    def callback_last(self, event):
        self.reset_fig()
        Boards_all(self.pointtoStree.parent, ax=self.ax, fig=self.fig)

    def callback_next(self, event):
        self.reset_fig()
        Boards_all(self.pointtoStree.real_move, ax=self.ax, fig=self.fig)

    def reset_fig(self):
        plt.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.set_ax(self.ax)

    def drawmove(self):
        if self.pointtoStree.real_move == None:
            return 0
        start = self.pointtoStree.real_move.last_move[0]
        end = self.pointtoStree.real_move.last_move[1]
        self.plotmove(start, end, '')


class bTreeNode:
    Tree_ax = None

    def __init__(self, STreeNode, layer=1, index=1, parent=None):
        self.ax = bTreeNode.Tree_ax
        self.pointtoStree = STreeNode
        self.layer = layer
        self.coord = point_coor[index-1]
        self.children = []
        self.parent = parent
        self.index = index
        self.is_leaf = self.pointtoStree.is_leaf()
        self.drawLine()
        self.drawButton()
        if(STreeNode.is_leaf() != True and self.layer < 4):
            self.pointtoStree.children.sort(
                key=lambda cnode: cnode.value, reverse=True)
            childlist = self.pointtoStree.children[0:2]
            for i in range(len(childlist)):
                self.children.append(
                    bTreeNode(childlist[i], layer+1, index*2+i, self))

    def drawLine(self):
        if(self.parent == None):
            return False
        data = 'va:'+str(self.pointtoStree.value)[0:5]+'\n' +\
            'p:'+str(self.pointtoStree.p)[0:4]+'\n' +\
            'vi:'+str(self.pointtoStree.visit_count)
        start = [self.parent.coord[0]+0.5/scale,
                 self.parent.coord[1]]
        end = [self.coord[0]+0.5/scale, self.coord[1]]
        self.plotLine(start, end, "<-", data)

    def plotLine(self, start, end, shape, text=None):
        arrow_args = dict(arrowstyle=shape)
        self.ax.annotate("", xy=start, xycoords='axes fraction',
                         xytext=end,  arrowprops=arrow_args)
        if text != None:
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            self.ax.text((start[0]+end[0])/2-0.5 /
                         scale, (start[1]+end[1])/2, text, bbox=props)

    def drawButton(self):
        self.button = Button(
            plt.axes(list(self.coord)+[1/scale, 1/scale]), str(self.index))
        self.button.on_clicked(self.callback)

    def callback(self, event):
        # self.drawboard()
        show_Board(self.pointtoStree)

    def drawboard(self):
        fig, self.ax = plt.subplots(figsize=(4.5, 5))
        plt.subplots_adjust(left=0, bottom=0, right=1.11, top=1)
        self.ax.invert_yaxis()
        self.ax.yaxis.set_ticks_position('left')
        self.drawboardLine()
        self.drawboardChess()
        self.drawmove()
        self.drawnext3()
        self.drawscore()
        plt.show()

    def drawboardLine(self):
        for i in range(0, 10):
            self.plotLine([i/10, 0/10], [i/10, 9/10], "-")
            self.plotLine([0/10, i/10], [9/10, i/10], "-")

    def drawboardChess(self):
        game_map = self.pointtoStree.game_map
        for i in range(9):
            for j in range(9):
                self.ax.add_patch(self.plotFilledCircle(
                    [i+2, j+1], Index2Color[game_map[i][j]]))

    def drawmove(self):
        for i in range(len(self.children)):
            start = self.children[i].pointtoStree.last_move[0]
            end = self.children[i].pointtoStree.last_move[1]
            self.ax.add_patch(self.plotRetangle(start))
            self.ax.add_patch(self.plotRetangle(end))
            self.plotLine_index(start, end, '->', i+1)

    def plotLine_index(self, start, end, shape, text):
        arrow_args = dict(arrowstyle=shape)
        self.ax.annotate(text, xy=[(end[1]+0.5)/10, (8-end[0]+0.5)/10], xycoords='axes fraction',
                         xytext=[(start[1]+0.5)/10, (8-start[0]+0.5)/10],  arrowprops=arrow_args)

    def drawnext3(self):
        comingcolor = self.pointtoStree.next_three
        for i in range(3):
            self.ax.add_patch(self.plotFilledCircle(
                [1, i+4], Index2Color[comingcolor[i]]))
        self.plotLine([3/10, 10/10], [6/10, 10/10], "-")
        for i in range(3, 7):
            self.plotLine([i/10, 9/10], [i/10, 10/10], "-")

    def drawscore(self):
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        self.ax.text(0.7, 0.06, 'score:' +
                     str(self.pointtoStree.score), bbox=props)

    @staticmethod
    def plotFilledCircle(xy, color):
        return mpathes.Circle([(xy[1]-0.5)/10, (xy[0]-0.5)/10], 0.8/20, color=color)

    @staticmethod
    def plotRetangle(xy):
        return mpathes.Rectangle([(xy[1]+0.05)/10, (xy[0]+0.05+1)/10], 0.9/10, 0.9/10, color='r', fill=False)


if __name__ == '__main__':
    # createPlot()
    # creatButton()
    # plt.show()
    visualization(None)
