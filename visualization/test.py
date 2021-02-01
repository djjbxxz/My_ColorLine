from os import stat
from typing import ContextManager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from numpy.lib.arraypad import _get_linear_ramps
from numpy.core.fromnumeric import sort
import matplotlib as mpl
import matplotlib.patches as mpathes
from numpy.lib.function_base import trim_zeros
mpl.rcParams['toolbar'] = 'None'

Index2Color = {0:'#FFFFFF',1:'#009C08',2:'#ED1C24',3:'#000084',4:'#B5A518',5:'#18C6F7',6:'#C618C6',7:'#943100'}

# def createPlot():
#     # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色
#     fig = plt.figure(1, facecolor='white')
#     # 把画布清空
#     fig.clf()
#     # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
#     # frameon表示是否绘制坐标轴矩形
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     # 绘制结点
#     plotLine('desicionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotLine('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
#     # 显示画图


# if __name__ == '__main__':
#     createPlot()
#     creatButton()
#     plt.show()

# fig, ax = plt.subplots(figsize=(15, 8))
# plt.subplots_adjust


scale = 20

point_coor = [[8, 7], [4, 5], [12, 5], [2, 3], [6, 3], [10, 3],
              [14, 3], [1, 1], [3, 1], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1], [15, 1]]
for i in range(len(point_coor)):
    for j in range(2):
        point_coor[i][j] = (point_coor[i][j])/scale
        if j == 1:
            point_coor[i][j] = point_coor[i][j]*2.5  # 抬高


# arrow_args = dict(arrowstyle="<-")


# callback = Index()
# axprev = plt.axes([0.5, 0.5, 0.1, 0.075])
# axnext = plt.axes([0.8, 0.1, 0.1, 0.075])
# bnext = Button(axnext, 'Next')
# bnext.on_clicked(callback.next)
# bprev = Button(axprev, 'Previous')
# bprev.on_clicked(callback.prev)
# plotLine(ax, ' ', point_coor[0], point_coor[1])
# plotLine(ax, ' ', (0, 0), (0.5, 0.5))

# for i in range(len(point_coor)):
#     temp = Button(plt.axes(list(point_coor[i])+[1/scale, 1/scale]), str(i+1))
#     temp.on_clicked(callback.next)


# plt.show()


def visualization(node):
    global fig
    fig, bTreeNode.Tree_ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    visual_node = bTreeNode(node)
    plt.show()

    return visual_node


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
        # plt.show()
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
        data = 'value : '+str(self.pointtoStree.value)[0:5]+'\n' +\
            'p : '+str(self.pointtoStree.p)[0:5]
        start = [self.parent.coord[0]+0.5/scale,
                 self.parent.coord[1]]
        end = [self.coord[0]+0.5/scale, self.coord[1]]
        self.plotLine(start, end, "<-", data)

    def plotLine(self,start, end, shape, text=None):
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
        self.drawboard()

    def drawboard(self):
        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.invert_yaxis()
        self.ax.yaxis.set_ticks_position('left') 
        self.drawboardLine()
        self.drawboardChess()
        plt.show()

    def drawboardLine(self):
        for i in range(0,10):
            self.plotLine([i/10,0/10],[i/10,9/10],"-")
            self.plotLine([0/10,i/10],[9/10,i/10],"-")
      
    def drawboardChess(self):
        game_map = self.pointtoStree.game_map
        for i in range(9):
            for j in range(9):
                self.ax.add_patch(self.plotFilledCircle([i+2,j+1],Index2Color[game_map[i][j]]))
    
    def move(self):
        a = 3

    def drawnext3(self):
        comingcolor = self.pointtoStree.comingcolor
        for i in range(3):
            self.Tree_ax.add_patch(self.plotFilledCircle([1,i+4],Index2Color[comingcolor[i]]))
        self.plotLine([3/10,10/10],[6/10,10/10],"-")
        for i in range(3,7):
            self.plotLine([i/10,9/10],[i/10,10/10],"-")

    def drawscore(self):
        a = 3

    @staticmethod
    def plotFilledCircle(xy,color):
        return mpathes.Circle([(xy[1]-0.5)/10,(xy[0]-0.5)/10],0.8/20,color = color)



if __name__ == '__main__':
    # createPlot()
    # creatButton()
    # plt.show()
    visualization(None)
