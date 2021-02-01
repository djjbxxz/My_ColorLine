from random import choice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons,Button
import matplotlib.patches as mpathes

fig, ax = plt.subplots(figsize=(6, 6))
scale = 20
Index2Color = {0:'#FFFFFF',1:'#009C08',2:'#ED1C24',3:'#000084',4:'#B5A518',5:'#18C6F7',6:'#C618C6',7:'#943100'}

def plotLine(start, end, shape, text=None):
        arrow_args = dict(arrowstyle=shape)
        ax.annotate("", xy=start, xycoords='axes fraction',
                              xytext=end,  arrowprops=arrow_args)
        if text != None:
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text((start[0]+end[0])/2-0.5 /
                              scale, (start[1]+end[1])/2, text, bbox=props)


def drawboardChess(i,j,color):
    return plotFilledCircle([i+2,j+1],Index2Color[color])

def plotFilledCircle(xy,color):
    return mpathes.Circle([(xy[1]-0.5)/10,(xy[0]-0.5)/10],0.8/20,color = color)

def drawnext3():
    comingcolor = [1,2,3]
    for i in range(3):
        ax.add_patch(plotFilledCircle([1,i+4],Index2Color[comingcolor[i]]))
    plotLine([3/10,10/10],[6/10,10/10],"-")
    for i in range(3,7):
        plotLine([i/10,9/10],[i/10,10/10],"-")

drawnext3()
for i in range(0,10):
    plotLine([i/10,0/10],[i/10,9/10],"-")
    plotLine([0/10,i/10],[9/10,i/10],"-")

for i in range(8):    
    ax.add_patch(drawboardChess(i,i,i))

plt.show()