import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy.ctypeslib as npct
from PathfindingDll import loadDLL


def load_model(model_name):
   return keras.models.load_model(model_name)

def trans_map(cint):
    if cint < 0:
        print ("不合法")
        return
    elif cint < 10:
        return cint

    elif cint >= 10:
        return chr(cint - 10 + 65)

def tenToAny(n, origin):
   l = []
   while True:
         s = origin // n
         tmp = origin % n
         l.append(trans_map(tmp))
         if s == 0:
               break
         origin = s
   l.reverse()
   return l
    
def get_move_index(move):
   index = tenToAny(9,move)
   
   a = list(np.zeros(shape=(4-len(index),),dtype=int))+index
   return np.reshape(a,[2,2])

def evaluate(game_map,next_three,get_max_p_move=False):
   inputs = np.ndarray(shape=(9,9,4))
   inputs[:,:,0] = game_map
   for i in range(1,4):
      inputs[:,:,i] = next_three[i-1]
   # estimate = loadDLL()
   policy,value=model(inputs,training=False)
   policy = tf.cast(policy,np.float32)
   policy = tf.reshape(policy,[6561])
   value = tf.reshape(value,[1,])
   game_map = tf.cast(inputs[0,:,:,0],tf.int32)
   policy = np.array(policy)
   value = np.array(value)
   game_map = np.array(game_map)
   estimate(policy,game_map)
   return policy,value

   # if get_max_p_move:
   #    move = np.argmax(policy)
   #    index = get_move_index(move)


if __name__ == '__main__':
   #DEBUG
   from numpy import random
   inputs = np.round(np.zeros(shape=(1,9,9,4)))
   inputs[0,0,0,0] = 1
   model = load_model()
   evaluate(inputs)
   ba=3
