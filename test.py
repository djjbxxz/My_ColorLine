import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ctypes import *
import numpy as np
import numpy.ctypeslib as npct
import tensorflow as tf

arr_1 = np.array([1,2,3,4], dtype = np.float32)
arr_2 = np.array([[1,2,3],[4,5,6]], dtype = np.int)

lib = npct.load_library("PathFindingDllforPython",r"C:\Users\Djj\source\repos\PathFindingDllforPython\x64\Release")

lib.Estimate.argtypes = [npct.ndpointer(dtype = np.float32, ndim = 1, flags="C_CONTIGUOUS"),
npct.ndpointer(dtype = np.int, ndim = 2, flags="C_CONTIGUOUS",shape = (2,3))]
tf.cast(arr_1,np.float32)
tf.cast(arr_2,tf.int16)
lib.Estimate(arr_1,arr_2)
print(arr_1)
print(arr_2)
a=3

   