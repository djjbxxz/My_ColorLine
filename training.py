import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers
from inference import load_model,inference
from PathfindingDll import load_PathfindingDLL
import MCTS

estimate = load_PathfindingDLL()
model = load_model()

model.fit(test_input, test_target)
model.save('path/to/location')