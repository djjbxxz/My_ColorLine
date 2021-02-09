from MCTS import Node
from PathfindingDll import load_PathfindingDLL
from GameControlerDLL import load_GameControlerDLL
from Game import get_random_start
import numpy as np
from inference import get_move_to_index, load_model
import tensorflow as tf

model = load_model('test')
inputs = np.ones(shape=(1,9,9,4),dtype=np.float32)
policy = model(inputs, training=True)