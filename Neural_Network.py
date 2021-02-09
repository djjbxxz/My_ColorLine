import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers
import numpy as np


def Preprocess(x):
    x=layers.Conv2D(128,3,padding = 'same')(x)
    x=layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def ResidualBlock(t):
    x=layers.Conv2D(128,3,padding = 'same')(t)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    x=layers.Conv2D(128,3,padding = 'same')(x)
    x=layers.BatchNormalization()(x)
    x=tf.add(x,t)
    return layers.ReLU()(x)


def Policy_head(x):
    x=layers.Conv2D(81,1)(x)
    x=layers.Flatten()(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU(max_value=1,name = 'policy_out')(x)
    return x

def Value_head(x):
    x=layers.Conv2D(1,1)(x)
    x=layers.Flatten()(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    x=layers.Dense(128)(x)
    x=layers.ReLU()(x)
    x=layers.Dense(1,name = 'value_out')(x)
    return x


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Model visualization
inputs = keras.Input(shape=(9,9,4),dtype = np.uint8)
out= Preprocess(inputs)
for i in range(1): #9
    out = ResidualBlock(out)
policy_out= Policy_head(out)
# value_out= Value_head(out)
model = keras.Model(inputs=inputs, outputs=[policy_out], name="My_model")

# Model visualization
model.summary()
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)



model.compile(
    loss={
        'policy_out': keras.losses.CategoricalCrossentropy()
    },
    optimizer=keras.optimizers.Adam()
)

model.save('test2')
