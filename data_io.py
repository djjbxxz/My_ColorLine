import tensorflow as tf
import numpy as np
import os
from inference import get_move_from_index, get_move_from_index
import time


def example_path(num):
    return os.path.join(os.getcwd(), "dataset\\"+str(time.time())[0:11]+str(num)+".tfrecords")
# Write the records to a file.
# with tf.io.TFRecordWriter(example_path) as file_writer:
#   for _ in range(4):
#     x, y = np.random.random(), np.random.random()

#     record_bytes = tf.train.Example(features=tf.train.Features(feature={
#         "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
#         "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
#     })).SerializeToString()
#     file_writer.write(record_bytes)


# Read the data back out.
def decode_fn1(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"x": tf.io.FixedLenFeature([10], dtype=tf.float32),
         "y": tf.io.FixedLenFeature([10], dtype=tf.float32)}
    )


# for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
#   print("x = {x:.4f},  y = {y:.4f}".format(**batch))
from_stored = {"x": tf.io.FixedLenFeature([84*102], dtype=tf.float32),
               "y": tf.io.FixedLenFeature([4*102], dtype=tf.float32)}

from_trainable = {"x": tf.io.FixedLenFeature([9*9*4], dtype=tf.float32),
                  "y": tf.io.FixedLenFeature([6561], dtype=tf.float32)}


# def make_tensor(node):
#     x = np.ones(shape=[9, 9, 4], dtype=np.float32)
#     x[:, :, 0] = node.game_map
#     for i in range(3):
#         x[:, :, i+1] *= node.next_three[i]
#     move = get_move_from_index(node.real_move.last_move)


def write_data_store(node):
    x = np.concatenate([node.game_map.flatten(), node.next_three.flatten()])
    y = node.real_move.last_move.flatten()
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    writer(example_path, x, y)


# def read_data_store(filename):
#     dataset = tf.data.TFRecordDataset([filename]).map(decode_stored)
#     for record in dataset.take(1):
#         print(repr(record))

def read_data_store(filename):
    dataset = tf.data.TFRecordDataset([filename])
    for record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        print((example))

def decode_stored(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        from_stored
    )


# def writer(filename, x, y):
#     with tf.io.TFRecordWriter(filename) as file_writer:
#         record_bytes = tf.train.Example(features=tf.train.Features(feature={
#             "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
#             "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
#         })).SerializeToString()
#         file_writer.write(record_bytes)


def writer(filename, x, y):
    with tf.io.TFRecordWriter(filename) as file_writer:
        for i in range(x.shape[0]):
            example = serialize(x[i],y[i])
            file_writer.write(example)


def serialize(x, y):
    return tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
    })).SerializeToString()

# def write_data_trainable(nodes):
#     num = len(nodes)
#     x, y = np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)
#     # x = np.ones(shape=[84*num], dtype=np.uint8)
#     # y = np.zeros(shape=[4*num], dtype=np.uint8)
#     for node_num in range(len(nodes)):
#         x[node_num, :, :, 0] = nodes[node_num].game_map
#         for i in range(3):
#             x[node_num, :, :, i+1] *= nodes[node_num].next_three[i]
#         move = get_move_from_index(nodes[node_num].real_move.last_move)
#         y[node_num, move] = 1
#         x, y = x.flatten(), y .flatten()
#     writer(example_path, x, y)


def write_data_trainable(nodes):
    num = len(nodes)
    x, y = [], []
    for node in nodes:
        x.append(np.concatenate(
            (node.parent.game_map, node.parent.next_three), axis=None))
        y.append(np.concatenate((node.last_move), axis=None))
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    writer(example_path(num), x, y)


def change_color(color1, color2):
    a = 3


def take_a_look(x, y, index):
    temp_x = x[84*index:81+index*84]
    temp_x = np.reshape(temp_x, (9, 9))
    print(temp_x)
    print(y[0+4*index:4+4*index])


if __name__ == '__main__':
    example_path1 = os.path.join(
        os.getcwd(), "dataset\\"+"1612883675.12"+".tfrecords")
    read_data_store(example_path1)
