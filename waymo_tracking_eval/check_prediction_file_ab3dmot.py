import os
import pickle
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

try:
    tf.enable_eager_execution()
except:
    pass

def main():
    sequence_file = "/waymo-od/training/segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord"
    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    tot = 0
    for cnt, data in enumerate(dataset):
        tot += 1
        if (tot+1) % 500 == 0:
            print("Processed {} entries so far!".format(tot+1))
        # if cnt > 2:
        #     break
        # frame = dataset_pb2.Frame()
        # frame.ParseFromString(bytearray(data.numpy()))
        # print(frame.context.name)
        # print(frame.camera_labels[0].name)
        # print(frame.timestamp_micros)
    print("Total Entries: {}".format(tot))

if __name__ == "__main__":
    main()