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
    gt_annos = "/team1/codes/3dObjDet/OpenPCDet_ravi/data/waymo/waymo_processed_data_25/segment-10241508783381919015_2889_360_2909_360_with_camera_labels/segment-10241508783381919015_2889_360_2909_360_with_camera_labels.pkl"
    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    gt_dset = np.load(gt_annos, allow_pickle=True)
    tot = 0
    for cnt, data in enumerate(dataset):
        cur_frame = gt_dset[cnt]
        print(cur_frame.keys())
        tot += 1
        if cnt > 2:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        print(cur_frame['annos'].keys())
        # context name
        print(frame.context.name)
        # camera name
        print(frame.camera_labels[0].name)
        # timestamp
        print(frame.timestamp_micros)
        # x, y, z, l, w, h, heading
        print(cur_frame['annos']['location'].shape)
        print(cur_frame['annos']['dimensions'].shape)
        print(cur_frame['annos']['heading_angles'].shape)
        # score = 1.0
        # object id
        print(cur_frame['annos']['obj_ids'].shape)
        # object type
        print(cur_frame['annos']['name'])
    print("Total Entries: {}".format(tot))
    assert(tot == len(gt_dset))

if __name__ == "__main__":
    main()