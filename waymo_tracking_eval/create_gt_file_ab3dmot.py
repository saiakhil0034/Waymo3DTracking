# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

import os
import numpy as np

import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils

try:
    tf.enable_eager_execution()
except:
    pass

def _create_gt_file_example():
  """Creates a GT objects file."""
  objects = metrics_pb2.Objects()

  # Store all sequence names from val.txt
  val_seq_list = list()
  val_path = "/team1/codes/3dObjDet/OpenPCDet_ravi/data/waymo/ImageSets/val_100b.txt"
  with open(val_path, "r") as f:
    val_seq_list = f.readlines()
  val_seq_list = [v.split(".")[0] for v in val_seq_list]

  # tfrecord base
  tf_base_pth = "/waymo-od/training/"
  # gt annos base
  anno_base_pth = "/team1/codes/3dObjDet/OpenPCDet_ravi/data/waymo/waymo_processed_data_100/"

  # Loop through each sequence
  for seq in val_seq_list:
    tf_pth = tf_base_pth + "{}.tfrecord".format(seq)
    anno_pth = anno_base_pth + "{}/{}.pkl".format(seq, seq)
    if not os.path.exists(tf_pth):
      print("Sequence {} is a validation sequence!".format(seq))
      tf_pth = "/waymo-od/validation/{}.tfrecord".format(seq)

    # load into memory
    dataset = tf.data.TFRecordDataset(str(tf_pth), compression_type='')
    gt_dset = np.load(anno_pth, allow_pickle=True)

    # loop through all frames in each segment
    tot = 0
    for cnt, data in enumerate(dataset):
      tot += 1
      print("Processing sequence: {}; frame: {}".format(seq, cnt))

      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(data.numpy()))
      annos = gt_dset[cnt]

      # x, y, z
      locs = annos['annos']['location']
      # l, w, h
      dims = annos['annos']['dimensions']
      # ry
      ry = annos['annos']['heading_angles']
      # obj_id
      objs = annos['annos']['obj_ids']
      # class
      clss = annos['annos']['name']
      nobj = objs.shape[0]
      # loop through all objects in given frame
      for i in range(nobj):
        o = metrics_pb2.Object()
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        #o.context_name = ('context_name for the prediction. See Frame::context::name '
        #                  'in  dataset.proto.')
        o.context_name = frame.context.name
        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        invalid_ts = -1
        # o.frame_timestamp_micros = invalid_ts
        o.frame_timestamp_micros = frame.timestamp_micros
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
        # o.camera_name = dataset_pb2.CameraName.FRONT
        o.camera_name = frame.camera_labels[0].name
        # extract x, y, z, l, w, h, ry, score, object_id, type
        # Populating box and score.
        box = label_pb2.Label.Box()
        box.center_x = locs[i, 0]
        box.center_y = locs[i, 1]
        box.center_z = locs[i, 2]
        box.length = dims[i, 0]
        box.width = dims[i, 1]
        box.height = dims[i, 2]
        box.heading = ry[i]
        o.object.box.CopyFrom(box)
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        o.score = 1.0
        # For tracking, this must be set and it must be unique for each tracked
        # sequence.
        o.object.id = objs[i]
        # Use correct type.
        """
        enum Type {
          TYPE_UNKNOWN = 0;
          TYPE_VEHICLE = 1;
          TYPE_PEDESTRIAN = 2;
          TYPE_SIGN = 3;
          TYPE_CYCLIST = 4;
        }

        WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
        """
        # o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
        cl = clss[i]
        if cl == "Cyclist":
          o.object.type = label_pb2.Label.TYPE_CYCLIST
        elif cl == "Vehicle":
          o.object.type = label_pb2.Label.TYPE_VEHICLE
        elif cl == "Sign":
          o.object.type = label_pb2.Label.TYPE_SIGN
        elif cl == "Pedestrian":
          o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
        else:
          o.object.type = label_pb2.Label.TYPE_UNKNOWN

        objects.objects.append(o)
    assert(tot == len(gt_dset))

  # Add more objects. Note that a reasonable detector should limit its maximum
  # number of boxes predicted per frame. A reasonable value is around 400. A
  # huge number of boxes can slow down metrics computation.

  # file to save the preds.bin to
  save_pth = "/team1/codes/3dObjDet/OpenPCDet_ravi/output/tracking_bins/waymo_100_25/gtb.bin"
  # Write objects to a file.
  f = open(save_pth, 'wb')
  f.write(objects.SerializeToString())
  f.close()


def main():
  _create_gt_file_example()


if __name__ == '__main__':
  main()
