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

def _create_pd_file_example():
  """Creates a prediction objects file."""
  objects = metrics_pb2.Objects()

  o = metrics_pb2.Object()
  # The following 3 fields are used to uniquely identify a frame a prediction
  # is predicted at. Make sure you set them to values exactly the same as what
  # we provided in the raw data. Otherwise your prediction is considered as a
  # false negative.
  o.context_name = ('context_name for the prediction. See Frame::context::name '
                    'in  dataset.proto.')
  # The frame timestamp for the prediction. See Frame::timestamp_micros in
  # dataset.proto.
  invalid_ts = -1
  o.frame_timestamp_micros = invalid_ts
  # This is only needed for 2D detection or tracking tasks.
  # Set it to the camera name the prediction is for.
  o.camera_name = dataset_pb2.CameraName.FRONT

  # Store all the object classes as a list
  # Store all sequence names from val.txt
  classes = ["CYCLIST", "PEDESTRIAN", "SIGN", "VEHICLE"]
  val_seq_list = list()
  val_path = "/team1/codes/3dObjDet/OpenPCDet_ravi/data/waymo/ImageSets/val_25.txt"
  with open(val_path, "r") as f:
    val_seq_list = f.readlines()
  val_seq_list = [v.split(".")[0] for v in val_seq_list]

  # this is the input to the algo
  tracking_output_base_pth = "/team1/codes/individual/vkonduru/AB3DMOT/results/waymo_25_5/"
  # print(val_seq_list)
  # Loop through each class
  for cl in classes:
    cl_pth = tracking_output_base_pth + cl + "/trk_withid/"
    if not os.path.isdir(cl_pth):
      print("Outputs for the class {} are not present in this folder".format(cl))
      continue
    # Loop through each sequence waymo_25_5_val/VEHICLE/trk_withid/{$SEGMENT}
    for seq in val_seq_list:
      seq_pth = cl_pth + seq
      seq_dir = os.fsencode(seq_pth)
      # loop through all frames in each segment
      for fi in os.listdir(seq_dir):
        frame_no_txt = os.fsdecode(fi)
        objs = np.loadtxt(seq_pth + "/" + frame_no_txt, dtype=str)
        objs = objs.reshape(-1, 17)
        print("Processing class: {}; sequence: {}; file {}".format(cl, seq, frame_no_txt))
        nobj = objs.shape[0]
        # loop through all objects in given frame
        for i in range(nobj):
          print(i, nobj, objs.shape)
          curr_obj = objs[i, :]
          # extract x, y, z, l, w, h, ry, score, object_id, type
          # Populating box and score.
          box = label_pb2.Label.Box()
          box.center_x = float(curr_obj[11])
          box.center_y = float(curr_obj[12])
          box.center_z = float(curr_obj[13])
          box.length = float(curr_obj[10])
          box.width = float(curr_obj[9])
          box.height = float(curr_obj[8])
          box.heading = float(curr_obj[14])
          o.object.box.CopyFrom(box)
          # This must be within [0.0, 1.0]. It is better to filter those boxes with
          # small scores to speed up metrics computation.
          o.score = float(curr_obj[15])
          # For tracking, this must be set and it must be unique for each tracked
          # sequence.
          o.object.id = curr_obj[16]
          # Use correct type.
          """
          enum Type {
            TYPE_UNKNOWN = 0;
            TYPE_VEHICLE = 1;
            TYPE_PEDESTRIAN = 2;
            TYPE_SIGN = 3;
            TYPE_CYCLIST = 4;
          }
          """
          # o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
          if cl == "CYCLIST":
            o.object.type = label_pb2.Label.TYPE_CYCLIST
          elif cl == "VEHICLE":
            o.object.type = label_pb2.Label.TYPE_VEHICLE
          elif cl == "SIGN":
            o.object.type = label_pb2.Label.TYPE_SIGN
          elif cl == "PEDESTRIAN":
            o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
          else:
            o.object.type = label_pb2.Label.TYPE_UNKNOWN

          objects.objects.append(o)

  # Add more objects. Note that a reasonable detector should limit its maximum
  # number of boxes predicted per frame. A reasonable value is around 400. A
  # huge number of boxes can slow down metrics computation.

  # file to save the preds.bin to
  save_pth = "/team1/codes/3dObjDet/OpenPCDet_ravi/output/tracking_bins/waymo_25_5/preds.bin"
  # Write objects to a file.
  f = open(save_pth, 'wb')
  f.write(objects.SerializeToString())
  f.close()


def main():
  _create_pd_file_example()


if __name__ == '__main__':
  main()
