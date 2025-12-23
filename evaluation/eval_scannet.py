# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with ScanNet.
"""
import os
import sys
import numpy as np
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d

from collections import Counter
from utils.utils import *
from data_util.dataset import ScannetDetectionDataset
from data_util.model_util_scannet import ScannetDatasetConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.ap_helper import APCalculator
from utils.ap_helper import parse_groundtruths


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet_with_rn', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--data_path', type=str, default='/media/lyq/mydata/Dataset/ScanNet/', help='Data path. [default: /media/lyq/data/dataset/ScanNet/scannet_train_detection_data]')
parser.add_argument('--dump_dir', default='eval_scannet', help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.15,0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--scene_name', type=str, default='None', help='NMS IoU threshold. [default: scene0169_00]')
parser.add_argument('--pred_path', type=str, default='None', help='NMS IoU threshold. [default: scene0169_00]')
parser.add_argument('--pred_root', type=str, default='/home/lyq/myprojects/ml-cubifyanything/results/nofusion', help='NMS IoU threshold. [default: scene0169_00]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--gpu', type=int, default=0, help='gpu to allocate')


FLAGS = parser.parse_args()

# ---------------- Global Configuration Begin ----------------
# Set batch size and number of input points.
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

# Set evaluation dump directory
if FLAGS.dataset == 'scannet':
    FLAGS.dump_dir = 'eval_scannet'
elif FLAGS.dataset == 'ca1m':
    FLAGS.dump_dir = 'eval_ca1m'
DUMP_DIR = FLAGS.dump_dir
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
DUMP_DIR = os.path.join(DUMP_DIR, time_string)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

# Prepare output dump directory and log file
os.makedirs(DUMP_DIR, exist_ok=True)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)


# Initialize datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Dataset and dataloader setup
DATASET_CONFIG = ScannetDatasetConfig()
TEST_DATASET = ScannetDetectionDataset(
    'val',
    num_points=NUM_POINT,
    augment=False,
    use_color=FLAGS.use_color,
    use_height=(not FLAGS.no_height),
    data_path='./data_util/scannet_train_detection_data'
)

# Optionally evaluate only specific scene
if FLAGS.scene_name != 'None':
    TEST_DATASET.scan_names = [FLAGS.scene_name]

print(len(TEST_DATASET))

TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset,
    num_workers=1,
    worker_init_fn=my_worker_init_fn
)
print("FLAGS.shuffle_dataset:", FLAGS.shuffle_dataset)

# Set CUDA device and calculate number of input channels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

# AP calculation config
CONFIG_DICT = {
    'remove_empty_box': (not FLAGS.faster_eval),
    'use_3d_nms': FLAGS.use_3d_nms,
    'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms,
    'cls_nms': FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh,
    'dataset_config': DATASET_CONFIG
}
# ---------------- Global Configuration End ----------------


def evaluate_one_epoch():
    """
    Main evaluation loop over the test dataloader.
    """
    ap_calculator_list = [
        APCalculator(iou_thresh, DATASET_CONFIG.class2type)
        for iou_thresh in AP_IOU_THRESHOLDS
    ]

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key]  

        end_points = {}

        end_points['model'] = 'scannet'

        # Populate end_points with batch data
        for key in batch_data_label:
            if key != 'center_label':
                assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        # Parse ground truth boxes for AP
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        batch_gt_map_cls = [[(int(0), j[1]) for j in batch_gt_map_cls[0]]]
        gt_labels = [i[0] for i in batch_gt_map_cls[0]]

        # Obtain scene index/ID
        scan_idx = batch_data_label['scan_name'][0]

        # Load alignment matrix for coordinate transformation
        meta_file = os.path.join(FLAGS.data_path, scan_idx, scan_idx + ".txt")
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                # Parse 4x4 transformation matrix
                axis_align_matrix = [float(x)
                                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))


        pred_path = os.path.join(FLAGS.pred_root, str(scan_idx) + '_boxes.pkl')
        if not os.path.exists(pred_path):
            batch_pred_map_cls = [[]]
            continue
        else:
            batch_pred_map_cls = load_data(pred_path)
            bbox = [i[1] for i in batch_pred_map_cls[0]]

            tranformed_center = axis_align_matrix[None, :3, :3] @ np.transpose(bbox, (0, 2, 1))
            tranformed_center = np.transpose(tranformed_center, (0, 2, 1)) + axis_align_matrix[None, :3, 3]
            bbox = tranformed_center

            batch_pred_map_cls = flip_axis_to_camera(bbox)
            batch_pred_map_cls = obb_to_aabb_corners(batch_pred_map_cls)
            batch_pred_map_cls = reorganize_obb_to_aabb(batch_pred_map_cls)

            batch_pred_map_cls = [
                (int(0), batch_pred_map_cls[i], 1.0) for i in range(len(batch_pred_map_cls))
            ]
            batch_pred_map_cls = [batch_pred_map_cls]

        pred_labels = [i[0] for i in batch_pred_map_cls[0]]
        print("****************************************************")
        print("scan_idx", scan_idx)
        print("pred_labels", Counter(pred_labels))
        gt_labels = [i[0] for i in batch_gt_map_cls[0]]
        print("gt_labels", Counter(gt_labels))

        # Accumulate predictions and groundtruth for AP
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # AP metrics to evaluate
    class_list = ['mAP', 'APrec', 'ARecall']

    # Compute and log average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics_scannet(OBB_IOU="required")
        for cls in class_list:
            for key in metrics_dict:
                if cls in key:
                    log_string('eval %s: %f' % (key, metrics_dict[key]))


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # See: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    evaluate_one_epoch()


if __name__ == '__main__':
    eval()

