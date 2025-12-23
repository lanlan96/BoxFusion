# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with CA-1M.
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
from data_util.dataset import CA1MDetectionDataset
from data_util.model_util_scannet import CA1MDatasetConfig
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.ap_helper import APCalculator


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet_with_rn', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='ca1m', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--data_path', type=str, default='/media/lyq/temp/dataset/CA-1M-slam', help='Data path. [default: %(default)s]')
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

# --------------------------- Global Configurations ---------------------------

# Set some frequently used constants from FLAGS
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

# Automatically set the output directory name according to the dataset
if FLAGS.dataset == 'scannet':
    FLAGS.dump_dir = 'eval_scannet'
elif FLAGS.dataset == 'ca1m':
    FLAGS.dump_dir = 'eval_ca1m'

# Create a unique sub-directory for this evaluation run (timestamped)
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
DUMP_DIR = os.path.join(FLAGS.dump_dir, time_string)
FLAGS.DUMP_DIR = DUMP_DIR

# Parse AP IoU threshold(s)
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

# Prepare output directory and logging
os.makedirs(DUMP_DIR, exist_ok=True)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    """Log a string to both output file and stdout."""
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)

# --------------------------- Dataset and DataLoader ----------------------------

def my_worker_init_fn(worker_id):
    """Ensure each data loader worker is initialized with a different random seed."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


DATASET_CONFIG = CA1MDatasetConfig()
TEST_DATASET = CA1MDetectionDataset(data_root=FLAGS.data_path)


# If a specific scene is specified, restrict the evaluation to that scene only
if FLAGS.scene_name != 'None':
    TEST_DATASET.scan_names = [FLAGS.scene_name]

print(f"Num scenes in test dataset: {len(TEST_DATASET)}")

TEST_DATALOADER = DataLoader(
    TEST_DATASET, 
    batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, 
    num_workers=1, 
    worker_init_fn=my_worker_init_fn
)
print("FLAGS.shuffle_dataset:", FLAGS.shuffle_dataset)

# -------------------------- Model/Hardware/Config Dicts -----------------------

# Device info and input dimension config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = (3 if FLAGS.use_color else 0) + (1 if not FLAGS.no_height else 0)

# Config dict for AP/evaluation calculations
CONFIG_DICT = {
    'remove_empty_box': not FLAGS.faster_eval,
    'use_3d_nms': FLAGS.use_3d_nms,
    'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms,
    'cls_nms': FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh,
    'dataset_config': DATASET_CONFIG
}

# End of global configuration section.



# --------------------------- Evaluate One Epoch ---------------------------


def evaluate_one_epoch():
    """
    Run evaluation for one epoch of the test dataset. 
    This function loads ground-truth and predicted bounding boxes, filters 
    data as needed, and computes average precision (AP) metrics for each IoU threshold.
    """
    # Create a list of AP calculators, one for each IoU threshold
    ap_calculator_list = [
        APCalculator(iou_thresh, DATASET_CONFIG.class2type) 
        for iou_thresh in AP_IOU_THRESHOLDS
    ]

    # Iterate through the test dataloader batches
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        # Get the ground truth corners of bounding boxes for current batch
        corners_array = batch_data_label['gt_corners'][0].cpu().numpy()
        scan_idx = batch_data_label['scan_idx'][0]

        # Prepare ground truth labels: assign class 0 and corners
        batch_gt_map_cls = [
            [(int(0), corners_array[i]) for i in range(len(corners_array))]
        ]

        # Compose the path where predicted results should be loaded from
        pred_path = os.path.join(FLAGS.pred_root, str(scan_idx) + '_boxes.pkl')

        # Only evaluate a fixed subset of scan indices; skip all others
        if not os.path.exists(pred_path):
            continue

        print(f'Eval batch: {batch_idx} scan_idx {scan_idx}')
        print(f'pred_path {pred_path}')

        # Load predicted bounding boxes and classes from file
        batch_pred_map_cls = load_data(pred_path)

        # Print count of predicted and ground truth classes for debugging
        pred_labels = [item[0] for item in batch_pred_map_cls[0]]
        gt_labels = [item[0] for item in batch_gt_map_cls[0]]
        print("pred_labels", Counter(pred_labels))
        print("gt_labels", Counter(gt_labels))

        # Update all AP calculators with predictions and ground truths
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Set of metric classes to evaluate and display
    class_list = ['mAP', 'APrec', 'ARecall']

    # Print evaluation results for all IoU thresholds and relevant metrics
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, f'iou_thresh: {AP_IOU_THRESHOLDS[i]:.6f}', '-' * 10)
        metrics_dict = ap_calculator.compute_metrics(OBB_IOU="required")
        # print('AP metrics: ', metrics_dict.keys())
        for cls in class_list:
            for key in metrics_dict:
                if cls in key:
                    log_string(f'eval {key}: {metrics_dict[key]:.6f}')



def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    evaluate_one_epoch()

if __name__=='__main__':
    eval()


