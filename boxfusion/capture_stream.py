"""
Dataset to stream RGB-D data from the NeRFCapture iOS App -> Cubify Transformer

Adapted from SplaTaM: https://github.com/spla-tam/SplaTAM
"""

import numpy as np
import time
import torch
import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import re

from dataclasses import dataclass

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset

from boxfusion.boxes import DepthInstance3DBoxes
from boxfusion.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from boxfusion.orientation import ImageOrientation, rotate_tensor, ROT_Z
from boxfusion.sensor import SensorArrayInfo, SensorInfo, PosedSensorInfo

# for ros2 version
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
import cv_bridge
import numpy as np
import queue
import threading
import time
from scipy.spatial.transform import Rotation


def parse_transform_3x3_np(data):
    return torch.tensor(data.reshape(3, 3).astype(np.float32))

def parse_transform_4x4_np(data):
    return torch.tensor(data.reshape(4, 4).astype(np.float32))

def parse_size(data):
    return tuple(int(x) for x in data.decode("utf-8").strip("[]").split(", "))




T_RW_to_VW = np.array([[0, 0, -1, 0],
                       [-1,  0, 0, 0],
                       [0, 1, 0, 0],
                       [ 0, 0, 0, 1]]).reshape((4,4)).astype(np.float32)

T_RC_to_VC = np.array([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]]).reshape((4,4)).astype(np.float32)

T_VC_to_RC = np.array([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]]).reshape((4,4)).astype(np.float32)

def compute_VC2VW_from_RC2RW(T_RC_to_RW):
    T_vc2rw = np.matmul(T_RC_to_RW,T_VC_to_RC)
    T_vc2vw = np.matmul(T_RW_to_VW,T_vc2rw)
    return T_vc2vw

def get_camera_to_gravity_transform(pose, current, target=ImageOrientation.UPRIGHT):
    z_rot_4x4 = torch.eye(4).float()
    z_rot_4x4[:3, :3] = ROT_Z[(current, target)]
    pose = pose @ torch.linalg.inv(z_rot_4x4.to(pose))

    # This is somewhat lazy.
    fake_corners = DepthInstance3DBoxes(
        np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])).corners[:, [1, 5, 4, 0, 2, 6, 7, 3]]
    fake_corners = torch.cat((fake_corners, torch.ones_like(fake_corners[..., :1])), dim=-1).to(pose)

    fake_corners = (torch.linalg.inv(pose) @ fake_corners.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
    fake_basis = torch.stack([
        (fake_corners[:, 1] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 1] - fake_corners[:, 0], dim=-1)[:, None],
        (fake_corners[:, 3] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 3] - fake_corners[:, 0], dim=-1)[:, None],
        (fake_corners[:, 4] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 4] - fake_corners[:, 0], dim=-1)[:, None],
    ], dim=1).permute(0, 2, 1)

    # this gets applied _after_ predictions to put it in camera space.
    T = Rotation.from_euler("xz", Rotation.from_matrix(fake_basis[-1].cpu().numpy()).as_euler("yxz")[1:]).as_matrix()

    return torch.tensor(T).to(pose)

MAX_LONG_SIDE = 1024





class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion_node')
        
        # 1. 
        self.frame_count = 0
        self.last_log_time = time.time()
        
        # 2. 
        self.bridge = cv_bridge.CvBridge()
        
 
        # 3. 
        self.rgb_queue = queue.Queue(maxsize=200)
        self.depth_queue = queue.Queue(maxsize=200)
        self.pose_queue = queue.Queue(maxsize=200)
        self.result_queue = queue.Queue(maxsize=200) 
        
        self.last_rgb_put_time = 0  # 
        self.last_depth_put_time = 0  
        self.last_pose_put_time = 0  # 
        self.last_pose_put_time = 0  # 
        self.MIN_INTERVAL = 0.05  # 

        # 4. 
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame = 'map'
        self.target_frame = 'camera_link'
        
        # 5. 
        self.rgb_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.rgb_callback, 5)  # QoS=5
        
        self.depth_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 5)
        
        # 6. 
        self.pose_timer = self.create_timer(0.02, self.pose_update)  # 50Hz
        
        # 7. 
        self.sync_timer = self.create_timer(0.033, self.process_synced_data)  # 30Hz
        self.data_callback = None  #

        self.get_logger().info("🚀 start")

    def set_data_callback(self, callback):

        self.data_callback = callback

    def rgb_callback(self, msg):

        current_time = time.monotonic()
        if current_time - self.last_rgb_put_time < self.MIN_INTERVAL:
            return  
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.rgb_queue.put((timestamp, cv_image), timeout=0.001)
            self.last_rgb_put_time = current_time  # 
        except Exception as e:
            self.get_logger().warn(f"RGB error: {str(e)}")

    def depth_callback(self, msg):

        current_time = time.monotonic()
        if current_time - self.last_depth_put_time < self.MIN_INTERVAL:
            return  # 
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.depth_queue.put((timestamp, depth_image), timeout=0.001)
            self.last_depth_put_time = current_time
        except Exception as e:
            self.get_logger().warn(f"error: {str(e)}")

    def pose_update(self):

        current_time = time.monotonic()
        if current_time - self.last_pose_put_time < self.MIN_INTERVAL:
            return  # skip
        try:
            if self.tf_buffer.can_transform(
                self.source_frame, 
                self.target_frame, 
                rclpy.time.Time()
            ):
                transform = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.target_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.time.Duration(seconds=0.05)
                )
                

                translation = transform.transform.translation
                rotation = transform.transform.rotation
                pose_matrix = self._quaternion_to_matrix(
                    translation.x, translation.y, translation.z,
                    rotation.x, rotation.y, rotation.z, rotation.w
                )
                

                stamp = transform.header.stamp
                timestamp = stamp.sec * 10**9 + stamp.nanosec
                

                self.pose_queue.put((timestamp, pose_matrix), timeout=0.001)
                self.last_pose_put_time = current_time
        except (TransformException, queue.Full) as e:
            pass
    
    def _quaternion_to_matrix(self, x, y, z, qx, qy, qz, qw):
        rot = Rotation.from_quat([qx, qy, qz, qw])
        rotation_matrix = rot.as_matrix()
        
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[0, 3] = x
        pose_matrix[1, 3] = y
        pose_matrix[2, 3] = z
        return pose_matrix


    def process_synced_data(self):
        """30Hz"""
        try:
            # 1. 
            rgb_stamp, rgb_data = self.rgb_queue.get(timeout=0.01)
            depth_stamp, depth_data = self.depth_queue.get(timeout=0.01)
            
            # 2. 
            rgb_data = cv2.resize(rgb_data, (640, 480))
            depth_data = cv2.resize(depth_data, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            # 3. 
            closest_pose = None
            min_time_diff = float('inf')
            MAX_TIME_DIFF = 50 * 1e6  # 50ms
            
            pose_items = []
            while not self.pose_queue.empty():
                pose_stamp, pose_matrix = self.pose_queue.get()
                pose_items.append((pose_stamp, pose_matrix))
                
                time_diff = abs(pose_stamp - rgb_stamp)
                if time_diff < min_time_diff and time_diff < MAX_TIME_DIFF:
                    min_time_diff = time_diff
                    closest_pose = pose_matrix
                    pose_stamp_match = pose_stamp
            
                    # 4. 
                    for item in pose_items:
                        if item[0] != pose_stamp_match:  
                            self.pose_queue.put(item)
            
            if closest_pose is None:
                return
                

            self.frame_count += 1
            current_time = time.time()
            # if current_time - self.last_log_time >= 1.0:
            
            fps = self.frame_count / (current_time - self.last_log_time)

            self.frame_count = 0
            self.last_log_time = current_time
            

            self.process_fusion_data(rgb_data, depth_data, closest_pose)
            
            
        except queue.Empty:
            pass

    def process_fusion_data(self, rgb, depth, pose):
        """
            rgb: [H, W, 3] numpy
            depth: [H, W] numpy (dtype=uint16)
            pose: [4, 4] numpy(dtype=float64)
        """

        position = pose[:3, 3]
        rotation = pose[:3, :3]

        

        try:
            self.result_queue.put({
                'rgb': rgb,
                'depth': depth,
                'pose': pose
            }, timeout=0.001)
        except queue.Full:
            self.get_logger().warn("full skip data")

    def get_synced_data(self, timeout=1.0):

        return self.result_queue.get(timeout=timeout)


class ROSDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(ROSDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']


        # self.frame_ids = range(0, len(self.img_files))
        self.num_frames = 10000000000000000 #len(self.frame_ids)
        self.cfg = cfg
        self.img_height = cfg['cam']['H']
        self.img_width = cfg['cam']['W']
        self.K = np.array([[cfg['cam']['fx'], 0.0, cfg['cam']['cx']],
                            [0.0, cfg['cam']['fy'], cfg['cam']['cy']],
                            [0.0,0.0,1.0]])
        self.fx = cfg['cam']['fx']
        self.fy = cfg['cam']['fy']
        self.cx = cfg['cam']['cx']
        self.cy = cfg['cam']['cy']
        self.depth_scale = cfg['cam']['png_depth_scale']
        self.has_depth = has_depth

        self.video_id = 'ros' #matches


        #ROS INITIALIZATION
        rclpy.init(args=None)
    
        self.node = MultiSensorFusion()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        
        # 
        self.spin_thread = threading.Thread(target=self.executor.spin)
        self.spin_thread.start()

    def __len__(self):
        return 100000000

    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        #start ROS Loop
        while True:
            try:
                # 
                data = self.node.get_synced_data(timeout=1.0)
                
                color_data = data['rgb']
                depth_data = data['depth']
                pose = data['pose']

                
                color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

                H, W = depth_data.shape
                color_data = cv2.resize(color_data, (W, H))

                #
                #Step2:try to warp the data like the original dataset    
                result = dict(wide=dict())
                wide = PosedSensorInfo()            
                
                # OK, we have a frame. Fill on the requisite data/fields.
                image_info = ImageMeasurementInfo(
                    size=(self.img_width, self.img_height),
                    K=torch.tensor([
                        [self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0]
                    ])[None])


                image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

                wide.image = image_info
                result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

                if self.load_arkit_depth and not self.has_depth:
                    raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

                depth_info = None            
                if self.has_depth:
                    # We'll eventually ensure this is 1/4.
                    depth_info = DepthMeasurementInfo(
                        size=(self.img_width, self.img_height),
                        K=torch.tensor([
                            [self.fx  , 0.0, self.cx ],
                            [0.0, self.fy , self.cy ],
                            [0.0, 0.0, 1.0]
                        ])[None])

                    depth_scale = self.depth_scale
                    wide.depth = depth_info


                    depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                    

                    depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                    result["wide"]["depth"] = depth
                    


                    if max(wide.image.size) > MAX_LONG_SIDE:
                        scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                        # scale_factor = 1
                        new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                        wide.image = wide.image.resize(new_size)
                        result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                    
                else:
                    # Even for RGB-only, only support a certain long size.
                    # if max(wide.image.size) > MAX_LONG_SIDE:
                    # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    scale_factor = 1

                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]



                RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
                wide.RT = RT[None]

                current_orientation = wide.orientation
                target_orientation = ImageOrientation.UPRIGHT #UPRIGHT

                T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation) #[3,3]
                wide = wide.orient(current_orientation, target_orientation)
                # T_gravity = torch.eye(3)

                # No need for pose anymore.
                wide.RT = torch.eye(4)[None]
                wide.T_gravity = T_gravity[None]


                gt = PosedSensorInfo()        
                gt.RT = parse_transform_4x4_np(pose)[None]
                if depth_info is not None:
                    gt.depth = depth_info

                sensor_info = SensorArrayInfo()
                sensor_info.wide = wide
                sensor_info.gt = gt

                result["meta"] = dict(video_id=video_id, timestamp=index)
                result["sensor_info"] = sensor_info



                index+=1
                yield result


            except queue.Empty:
                print("waiting data...")
                time.sleep(0.1)  

            # except KeyboardInterrupt:
            #     print("stop")
            # finally:
            #     #
            #     self.executor.shutdown()
            #     self.spin_thread.join()
            #     self.node.destroy_node()
            #     rclpy.shutdown()










class ScannetDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(ScannetDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']

        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.basedir, 'pose'))
        
        self.img_files=self.img_files[self.start:]
        self.depth_paths=self.depth_paths[self.start:]
        self.poses=self.poses[self.start:]

        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.cfg = cfg
        self.img_height = cfg['cam']['H']
        self.img_width = cfg['cam']['W']
        self.K = np.array([[cfg['cam']['fx'], 0.0, cfg['cam']['cx']],
                            [0.0, cfg['cam']['fy'], cfg['cam']['cy']],
                            [0.0,0.0,1.0]])
        self.fx = cfg['cam']['fx']
        self.fy = cfg['cam']['fy']
        self.cx = cfg['cam']['cx']
        self.cy = cfg['cam']['cy']
        self.depth_scale = cfg['cam']['png_depth_scale']
        self.has_depth = has_depth
        pattern = r'scene\d{4}_\d{2}'  
        matches = re.findall(pattern, cfg['data']['datadir'])
        self.video_id = matches

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        self.last_valid_pose = None
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)

            if not np.isinf(c2w).any():
                self.last_valid_pose = c2w
            else:
                c2w = self.last_valid_pose 
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def __len__(self):
        return self.num_frames



    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        while True:

            #Step1: load data
            color_path = self.img_files[index]
            depth_path = self.depth_paths[index]
            color_data = cv2.imread(color_path)

            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.exr' in depth_path:
                raise NotImplementedError()

            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
            result = dict(wide=dict())
            wide = PosedSensorInfo()            
            
            # OK, we have a frame. Fill on the requisite data/fields.
            image_info = ImageMeasurementInfo(
                size=(self.img_width, self.img_height),
                K=torch.tensor([
                    [self.fx, 0.0, self.cx],
                    [0.0, self.fy, self.cy],
                    [0.0, 0.0, 1.0]
                ])[None])

            # print(image_info.size)

            image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
                # We'll eventually ensure this is 1/4.
                depth_info = DepthMeasurementInfo(
                    size=(self.img_width, self.img_height),
                    K=torch.tensor([
                        [self.fx  , 0.0, self.cx ],
                        [0.0, self.fy , self.cy ],
                        [0.0, 0.0, 1.0]
                    ])[None])

                depth_scale = self.depth_scale
                wide.depth = depth_info

                # If I understand this correctly, it looks like this might just want the lower 16 bits?
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                result["wide"]["depth"] = depth
                
                # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                # wide.image = wide.image.resize(desired_image_size)

                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    # scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
            else:
                # Even for RGB-only, only support a certain long size.
                # if max(wide.image.size) > MAX_LONG_SIDE:
                # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                scale_factor = 1

                new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]
            # print(f"T_gravity: {T_gravity}")

            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = depth_info

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info

            index+=1
            
            yield result



class CA1MDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(CA1MDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']
        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'rgb', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.basedir, 'all_poses.npy'))
        
        self.img_files=self.img_files[self.start:]
        self.depth_paths=self.depth_paths[self.start:]
        self.poses=self.poses[self.start:]

        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.cfg = cfg

        depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_depth.txt')).reshape(3,3)
        self.K = np.array([[depth_intric[0,0], 0.0, depth_intric[0,2]],
                            [0.0, depth_intric[1,1], depth_intric[1,2]],
                            [0.0,0.0,1.0]])
        self.fx = self.K[0,0]
        self.fy = self.K[1,1]
        self.cx = self.K[0,2]
        self.cy = self.K[1,2]

        if self.K[0,2]< self.K[1,2]:
            self.img_height=cfg["cam"]["W"] #l
            self.img_width=cfg["cam"]["H"] #s
        else:
            self.img_height=cfg["cam"]["H"]
            self.img_width=cfg["cam"]["W"]


        self.depth_scale = cfg['cam']['png_depth_scale']
        self.has_depth = has_depth
        pattern = r'\b4\d{7}\b'  
        matches = re.findall(pattern, cfg['data']['datadir'])
        self.video_id = matches


    def load_poses(self, path):
        self.poses = np.load(path).reshape(-1,4,4)

    def __len__(self):
        return self.num_frames



    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        while True:

            #Step1: load data
            color_path = self.img_files[index]
            depth_path = self.depth_paths[index]
   
            color_data = cv2.imread(color_path)

            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.exr' in depth_path:
                raise NotImplementedError()
            
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
            result = dict(wide=dict())
            wide = PosedSensorInfo()            
            
            # OK, we have a frame. Fill on the requisite data/fields.
            image_info = ImageMeasurementInfo(
                size=(self.img_width, self.img_height),
                K=torch.tensor([
                    [self.fx, 0.0, self.cx],
                    [0.0, self.fy, self.cy],
                    [0.0, 0.0, 1.0]
                ])[None])

            image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
                # We'll eventually ensure this is 1/4.
                depth_info = DepthMeasurementInfo(
                    size=(self.img_width, self.img_height),
                    K=torch.tensor([
                        [self.fx, 0.0, self.cx ],
                        [0.0, self.fy, self.cy ],
                        [0.0, 0.0, 1.0]
                    ])[None])

                depth_scale = self.depth_scale
                wide.depth = depth_info

                # If I understand this correctly, it looks like this might just want the lower 16 bits?
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                result["wide"]["depth"] = depth

                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    # scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
            else:
                # Even for RGB-only, only support a certain long size.
                # if max(wide.image.size) > MAX_LONG_SIDE:
                # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                scale_factor = 1

                new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]


            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = depth_info

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info


            index+=1
            yield result

