"""
Dummy NuScenes DataParser for debugging
好品味的原则: 
1. 不破坏现有接口
2. 最小化特殊情况
3. 代码简洁直接
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Type
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.nuscenes_dataparser import (
    NuScenesDataParserConfig,
    ADDataParser,
    OPENCV_TO_NERFSTUDIO,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs


@dataclass
class DummyNuScenesDataParserConfig(NuScenesDataParserConfig):
    """Dummy配置 - 继承真实配置,只改必要的"""
    _target: Type = field(default_factory=lambda: DummyNuScenesDataParser)
    num_frames: int = 10
    """生成多少帧dummy数据"""
    points_per_frame: int = 5000
    """每帧点云数量"""


@dataclass  
class DummyNuScenesDataParser(ADDataParser):
    """
    Dummy DataParser - 生成假数据但符合真实格式
    
    设计哲学:
    - 接口与真实dataparser完全一致
    - 内部生成假数据,但结构真实
    - 可以无缝切换到真实数据
    """
    
    config: DummyNuScenesDataParserConfig
    
    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """
        核心方法 - 生成符合NeuRAD期望格式的输出
        这里体现"好品味": 消除了真实数据加载的复杂性,但保持接口一致
        """
        print(f"\n[DUMMY] 生成{split}数据...")
        
        # 生成相机数据
        cameras, image_filenames = self._get_dummy_cameras()
        
        # 生成lidar数据  
        lidars = self._get_dummy_lidars()
        
        # 生成actor轨迹(可选)
        actor_trajectories = []  # 简化:不生成动态物体
        
        # 场景边界框
        scene_box = SceneBox(
            aabb=torch.tensor([
                [-30.0, -30.0, -5.0],
                [30.0, 30.0, 10.0]
            ])
        )
        
        # 元数据
        metadata = {
            "is_dummy": True,
            "num_frames": self.config.num_frames,
        }
        
        # 返回DataparserOutputs
        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                **metadata,
                "lidars": lidars,
                "actor_trajectories": actor_trajectories,
            },
        )
    
    def _get_dummy_cameras(self) -> Tuple[Cameras, List[Path]]:
        """
        生成dummy相机数据
        简单直接: 圆形轨迹,固定内参
        """
        num_frames = self.config.num_frames
        num_cams = 6 if "all" in self.config.cameras else len(self.config.cameras)
        
        poses_list = []
        intrinsics_list = []
        times_list = []
        cam_idx_list = []
        filenames = []
        
        # 相机内参 (NuScenes典型值)
        fx, fy = 1266.4, 1266.4
        cx, cy = 800.0, 450.0  # 图像中心
        
        for frame_idx in range(num_frames):
            t = frame_idx / num_frames * 2 * np.pi
            
            for cam_idx in range(num_cams):
                # 圆形轨迹
                angle = t + cam_idx * (2 * np.pi / num_cams)
                radius = 5.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle) 
                z = 1.5
                
                # Pose矩阵
                pose = self._create_lookat_pose(
                    eye=np.array([x, y, z]),
                    target=np.array([0, 0, 0]),
                    up=np.array([0, 0, 1])
                )
                
                # 应用OpenCV到Nerfstudio的转换
                pose[:3, :3] = pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                
                poses_list.append(pose)
                intrinsics_list.append([fx, fy, cx, cy])
                times_list.append(frame_idx * 0.5)
                cam_idx_list.append(cam_idx)
                
                # Dummy文件名
                filenames.append(
                    Path(f"/tmp/dummy/cam_{cam_idx}/frame_{frame_idx:04d}.jpg")
                )
        
        # 转换为torch tensors
        camera_to_worlds = torch.from_numpy(np.stack(poses_list)).float()
        fx = torch.tensor([intr[0] for intr in intrinsics_list])
        fy = torch.tensor([intr[1] for intr in intrinsics_list])  
        cx = torch.tensor([intr[2] for intr in intrinsics_list])
        cy = torch.tensor([intr[3] for intr in intrinsics_list])
        
        height = torch.tensor([900] * len(poses_list))
        width = torch.tensor([1600] * len(poses_list))
        times = torch.tensor(times_list).float()
        
        # 创建Cameras对象
        cameras = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={
                "sensor_idxs": torch.tensor(cam_idx_list).unsqueeze(-1),
            }
        )
        
        print(f"[DUMMY] 生成 {len(poses_list)} 个相机poses")
        return cameras, filenames
    
    def _get_dummy_lidars(self) -> Lidars:
        """
        生成dummy激光雷达数据
        """
        num_frames = self.config.num_frames
        points_per_frame = self.config.points_per_frame
        
        origins_list = []
        directions_list = []
        ranges_list = []
        intensities_list = []
        times_list = []
        
        for frame_idx in range(num_frames):
            # 生成随机点云
            # 地面点
            num_ground = points_per_frame // 2
            ground_xy = np.random.randn(num_ground, 2) * 15
            ground_z = np.ones(num_ground) * -1.5 + np.random.randn(num_ground) * 0.1
            ground_points = np.column_stack([ground_xy, ground_z])
            
            # 物体点
            num_obj = points_per_frame - num_ground
            obj_points = np.random.randn(num_obj, 3) * np.array([10, 10, 2])
            
            # 合并
            all_points = np.vstack([ground_points, obj_points])
            
            # Lidar原点(车顶)
            origin = np.array([0, 0, 2.0])
            
            # 计算方向和距离
            directions = all_points - origin
            ranges = np.linalg.norm(directions, axis=1)
            directions = directions / ranges[:, None]
            
            # 强度(随机)
            intensities = np.random.rand(len(all_points)) * 255
            
            origins_list.append(np.tile(origin, (len(all_points), 1)))
            directions_list.append(directions)
            ranges_list.append(ranges)
            intensities_list.append(intensities)
            times_list.append(np.ones(len(all_points)) * frame_idx * 0.5)
        
        # 转为torch tensors
        origins = torch.from_numpy(np.vstack(origins_list)).float()
        directions = torch.from_numpy(np.vstack(directions_list)).float()
        ranges = torch.from_numpy(np.hstack(ranges_list)).float()
        intensities = torch.from_numpy(np.hstack(intensities_list)).float()
        times = torch.from_numpy(np.hstack(times_list)).float()
        
        # Lidar到世界的变换(固定)
        lidar_to_worlds = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        lidar_to_worlds[:, 2, 3] = 2.0  # Z方向偏移
        
        # 创建Lidars对象
        lidars = Lidars(
            origins=origins,
            directions=directions,
            ranges=ranges,
            intensities=intensities,
            lidar_to_worlds=lidar_to_worlds,
            lidar_type=LidarType.VELODYNE,
            times=times,
        )
        
        print(f"[DUMMY] 生成 {len(ranges)} 个lidar点")
        return lidars
    
    def _create_lookat_pose(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        创建lookat pose矩阵
        好品味: 标准的图形学操作,没有特殊情况
        """
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-10)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-10)
        
        up_new = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_new
        pose[:3, 2] = forward
        pose[:3, 3] = eye
        
        return pose