"""
构建场景数据生成脚本 - 点云位姿估计
Generate construction scene data with point cloud for pose estimation

功能:
1. 从world2.usd场景采集多模态数据
2. RGB图像、深度图、点云（带RGB）
3. 3D边界框位姿标注
4. 实例分割掩码
5. 相机参数
"""

import omni.timeline
import asyncio
import omni.usd as usd
import numpy as np
from pxr import Usd, UsdGeom
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import cv2
import os
import json
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import re

print("="*60)
print("构建场景点云数据生成脚本")
print("="*60)

"""========== 参数配置 =========="""

"""输出路径配置"""
taskNum = 'construction_world2_v3'  # v3：改进点云采集同步
path_dir_root = "/home/xander/partime/ConstructionScenePoseEstimation"
path_dir_dataset = f"{path_dir_root}/dataset_{taskNum}"
path_dir_rgb = f"{path_dir_dataset}/rgb"
path_dir_depth = f"{path_dir_dataset}/depth"
path_dir_pointcloud = f"{path_dir_dataset}/pointcloud"
path_dir_label = f"{path_dir_dataset}/labels"
path_dir_logs = f"{path_dir_dataset}/logs"

"""场景参数"""
# 相机参数
camera_path = "/World/Camera_0"
img_width = 1280
img_height = 720

# 相机位置范围（用于随机采样）
# 场景包围盒: 50x50x8米, 中心[0,0,3.7]
cam_distance_range = [15.0, 30.0]   # 距离中心15-30米（平衡距离和质量）
cam_height_range = [2.0, 6.0]       # 相机高度2-6米（扩大范围增加多样性）
cam_angle_range = [0, 360]          # 水平角度范围

# 目标瞄准点（场景中心）
# 注意：在采样时会动态调整为与相机相同的高度，保持水平拍摄

# 数据质量控制
min_pointcloud_points = 100   # 最小点云点数阈值
max_retry_per_frame = 5       # 每帧最大重试次数（降低以加快生成）
enable_pointcloud_validation = False  # 禁用验证，先采集所有数据

"""数据集参数"""
max_iterations = 41  # 生成的总帧数（快速测试）
current_iter = 0

"""物体类别定义（根据world2.usd场景中的语义标签）"""
# 语义标签名称 -> 类别ID（必须与add_semantic_labels.py中设置的名称匹配）
construction_class = {
    # 交通锥 (class_id = 0)
    "trafficcone": 0,   # 语义标签: TrafficCone
    "cone": 0,          # 路径匹配
    
    # 树木 (class_id = 1)
    "tree": 1,          # 语义标签: Tree
    
    # 围栏 (class_id = 2)
    "fence": 2,         # 语义标签: Fence
    "fencing": 2,
    "construction_site": 2,
    
    # 吊车/起重机 - 整体 (class_id = 3)
    "crane": 3,         # 语义标签: Crane
    "pk7": 3,           # 路径匹配
    
    # 吊车部件 - 底座 (class_id = 6)
    "cranebase": 6,     # 语义标签: CraneBase
    
    # 吊车部件 - 立柱/转台 (class_id = 7)
    "cranecolumn": 7,   # 语义标签: CraneColumn
    
    # 吊车部件 - 主臂 (class_id = 8)
    "craneboom": 8,     # 语义标签: CraneBoom
    
    # 吊车部件 - 伸缩臂 (class_id = 9)
    "cranetelescopic": 9,  # 语义标签: CraneTelescopic
    
    # 卡车/自卸车 (class_id = 4)
    "dumper": 4,        # 语义标签: Dumper
    "09684481": 4,      # 路径匹配
    
    # 人物 (class_id = 5)
    "human": 5,         # 语义标签: Human
    "dhgen": 5,         # 路径匹配
    "skelroot": 5,
}

# 吊车部件路径映射（来自add_crane_part_labels.py的精确路径）
# 键: 吊车根下的一级子节点名(小写) -> (部件类名, 类别ID)
CRANE_PART_CHILD_MAP = {
    "s104gg03a_sw":           ("cranebase", 6),
    "s104s01kb_sw":           ("cranebase", 6),
    "s104hz01ka_sw":          ("cranecolumn", 7),
    "s104h01kb_sw":           ("cranecolumn", 7),
    "s104hz02ka_sw":          ("cranecolumn", 7),
    "s104kz01ka_sw":          ("cranecolumn", 7),
    "tn__s104ekb_as_sw_jj7":  ("craneboom", 8),
    "s104kz02ka_sw":          ("cranetelescopic", 9),
    "tn__hhk320ka_sw_lg":     ("cranetelescopic", 9),
    "tn__hhk319_sw_od":       ("cranetelescopic", 9),
}

# 运行时构建的完整吊车路径映射（由build_crane_part_map填充）
_crane_part_map = {}

# 物体根路径模式（用于将Mesh组件聚合成完整物体）
# 这些是场景中顶层物体的路径前缀
OBJECT_ROOT_PATTERNS = [
    # 围栏：/World/GroundPlane/Construction_Site_..._Fencing_height_XX
    "/World/GroundPlane/Construction_Site_Construction_Zeppelin_Rental_GmbH_Metal_Construction_Site_Fencing_height_",
    # 吊车：/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM
    "/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM",
    # 卡车：/World/GroundPlane/tn__09684481_
    "/World/GroundPlane/tn__09684481_",
    # 交通锥：/World/GroundPlane/Cone001_XX 或 /World/GroundPlane/Cone001
    "/World/GroundPlane/Cone001",
    # 人物：/World/GroundPlane/DHGen
    "/World/GroundPlane/DHGen",
    # 树木：/World/Tree/Tree_XX 或 /World/Tree/Tree
    "/World/Tree/Tree",
]


def get_object_root(prim_path):
    """
    从Mesh路径获取物体根路径
    例如: /World/GroundPlane/Cone001_01/Cone001 -> /World/GroundPlane/Cone001_01
    
    返回: (object_root_path, class_name, class_id) 或 (None, None, None)
    """
    prim_path_lower = prim_path.lower()
    
    # 特殊处理：围栏有编号后缀
    if "fencing_height_" in prim_path_lower:
        # 提取到 fencing_height_XX 为止
        parts = prim_path.split("/")
        for i, part in enumerate(parts):
            if "Fencing_height_" in part:
                root = "/".join(parts[:i+1])
                return root, "fence", construction_class.get("fence", 2)
    
    # 特殊处理：树木
    if "/world/tree/tree" in prim_path_lower:
        parts = prim_path.split("/")
        # /World/Tree/Tree_XX 或 /World/Tree/Tree
        if len(parts) >= 4:
            root = "/".join(parts[:4])  # /World/Tree/Tree 或 /World/Tree/Tree_01
            return root, "tree", construction_class.get("tree", 1)
    
    # 特殊处理：交通锥
    if "/cone001" in prim_path_lower:
        parts = prim_path.split("/")
        for i, part in enumerate(parts):
            if part.lower().startswith("cone001"):
                root = "/".join(parts[:i+1])
                return root, "trafficcone", construction_class.get("trafficcone", 0)
    
    # 特殊处理：吊车及其部件
    if "pk7501sld" in prim_path_lower or "pk7" in prim_path_lower:
        crane_root = "/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM"
        
        # 方法1: 使用运行时构建的完整映射表（最准确）
        if _crane_part_map and prim_path in _crane_part_map:
            part_name, class_id = _crane_part_map[prim_path]
            # 部件的聚合根路径 = crane_root + 部件类名
            part_root = crane_root + "#" + part_name
            return part_root, part_name, class_id
        
        # 方法2: 通过已知的一级子节点名精确匹配
        if prim_path.startswith(crane_root + "/") or prim_path_lower.startswith(crane_root.lower() + "/"):
            remaining = prim_path[len(crane_root) + 1:]
            first_segment = remaining.split("/")[0]
            first_segment_lower = first_segment.lower()
            
            if first_segment_lower in CRANE_PART_CHILD_MAP:
                part_name, class_id = CRANE_PART_CHILD_MAP[first_segment_lower]
                part_root = crane_root + "#" + part_name
                return part_root, part_name, class_id
        
        # 方法3: 关键词匹配（兜底）
        sub_path = prim_path_lower[prim_path_lower.find("pk7"):]
        base_kw = ["base", "chassis", "footer", "support", "grund", "fahrwerk"]
        column_kw = ["column", "turret", "mast", "tower", "saeule", "drehwerk", "oberwagen"]
        boom_kw = ["boom", "arm", "jib", "ausleger"]
        telescopic_kw = ["telescop", "extension", "teleskop", "auszug"]
        
        if any(kw in sub_path for kw in base_kw):
            return crane_root + "#cranebase", "cranebase", construction_class.get("cranebase", 6)
        elif any(kw in sub_path for kw in column_kw):
            return crane_root + "#cranecolumn", "cranecolumn", construction_class.get("cranecolumn", 7)
        elif any(kw in sub_path for kw in boom_kw):
            return crane_root + "#craneboom", "craneboom", construction_class.get("craneboom", 8)
        elif any(kw in sub_path for kw in telescopic_kw):
            return crane_root + "#cranetelescopic", "cranetelescopic", construction_class.get("cranetelescopic", 9)
        
        # 默认返回整体吊车
        return crane_root, "crane", construction_class.get("crane", 3)
    
    # 特殊处理：卡车
    if "09684481" in prim_path_lower:
        return "/World/GroundPlane/tn__09684481_", "dumper", construction_class.get("dumper", 4)
    
    # 特殊处理：人物
    if "dhgen" in prim_path_lower:
        return "/World/GroundPlane/DHGen", "human", construction_class.get("human", 5)
    
    # 默认：尝试根据类别关键词匹配
    for key, class_id in construction_class.items():
        if key in prim_path_lower:
            # 返回原始路径作为根（可能不理想但至少能工作）
            return prim_path, key, class_id
    
    return None, None, None

"""========== 日志记录类 =========="""

class DataQualityLogger:
    """数据质量日志记录器"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.frame_logs = []
        self.statistics = {
            "total_frames_attempted": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "retry_count": 0,
            "pointcloud_stats": {"valid": 0, "empty": 0, "insufficient": 0},
            "rgb_stats": {"valid": 0, "failed": 0},
            "depth_stats": {"valid": 0, "failed": 0, "all_zero": 0, "all_inf": 0},
            "label_stats": {"valid": 0, "empty": 0},
            "object_count": {"total": 0, "per_frame_avg": 0}
        }
        
        # 创建详细日志文件
        timestamp = Path(log_dir).parent.name
        self.detail_log_path = f"{log_dir}/generation_detail.log"
        self.summary_log_path = f"{log_dir}/generation_summary.json"
        
        with open(self.detail_log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 数据生成详细日志 ===\n")
            f.write(f"开始时间: {timestamp}\n\n")
    
    def log_frame_start(self, frame_id, cam_pos):
        """记录帧开始"""
        msg = f"\n{'='*60}\n帧 {frame_id} 开始采集\n相机位置: {cam_pos}\n"
        self._write_log(msg)
        
        self.current_frame = {
            "frame_id": frame_id,
            "camera_position": cam_pos.tolist() if hasattr(cam_pos, 'tolist') else cam_pos,
            "retry_count": 0,
            "status": "processing",
            "issues": []
        }
    
    def log_retry(self, retry_count):
        """记录重试"""
        self.current_frame["retry_count"] = retry_count
        self.statistics["retry_count"] += 1
        msg = f"  ⚠ 重试 {retry_count} 次\n"
        self._write_log(msg)
    
    def log_pointcloud(self, valid, point_count=0, reason=""):
        """记录点云状态"""
        if valid:
            self.statistics["pointcloud_stats"]["valid"] += 1
            self.current_frame["pointcloud"] = {"status": "valid", "points": point_count}
            msg = f"  ✓ 点云: {point_count} 个点\n"
        else:
            if point_count == 0:
                self.statistics["pointcloud_stats"]["empty"] += 1
                self.current_frame["issues"].append(f"点云为空: {reason}")
                msg = f"  ✗ 点云为空: {reason}\n"
            else:
                self.statistics["pointcloud_stats"]["insufficient"] += 1
                self.current_frame["issues"].append(f"点云不足: {point_count} 点")
                msg = f"  ✗ 点云不足: {point_count} 点 ({reason})\n"
        self._write_log(msg)
    
    def log_rgb(self, valid, reason=""):
        """记录RGB图像状态"""
        if valid:
            self.statistics["rgb_stats"]["valid"] += 1
            self.current_frame["rgb"] = {"status": "valid"}
            msg = f"  ✓ RGB图像采集成功\n"
        else:
            self.statistics["rgb_stats"]["failed"] += 1
            self.current_frame["issues"].append(f"RGB失败: {reason}")
            msg = f"  ✗ RGB图像失败: {reason}\n"
        self._write_log(msg)
    
    def log_depth(self, valid, depth_data=None, reason=""):
        """记录深度图状态"""
        if valid and depth_data is not None:
            # 检查深度数据质量
            valid_pixels = np.sum(np.isfinite(depth_data) & (depth_data > 0))
            total_pixels = depth_data.size
            zero_pixels = np.sum(depth_data == 0)
            inf_pixels = np.sum(np.isinf(depth_data))
            
            # 修复：检查是否有有效数据
            valid_depth = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
            if len(valid_depth) > 0:
                depth_min = np.min(valid_depth)
                depth_max = np.max(valid_depth)
                depth_mean = np.mean(valid_depth)
            else:
                depth_min = depth_max = depth_mean = 0.0
            
            self.current_frame["depth"] = {
                "status": "valid",
                "valid_pixels": int(valid_pixels),
                "total_pixels": int(total_pixels),
                "valid_ratio": float(valid_pixels / total_pixels),
                "zero_pixels": int(zero_pixels),
                "inf_pixels": int(inf_pixels),
                "depth_range": [float(depth_min), float(depth_max)],
                "depth_mean": float(depth_mean)
            }
            
            if zero_pixels == total_pixels:
                self.statistics["depth_stats"]["all_zero"] += 1
                self.current_frame["issues"].append("深度图全为零")
                msg = f"  ⚠ 深度图: 全为零值！\n"
            elif inf_pixels == total_pixels:
                self.statistics["depth_stats"]["all_inf"] += 1
                self.current_frame["issues"].append("深度图全为无穷")
                msg = f"  ⚠ 深度图: 全为无穷值！\n"
            else:
                self.statistics["depth_stats"]["valid"] += 1
                msg = f"  ✓ 深度图: 有效像素 {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)\n"
                msg += f"    深度范围: [{depth_min:.2f}, {depth_max:.2f}] 平均: {depth_mean:.2f}\n"
        else:
            self.statistics["depth_stats"]["failed"] += 1
            self.current_frame["issues"].append(f"深度图失败: {reason}")
            msg = f"  ✗ 深度图失败: {reason}\n"
        self._write_log(msg)
    
    def log_labels(self, object_count):
        """记录标签状态"""
        if object_count > 0:
            self.statistics["label_stats"]["valid"] += 1
            self.statistics["object_count"]["total"] += object_count
            self.current_frame["labels"] = {"status": "valid", "object_count": object_count}
            msg = f"  ✓ 标签: {object_count} 个物体\n"
        else:
            self.statistics["label_stats"]["empty"] += 1
            self.current_frame["issues"].append("未识别到物体")
            msg = f"  ⚠ 标签: 0 个物体（可能视野外或未匹配类别）\n"
        self._write_log(msg)
    
    def log_frame_end(self, success):
        """记录帧结束"""
        self.statistics["total_frames_attempted"] += 1
        if success:
            self.statistics["successful_frames"] += 1
            self.current_frame["status"] = "success"
            msg = f">>> 帧 {self.current_frame['frame_id']} 完成 ✓\n"
        else:
            self.statistics["failed_frames"] += 1
            self.current_frame["status"] = "failed"
            msg = f">>> 帧 {self.current_frame['frame_id']} 失败 ✗\n"
        
        self._write_log(msg)
        self.frame_logs.append(self.current_frame.copy())
    
    def save_summary(self):
        """保存汇总报告"""
        # 计算平均值
        if self.statistics["successful_frames"] > 0:
            self.statistics["object_count"]["per_frame_avg"] = \
                self.statistics["object_count"]["total"] / self.statistics["successful_frames"]
        
        # 成功率
        self.statistics["success_rate"] = \
            self.statistics["successful_frames"] / max(1, self.statistics["total_frames_attempted"])
        
        # 保存JSON汇总
        summary_data = {
            "statistics": self.statistics,
            "frame_logs": self.frame_logs
        }
        
        with open(self.summary_log_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        report = self._generate_report()
        with open(self.detail_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*60}\n")
            f.write(report)
        
        print(f"\n日志已保存:")
        print(f"  详细日志: {self.detail_log_path}")
        print(f"  汇总JSON: {self.summary_log_path}")
        return report
    
    def _generate_report(self):
        """生成可读报告"""
        stats = self.statistics
        report = "=== 数据生成汇总报告 ===\n\n"
        
        report += f"总体统计:\n"
        report += f"  尝试帧数: {stats['total_frames_attempted']}\n"
        report += f"  成功帧数: {stats['successful_frames']}\n"
        report += f"  失败帧数: {stats['failed_frames']}\n"
        report += f"  成功率: {stats['success_rate']*100:.1f}%\n"
        report += f"  总重试次数: {stats['retry_count']}\n\n"
        
        report += f"点云质量:\n"
        report += f"  有效: {stats['pointcloud_stats']['valid']}\n"
        report += f"  为空: {stats['pointcloud_stats']['empty']}\n"
        report += f"  不足: {stats['pointcloud_stats']['insufficient']}\n\n"
        
        report += f"RGB图像:\n"
        report += f"  成功: {stats['rgb_stats']['valid']}\n"
        report += f"  失败: {stats['rgb_stats']['failed']}\n\n"
        
        report += f"深度图:\n"
        report += f"  有效: {stats['depth_stats']['valid']}\n"
        report += f"  失败: {stats['depth_stats']['failed']}\n"
        report += f"  全零: {stats['depth_stats']['all_zero']}\n"
        report += f"  全无穷: {stats['depth_stats']['all_inf']}\n\n"
        
        report += f"标签识别:\n"
        report += f"  有效: {stats['label_stats']['valid']}\n"
        report += f"  为空: {stats['label_stats']['empty']}\n"
        report += f"  总物体数: {stats['object_count']['total']}\n"
        report += f"  平均每帧: {stats['object_count']['per_frame_avg']:.2f}\n\n"
        
        # 常见问题分析
        report += f"常见问题:\n"
        issue_count = {}
        for frame in self.frame_logs:
            for issue in frame.get("issues", []):
                issue_type = issue.split(":")[0]
                issue_count[issue_type] = issue_count.get(issue_type, 0) + 1
        
        for issue_type, count in sorted(issue_count.items(), key=lambda x: x[1], reverse=True):
            report += f"  {issue_type}: {count} 次\n"
        
        return report
    
    def _write_log(self, msg):
        """写入日志文件"""
        with open(self.detail_log_path, 'a', encoding='utf-8') as f:
            f.write(msg)
        print(msg, end='')


"""========== 工具函数 =========="""

def rotMtx2quaternion(R):
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    """
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def camPosOri(target_point, aimed_point):
    """
    计算相机朝向目标点的姿态（四元数）
    强制相机始终保持正向（up vector = [0, 0, 1]）
    Input: 
        target_point: 相机位置 (3,)
        aimed_point: 目标瞄准点 (3,)
    Output: 
        q: 四元数 (w, x, y, z)
    """
    # 计算相机的forward方向（指向目标）
    forward = aimed_point - target_point
    forward = forward / np.linalg.norm(forward)
    
    # 世界坐标系的up方向
    world_up = np.array([0.0, 0.0, 1.0])
    
    # 计算相机的right方向（forward × world_up）
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    
    # 如果forward和world_up几乎平行，使用备用方案
    if right_norm < 1e-6:
        # 相机几乎垂直向上或向下，使用备用轴
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    
    # 重新计算相机的up方向（right × forward）
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # 构建旋转矩阵（相机坐标系 -> 世界坐标系）
    # Isaac Sim相机默认朝向-X轴，up是+Z轴
    # 世界坐标系: forward对应相机-X, right对应相机-Y, up对应相机+Z
    R_camera_to_world = np.array([
        [-forward[0], -right[0], up[0]],
        [-forward[1], -right[1], up[1]],
        [-forward[2], -right[2], up[2]]
    ])
    
    # 转换为四元数
    q = rotMtx2quaternion(R_camera_to_world)
    return q


def bboxDict_to_transform(bbox_dict):
    """
    从3D边界框数据中提取位姿信息
    Input: bbox_dict - bounding_box_3d annotator返回的单个bbox数据
    Output: 
        center_world: 中心位置 (x, y, z)
        size_world: 尺寸 (width, height, depth)
        euler_angle: 欧拉角 (roll, pitch, yaw) 单位：度
    """
    corner = np.array([[bbox_dict[1], bbox_dict[2], bbox_dict[3]],
                      [bbox_dict[4], bbox_dict[5], bbox_dict[6]]])
    trans_mtx = bbox_dict[7]
    
    center_local = np.mean(corner, axis=0)
    center_local_1 = np.append(center_local, 1.0)
    trans_mtx_T = trans_mtx.reshape(4, 4).T
    center_world = trans_mtx_T @ center_local_1
    center_world = center_world[:3].tolist()
    
    rot_mtx = trans_mtx_T[:3, :3]
    U, _, Vt = np.linalg.svd(rot_mtx)
    rot_mtx_pure = np.dot(U, Vt)
    r = R.from_matrix(rot_mtx_pure)
    euler_angle = r.as_euler('xyz', degrees=True)
    
    scale = np.array([np.linalg.norm(rot_mtx[:, 0]), 
                     np.linalg.norm(rot_mtx[:, 1]), 
                     np.linalg.norm(rot_mtx[:, 2])])
    size_local = np.abs(corner[1] - corner[0]).tolist()
    size_world = scale * size_local
    
    return center_world, size_world.tolist(), euler_angle.tolist()


def get_obj_pose(stage, prim_path):
    """
    获取物体的世界坐标位姿
    Output: [x, y, z, qx, qy, qz, qw]
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' not found.")
    
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    translation = matrix.ExtractTranslation()
    rotation_matrix = matrix.ExtractRotationMatrix()
    rot_np = np.array(rotation_matrix.GetTranspose())
    quat = R.from_matrix(rot_np).as_quat()  # [x, y, z, w]
    
    return [translation[0], translation[1], translation[2], 
            quat[0], quat[1], quat[2], quat[3]]


def save_label_json(label_dict, filename):
    """
    保存标注数据为JSON格式
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)


def depth_to_pointcloud_with_rgb(depth_data, rgb_image, camera_params, camera_pose):
    """
    从深度图和RGB图像生成点云（后备方案）
    
    当pointcloud annotator返回空数据时使用此方法
    
    参数:
        depth_data: 深度图 (H, W)
        rgb_image: RGB图像 (H, W, 3) 或 (H, W, 4)
        camera_params: 相机参数字典，包含 horizontal_aperture, vertical_aperture, focal_length, width, height
        camera_pose: 相机位姿 (7,) [x,y,z,qx,qy,qz,qw]
    
    返回:
        点云数组 (N, 6) [x, y, z, r, g, b] 或 None
    """
    try:
        h, w = depth_data.shape
        
        # 提取相机内参
        # 从相机参数计算内参矩阵
        focal_length = camera_params.get('focal_length', 18.14)
        horizontal_aperture = camera_params.get('horizontal_aperture', 20.955)
        vertical_aperture = camera_params.get('vertical_aperture', 15.2908)
        img_width = camera_params.get('width', w)
        img_height = camera_params.get('height', h)
        
        # 计算焦距（像素单位）
        # fx = (focal_length / horizontal_aperture) * img_width
        # fy = (focal_length / vertical_aperture) * img_height
        # 更准确的计算方式
        fx = (img_width * focal_length) / horizontal_aperture
        fy = (img_height * focal_length) / vertical_aperture
        cx = img_width / 2.0
        cy = img_height / 2.0
        
        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 过滤有效深度值
        valid_mask = np.isfinite(depth_data) & (depth_data > 0) & (depth_data < 250)
        
        if not valid_mask.any():
            print(f"  ⚠ 无有效深度值用于生成点云")
            return None
        
        # 计算相机坐标系下的3D坐标（标准pinhole模型）
        # 在pinhole模型中：u = fx * X/Z + cx, v = fy * Y/Z + cy
        # 所以：X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        z_camera = depth_data[valid_mask]  # 深度值（相机坐标系Z）
        x_camera = (u[valid_mask] - cx) * z_camera / fx
        y_camera = (v[valid_mask] - cy) * z_camera / fy
        
        # 标准pinhole相机坐标系：X右，Y下，Z前（朝向+Z）
        # Isaac Sim相机坐标系：相机朝向-X，up是+Z
        # 需要从pinhole坐标系转换到Isaac Sim相机坐标系
        # pinhole (X右, Y下, Z前) -> Isaac Sim (X前, Y右, Z上)
        # 转换：Isaac_X = -pinhole_Z, Isaac_Y = -pinhole_X, Isaac_Z = pinhole_Y
        # 但更简单的方法：直接使用pinhole坐标，然后通过旋转矩阵转换到世界坐标
        points_camera_pinhole = np.stack([x_camera, y_camera, z_camera], axis=-1)
        
        # 转换到世界坐标系
        position = np.array(camera_pose[:3])
        quaternion = np.array(camera_pose[3:])  # [qx, qy, qz, qw]
        
        # 四元数转旋转矩阵（从相机坐标系到世界坐标系）
        rotation_camera_to_world = R.from_quat(quaternion).as_matrix()
        
        # 应用变换：points_world = R @ points_camera.T + position
        # 注意：这里假设camera_pose的旋转矩阵已经是从相机到世界的变换
        points_world = (rotation_camera_to_world @ points_camera_pinhole.T).T + position
        
        # 获取对应的RGB值
        if rgb_image is not None and rgb_image.size > 0:
            # 确保RGB图像是3通道
            if rgb_image.shape[2] >= 3:
                rgb = rgb_image[valid_mask, :3]
                # 如果RGB值在[0,1]范围，转换为[0,255]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            else:
                rgb = np.ones((points_world.shape[0], 3), dtype=np.uint8) * 255
        else:
            rgb = np.ones((points_world.shape[0], 3), dtype=np.uint8) * 255
        
        # 合并为 (N, 6)
        xyzrgb = np.hstack([points_world, rgb])
        
        return xyzrgb
        
    except Exception as e:
        print(f"  ✗ 从深度图生成点云失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_pointcloud_with_rgb(pcd_data, filename):
    """
    保存点云数据（XYZ + RGB）
    格式: x y z r g b (每行一个点)
    """
    try:
        xyz = pcd_data['data']  # (N, 3)
        
        # 检查并修正xyz数据维度
        if xyz is None or len(xyz) == 0:
            print(f"  ⚠ 警告: 点云XYZ数据为空，跳过保存")
            return
        
        if xyz.ndim == 1:
            if len(xyz) == 3:
                xyz = xyz.reshape(1, 3)
            else:
                print(f"  ⚠ 警告: XYZ数据格式异常（形状={xyz.shape}），跳过保存")
                return
        
        # 处理RGB数据，兼容不同的数据格式
        if 'pointRgb' in pcd_data and pcd_data['pointRgb'] is not None:
            rgb_data = pcd_data['pointRgb']
            
            # 确保RGB数据是2D数组
            if rgb_data.ndim == 1:
                # 如果是1维数组，可能是单个点或空数据
                if len(rgb_data) == 0:
                    print(f"  ⚠ 警告: RGB数据为空，使用默认白色")
                    rgb = np.ones((xyz.shape[0], 3)) * 255
                elif len(rgb_data) == 3 or len(rgb_data) == 4:
                    # 单个点的情况
                    rgb = rgb_data[:3].reshape(1, 3)
                else:
                    print(f"  ⚠ 警告: RGB数据格式异常，形状={rgb_data.shape}，使用默认白色")
                    rgb = np.ones((xyz.shape[0], 3)) * 255
            else:
                # 2D数组，取前3列（RGB，丢弃Alpha）
                rgb = rgb_data[:, :3] if rgb_data.shape[1] >= 3 else rgb_data
        else:
            print(f"  ⚠ 警告: 无RGB数据，使用默认白色")
            rgb = np.ones((xyz.shape[0], 3)) * 255
        
        # 确保xyz和rgb的行数一致
        if xyz.shape[0] != rgb.shape[0]:
            print(f"  ⚠ 警告: XYZ和RGB点数不匹配 ({xyz.shape[0]} vs {rgb.shape[0]})")
            # 使用较小的数量
            min_points = min(xyz.shape[0], rgb.shape[0])
            xyz = xyz[:min_points]
            rgb = rgb[:min_points]
        
        # 合并为 (N, 6)
        xyzrgb = np.hstack([xyz, rgb])
        
        # 保存为文本文件
        np.savetxt(filename, xyzrgb, fmt='%.6f', delimiter=' ', 
                   header='x y z r g b', comments='')
        
    except Exception as e:
        print(f"  ✗ 保存点云异常: {e}")
        import traceback
        traceback.print_exc()


def get_systematic_camera_positions(num_frames=20):
    """
    网格化采样策略：在场景内均匀分布相机
    
    策略：
    1. 首先使用预定义的关键位置
    2. 如果需要更多帧，使用环形采样生成额外位置
    3. 确保覆盖整个场景
    
    返回：[(cam_pos, target_pos), ...]
    """
    positions = []
    heights = [1.6, 1.7, 1.8, 2.0, 2.5, 3.0]  # 扩展高度范围
    
    # 预定义的关键位置（覆盖场景的重要视角）
    # 铲车位置: [-7.37, -0.59, 0.69]
    dumper_center = [-7.37, -0.59]
    
    key_positions = [
        # ===== 拖车/铲车 专用视角 (40%占比，提升出现频率) =====
        ([-15, -0.6], dumper_center),    # 拖车左侧远距
        ([-2, -0.6], dumper_center),     # 拖车右侧近距
        ([-7.4, 6], dumper_center),      # 拖车前方
        ([-7.4, -7], dumper_center),     # 拖车后方
        ([-12, 4], dumper_center),       # 拖车左前远距
        ([-12, -5], dumper_center),      # 拖车左后远距
        ([-4, 4], dumper_center),        # 拖车右前近距
        ([-4, -4], dumper_center),       # 拖车右后近距
        ([-10, 0], dumper_center),       # 拖车正左侧
        ([-5, 2], dumper_center),        # 拖车右前45°
        ([-5, -3], dumper_center),       # 拖车右后45°
        ([-9, -4], dumper_center),       # 拖车左后45°
        
        # ===== 中心区域（物体最密集）=====
        ([-3, -3], [0, 0]),
        ([-3, 3], [0, 0]),
        ([0, 0], [5, 0]),
        ([0, 0], [-5, 0]),
        
        # ===== 环绕中心 =====
        ([6, 0], [0, 0]),
        ([0, 6], [0, 0]),
        ([0, -6], [0, 0]),
        ([-6, 0], [0, 0]),
        
        # ===== 对角线 =====
        ([5, 5], [0, 0]),
        ([5, -5], [0, 0]),
        ([-5, 5], [0, 0]),
        ([-5, -5], [0, 0]),
        
        # ===== 近距离 =====
        ([3, 0], [0, 0]),
        ([-3, 0], [0, 0]),
        ([0, 3], [0, 0]),
        ([0, -3], [0, 0]),
        
        # ===== 左侧区域 =====
        ([-8, -3], [0, 0]),
        ([-8, 3], [0, 0]),
    ]
    
    frame_count = 0
    
    # 首先添加预定义位置
    for cam_xy, target_xy in key_positions:
        if frame_count >= num_frames:
            break
        
        cam_z = heights[frame_count % len(heights)]
        cam_position = np.array([cam_xy[0], cam_xy[1], cam_z])
        target_point = np.array([target_xy[0], target_xy[1], cam_z])
        
        positions.append((cam_position, target_point))
        frame_count += 1
    
    # 如果需要更多位置，使用混合采样策略
    if frame_count < num_frames:
        # 不同半径的环
        radii = [4, 6, 8, 10, 12]
        points_per_ring = 8
        
        for radius in radii:
            for i in range(points_per_ring):
                if frame_count >= num_frames:
                    break
                
                angle = 2 * np.pi * i / points_per_ring
                cam_x = radius * np.cos(angle)
                cam_y = radius * np.sin(angle)
                cam_z = heights[frame_count % len(heights)]
                
                cam_position = np.array([cam_x, cam_y, cam_z])
                
                # 40% 概率看向拖车区域，60% 看向中心
                if np.random.random() < 0.4:
                    target_x = dumper_center[0] + np.random.uniform(-2, 2)
                    target_y = dumper_center[1] + np.random.uniform(-2, 2)
                    target_point = np.array([target_x, target_y, cam_z])
                else:
                    target_point = np.array([0, 0, cam_z])
                
                positions.append((cam_position, target_point))
                frame_count += 1
            
            if frame_count >= num_frames:
                break
    
    # 如果还需要更多，添加随机采样（偏向拖车区域）
    while frame_count < num_frames:
        cam_z = heights[frame_count % len(heights)]
        
        # 50% 概率生成拖车附近的相机位置
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(5, 12)
            cam_x = dumper_center[0] + dist * np.cos(angle)
            cam_y = dumper_center[1] + dist * np.sin(angle)
            target_x = dumper_center[0] + np.random.uniform(-1, 1)
            target_y = dumper_center[1] + np.random.uniform(-1, 1)
        else:
            cam_x = np.random.uniform(-10, 8)
            cam_y = np.random.uniform(-10, 10)
            target_x = np.random.uniform(-3, 3)
            target_y = np.random.uniform(-3, 3)
        
        cam_position = np.array([cam_x, cam_y, cam_z])
        target_point = np.array([target_x, target_y, cam_z])
        
        positions.append((cam_position, target_point))
        frame_count += 1
    
    print(f"[调试] 实际生成了 {len(positions)} 个相机位置（请求: {num_frames}）")
    return positions


def randomize_object_positions(stage):
    """
    随机改变场景中可移动物体的位置和姿态（改进版v2）
    
    参数:
        stage: USD Stage对象
    
    改进:
        1. 先放最大物体（吊车），再放其他物体
        2. 碰撞检测使用**半径之和**（而非max），确保不重叠
        3. 吊车使用实际BBox计算碰撞半径（吊臂可达7-8m）
        4. 交通锥每个单独放置
        5. 所有物体保持在原始地面高度
    """
    from pxr import UsdGeom, Gf
    
    # ===== 碰撞检测工具函数 =====
    placed_positions = []  # [(x, y, occupy_radius), ...]
    
    def check_no_overlap(x, y, own_radius):
        """
        检查新位置是否与所有已放置物体保持足够距离
        关键修复: 使用两个物体半径之和 (own_radius + pr) 而非 max()
        """
        for px, py, pr in placed_positions:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            required_dist = own_radius + pr  # 两物体半径之和
            if dist < required_dist:
                return False
        return True
    
    def find_valid_position(center_x, center_y, range_x, range_y, own_radius, max_attempts=80):
        """在指定范围内寻找不重叠的随机位置"""
        for _ in range(max_attempts):
            x = np.random.uniform(center_x - range_x, center_x + range_x)
            y = np.random.uniform(center_y - range_y, center_y + range_y)
            if check_no_overlap(x, y, own_radius):
                return x, y, True
        # 找不到则返回中心附近随机位置（标记失败）
        return center_x + np.random.uniform(-1, 1), center_y + np.random.uniform(-1, 1), False
    
    def compute_prim_xy_radius(prim, default=3.0):
        """
        根据BBox计算prim在XY平面上的碰撞半径
        考虑吊车吊臂等不对称结构
        """
        try:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            min_pt = bbox_range.GetMin()
            max_pt = bbox_range.GetMax()
            # XY平面对角线的一半 = 碰撞半径
            dx = (max_pt[0] - min_pt[0]) / 2.0
            dy = (max_pt[1] - min_pt[1]) / 2.0
            radius = np.sqrt(dx**2 + dy**2)
            return max(radius * 0.9, 1.0)  # 90%的对角线，至少1m
        except:
            return default
    
    def set_prim_transform(prim, new_x, new_y, new_z, rotation_deg=None):
        """设置prim的位移和绕Z轴旋转"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return False
        
        translate_op = None
        rotate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_op = op
        
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(new_x, new_y, new_z))
        
        if rotation_deg is not None:
            if rotate_op is None:
                rotate_op = xformable.AddRotateZOp()
            rotate_op.Set(float(rotation_deg))
        
        return True
    
    def get_prim_z(prim):
        """获取prim当前的Z坐标（地面高度）"""
        xformable = UsdGeom.Xformable(prim)
        try:
            transform = xformable.GetLocalTransformation()
            return float(transform.ExtractTranslation()[2])
        except:
            return 0.0
    
    def find_prims_by_prefix(parent_path, prefix):
        """查找指定父路径下所有以prefix开头的子prim"""
        parent = stage.GetPrimAtPath(parent_path)
        result = []
        if parent and parent.IsValid():
            for child in parent.GetChildren():
                if child.GetName().startswith(prefix):
                    result.append(child)
        return result
    
    randomized_objects = []
    gp_path = "/World/GroundPlane"
    
    # ===== 1. 吊车 - 最先放置（最大物体，吊臂跨度大） =====
    crane_prims = find_prims_by_prefix(gp_path, "tn__Pk7501SLD")
    for crane_prim in crane_prims:
        orig_z = get_prim_z(crane_prim)
        
        # 计算吊车实际碰撞半径（吊臂跨度，通常7-8m）
        crane_radius = compute_prim_xy_radius(crane_prim, default=7.0)
        # 吊车碰撞半径至少6m（吊臂很长）
        crane_radius = max(crane_radius, 6.0)
        
        print(f"    [碰撞] 吊车BBox碰撞半径: {crane_radius:.1f}m")
        
        # 吊车在中心小范围移动，不旋转
        new_x, new_y, ok = find_valid_position(0, 0, 2.0, 2.0, crane_radius)
        
        if set_prim_transform(crane_prim, new_x, new_y, orig_z):
            placed_positions.append((new_x, new_y, crane_radius))
            randomized_objects.append({
                'path': str(crane_prim.GetPath()),
                'new_pos': [new_x, new_y, orig_z],
                'rotation': None,
                'type': 'crane',
                'no_overlap': ok
            })
    
    # ===== 2. 拖车/铲车 (dumper) - 放在与吊车不重叠的位置 =====
    dumper_areas = [
        (-7, -1),    # 原始位置附近
        (-3, -5),    # 下方
        (5, 0),      # 右侧
        (-5, 5),     # 左前方
        (3, -4),     # 右下方
        (6, 3),      # 右前方
        (-6, -4),    # 左下方
    ]
    
    dumper_prims = find_prims_by_prefix(gp_path, "tn__09684481")
    for dumper_prim in dumper_prims:
        orig_z = get_prim_z(dumper_prim)
        
        # 计算铲车实际碰撞半径（通常2-3m）
        dumper_radius = compute_prim_xy_radius(dumper_prim, default=3.0)
        dumper_radius = max(dumper_radius, 2.5)
        
        print(f"    [碰撞] 铲车BBox碰撞半径: {dumper_radius:.1f}m")
        
        # 从多个候选区域中尝试找到合法位置
        best_pos = None
        for area in np.random.permutation(len(dumper_areas)):
            ax, ay = dumper_areas[int(area)]
            new_x, new_y, ok = find_valid_position(ax, ay, 2.0, 2.0, dumper_radius)
            if ok:
                best_pos = (new_x, new_y, ok)
                break
        
        if best_pos is None:
            # 所有候选区域都找不到，用最远的区域强制放
            ax, ay = dumper_areas[0]
            new_x, new_y, ok = find_valid_position(ax, ay, 3.0, 3.0, dumper_radius)
            best_pos = (new_x, new_y, ok)
        
        new_x, new_y, ok = best_pos
        rotation = np.random.uniform(-180, 180)
        
        if set_prim_transform(dumper_prim, new_x, new_y, orig_z, rotation):
            placed_positions.append((new_x, new_y, dumper_radius))
            randomized_objects.append({
                'path': str(dumper_prim.GetPath()),
                'new_pos': [new_x, new_y, orig_z],
                'rotation': rotation,
                'type': 'dumper',
                'no_overlap': ok
            })
    
    # ===== 3. 人物 - 中等范围移动 =====
    human_prims = find_prims_by_prefix(gp_path, "DHGen")
    human_radius = 0.8  # 人体半径约0.8m
    for human_prim in human_prims:
        orig_z = get_prim_z(human_prim)
        new_x, new_y, ok = find_valid_position(
            np.random.uniform(-5, 5), np.random.uniform(-5, 5),
            3.0, 3.0, human_radius
        )
        rotation = np.random.uniform(-180, 180)
        
        if set_prim_transform(human_prim, new_x, new_y, orig_z, rotation):
            placed_positions.append((new_x, new_y, human_radius))
            randomized_objects.append({
                'path': str(human_prim.GetPath()),
                'new_pos': [new_x, new_y, orig_z],
                'rotation': rotation,
                'type': 'human',
                'no_overlap': ok
            })
    
    # ===== 4. 交通锥 - 每个单独放置，分散在场景中 =====
    cone_prims = find_prims_by_prefix(gp_path, "Cone001")
    cone_radius = 0.5  # 锥筒半径约0.5m
    
    for cone_prim in cone_prims:
        orig_z = get_prim_z(cone_prim)
        cx = np.random.uniform(-7, 7)
        cy = np.random.uniform(-7, 7)
        new_x, new_y, ok = find_valid_position(cx, cy, 2.0, 2.0, cone_radius)
        rotation = np.random.uniform(-180, 180)
        
        if set_prim_transform(cone_prim, new_x, new_y, orig_z, rotation):
            placed_positions.append((new_x, new_y, cone_radius))
            randomized_objects.append({
                'path': str(cone_prim.GetPath()),
                'new_pos': [new_x, new_y, orig_z],
                'rotation': rotation,
                'type': 'trafficcone',
                'no_overlap': ok
            })
    
    # 统计碰撞情况
    overlap_count = sum(1 for obj in randomized_objects if not obj.get('no_overlap', True))
    if overlap_count > 0:
        print(f"  ⚠ {overlap_count} 个物体未能完全避免重叠")
    else:
        print(f"  ✓ 所有 {len(randomized_objects)} 个物体无碰撞")
    
    return randomized_objects


def build_crane_part_map(stage):
    """
    扫描吊车层级结构，建立完整的 prim路径 -> (部件名, 类别ID) 映射
    在初始化阶段调用一次，后续 get_object_root() 可直接查表
    """
    global _crane_part_map
    _crane_part_map = {}
    
    crane_root_path = "/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM"
    crane_prim = stage.GetPrimAtPath(crane_root_path)
    if not crane_prim or not crane_prim.IsValid():
        print(f"  ⚠ 吊车根节点不存在: {crane_root_path}")
        return _crane_part_map
    
    # 遍历一级子节点
    first_level = list(crane_prim.GetChildren())
    print(f"  [吊车部件映射] 一级子节点: {len(first_level)} 个")
    
    mapped_parts = {"cranebase": 0, "cranecolumn": 0, "craneboom": 0, "cranetelescopic": 0, "crane": 0}
    
    for child in first_level:
        child_name_lower = child.GetName().lower()
        child_path = str(child.GetPath())
        
        # 使用已知的精确映射
        if child_name_lower in CRANE_PART_CHILD_MAP:
            part_name, class_id = CRANE_PART_CHILD_MAP[child_name_lower]
        else:
            # 未知子节点归为整体吊车
            part_name, class_id = "crane", 3
        
        # 映射此子节点及所有后代
        _crane_part_map[child_path] = (part_name, class_id)
        for desc in Usd.PrimRange(child):
            _crane_part_map[str(desc.GetPath())] = (part_name, class_id)
        
        mapped_parts[part_name] = mapped_parts.get(part_name, 0) + 1
    
    # 打印映射统计
    print(f"  [吊车部件映射] 映射结果:")
    for pname, count in mapped_parts.items():
        if count > 0:
            print(f"    {pname}: {count} 个一级子节点")
    print(f"  [吊车部件映射] 共映射 {len(_crane_part_map)} 个prim路径")
    
    return _crane_part_map


def fix_scene_materials(stage):
    """
    修复树木和交通锥的材质 - 使用OmniPBR（MDL）材质
    
    UsdPreviewSurface在Isaac Sim RTX渲染器中不能正确显示,
    必须使用OmniPBR (MDL shader) 才能在RTX模式下正确渲染颜色。
    
    只在初始化时调用一次。
    """
    from pxr import UsdShade, Gf, Sdf
    
    # 获取场景路径，计算纹理目录
    scene_path = stage.GetRootLayer().identifier
    scene_dir = os.path.dirname(scene_path)         # cad_models/
    project_dir = os.path.dirname(scene_dir)          # 项目根目录
    textures_dir = os.path.join(project_dir, "textures")
    
    texture_leaves_path = os.path.join(textures_dir, "Branches0018_1_S.png")
    texture_bark_path = os.path.join(textures_dir, "BarkDecidious0107_M.jpg")
    texture_cone_path = os.path.join(textures_dir, "Traffic Cone UV Fixed.png")
    
    print(f"  纹理目录: {textures_dir}")
    print(f"  树叶纹理: {'✓' if os.path.exists(texture_leaves_path) else '✗'}")
    print(f"  树皮纹理: {'✓' if os.path.exists(texture_bark_path) else '✗'}")
    print(f"  交通锥纹理: {'✓' if os.path.exists(texture_cone_path) else '✗'}")
    
    # 确保 /World/Looks 存在
    if not stage.GetPrimAtPath("/World/Looks"):
        stage.DefinePrim("/World/Looks", "Scope")
    
    def create_omnipbr_material(mat_path, diffuse_color, texture_path=None, roughness=0.5):
        """
        创建OmniPBR (MDL) 材质 - Isaac Sim RTX渲染器原生支持
        
        参数:
            mat_path: 材质USD路径
            diffuse_color: Gf.Vec3f 漫反射颜色
            texture_path: 可选纹理文件绝对路径
            roughness: 粗糙度
        """
        if stage.GetPrimAtPath(mat_path):
            stage.RemovePrim(mat_path)
        
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        
        # === OmniPBR MDL Shader ===
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        
        # 设置漫反射颜色
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
        
        # 设置纹理（如果有）
        if texture_path and os.path.exists(texture_path):
            texture_filename = os.path.basename(texture_path)
            relative_path = f"../textures/{texture_filename}"
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(relative_path)
            # 启用纹理
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
            print(f"    OmniPBR + 纹理: {texture_filename}")
        else:
            print(f"    OmniPBR 纯色: RGB({diffuse_color[0]:.2f}, {diffuse_color[1]:.2f}, {diffuse_color[2]:.2f})")
        
        # 粗糙度和金属度
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # 连接MDL surface output
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        
        # 同时创建UsdPreviewSurface作为备用（非RTX渲染器可能用到）
        shader_preview = UsdShade.Shader.Define(stage, mat_path + "/PreviewShader")
        shader_preview.CreateIdAttr("UsdPreviewSurface")
        shader_preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
        shader_preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader_preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader_preview.ConnectableAPI(), "surface")
        
        return material
    
    # ===== 创建材质 =====
    leaves_color = Gf.Vec3f(0.15, 0.55, 0.12)   # 绿色
    bark_color = Gf.Vec3f(0.35, 0.22, 0.10)      # 棕色
    cone_color = Gf.Vec3f(1.0, 0.4, 0.0)         # 橙色
    
    print(f"\n  创建树叶材质 (OmniPBR)...")
    leaves_mat = create_omnipbr_material(
        "/World/Looks/TreeLeaves", leaves_color,
        texture_path=texture_leaves_path, roughness=0.7
    )
    
    print(f"  创建树干材质 (OmniPBR)...")
    bark_mat = create_omnipbr_material(
        "/World/Looks/TreeBark", bark_color,
        texture_path=texture_bark_path, roughness=0.8
    )
    
    print(f"  创建交通锥材质 (OmniPBR)...")
    cone_mat = create_omnipbr_material(
        "/World/Looks/TrafficCone", cone_color,
        texture_path=texture_cone_path, roughness=0.6
    )
    
    # ===== 应用树木材质 =====
    tree_parent = stage.GetPrimAtPath("/World/Tree")
    tree_mesh_count = 0
    if tree_parent and tree_parent.IsValid():
        for desc in Usd.PrimRange(tree_parent):
            if desc.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(desc)
                UsdShade.MaterialBindingAPI(desc).Bind(leaves_mat)
                
                # 设置displayColor（确保点云颜色）
                gprim = UsdGeom.Gprim(desc)
                gprim.GetDisplayColorAttr().Set([leaves_color])
                tree_mesh_count += 1
        print(f"  ✓ 树木: {tree_mesh_count} 个Mesh 已应用绿色OmniPBR材质")
    else:
        print("  ⚠ 未找到 /World/Tree")
    
    # ===== 应用交通锥材质 =====
    gp = stage.GetPrimAtPath("/World/GroundPlane")
    cone_mesh_count = 0
    cone_obj_count = 0
    if gp and gp.IsValid():
        for child in gp.GetChildren():
            if child.GetName().startswith("Cone001") or "Cone" in child.GetName():
                cone_obj_count += 1
                for desc in Usd.PrimRange(child):
                    if desc.IsA(UsdGeom.Mesh):
                        UsdShade.MaterialBindingAPI.Apply(desc)
                        UsdShade.MaterialBindingAPI(desc).Bind(cone_mat)
                        
                        gprim = UsdGeom.Gprim(desc)
                        gprim.GetDisplayColorAttr().Set([cone_color])
                        cone_mesh_count += 1
        print(f"  ✓ 交通锥: {cone_obj_count} 个锥, {cone_mesh_count} 个Mesh 已应用橙色OmniPBR材质")
    else:
        print("  ⚠ 未找到 /World/GroundPlane")
    
    return {"tree_meshes": tree_mesh_count, "cone_meshes": cone_mesh_count}


def setup_scene_lighting(stage):
    """
    设置场景光照 - 添加Dome Light（环境光/天空光）
    
    问题：场景只有一个方向光（太阳），背光面的物体全部纯黑，天空也是黑色。
    解决：添加Dome Light提供360°环境照明，模拟真实天空散射光。
    """
    from pxr import UsdLux, Gf
    
    dome_path = "/World/DomeLight"
    
    # 检查是否已存在
    existing = stage.GetPrimAtPath(dome_path)
    if existing and existing.IsValid():
        print(f"  Dome Light 已存在: {dome_path}")
        # 确保强度合适
        dome = UsdLux.DomeLight(existing)
        try:
            intensity = dome.GetIntensityAttr().Get()
            print(f"  当前强度: {intensity}")
            if intensity is None or intensity < 100:
                dome.GetIntensityAttr().Set(500.0)
                print(f"  → 已调整强度为 500")
        except:
            pass
        return
    
    # 创建Dome Light
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    
    # 强度 - 足够照亮阴影面，但不过亮
    dome.CreateIntensityAttr().Set(500.0)
    
    # 颜色 - 略带蓝色的天空光
    dome.CreateColorAttr().Set(Gf.Vec3f(0.75, 0.85, 1.0))
    
    # 设置为可见的天空背景（不是纯黑）
    # specular = 0.5 避免过强的环境反射
    try:
        dome.CreateSpecularAttr().Set(0.5)
    except:
        pass
    
    print(f"  ✓ 已创建 Dome Light: {dome_path}")
    print(f"    强度: 500, 颜色: 天空蓝 (0.75, 0.85, 1.0)")
    
    # 检查并调整现有方向光（如果太强会导致对比过高）
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DistantLight):
            distant = UsdLux.DistantLight(prim)
            try:
                intensity = distant.GetIntensityAttr().Get()
                if intensity and intensity > 2000:
                    distant.GetIntensityAttr().Set(1500.0)
                    print(f"  → 降低方向光强度: {prim.GetPath()} ({intensity} → 1500)")
            except:
                pass


"""========== 准备工作 =========="""

# 创建输出目录
for dir_path in [path_dir_dataset, path_dir_rgb, path_dir_depth, 
                 path_dir_pointcloud, path_dir_label, path_dir_logs]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")

# 检查已有数据，从最大编号+1开始
path_dir = Path(path_dir_label)
pattern = re.compile(r"label_(\d+)\.json")
existing_iters = []
for file in path_dir.glob("label_*.json"):
    match = pattern.match(file.name)
    if match:
        existing_iters.append(int(match.group(1)))

current_iter = max(existing_iters) + 1 if existing_iters else 0
print(f"起始迭代编号: {current_iter}")

# 获取Stage
stage = usd.get_context().get_stage()
if stage is None:
    print("错误: 无法获取USD Stage！请确保已加载场景。")
    raise RuntimeError("Stage is None")

print(f"当前场景: {stage.GetRootLayer().identifier}")

"""========== 主循环 =========="""

async def generate_data():
    global current_iter
    
    print("\n" + "="*60)
    print("开始数据生成")
    print("="*60)
    
    # 初始化日志记录器
    logger = DataQualityLogger(path_dir_logs)
    print(f"日志记录器已初始化: {path_dir_logs}")
    print(f"⚠ 点云验证: {'启用' if enable_pointcloud_validation else '禁用（诊断模式）'}")
    
    # 场景检查
    print(f"\n[诊断] 场景信息:")
    prim_count = len(list(stage.Traverse()))
    print(f"  Stage总Primitives数: {prim_count}")
    
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim:
        world_children = list(world_prim.GetChildren())
        print(f"  /World下的子对象数: {len(world_children)}")
        if len(world_children) > 0:
            print(f"  前5个子对象:")
            for child in world_children[:5]:
                print(f"    - {child.GetPath()} ({child.GetTypeName()})")
    else:
        print(f"  ⚠ /World路径不存在！")
    
    # 构建吊车部件映射（一次性）
    print(f"\n[诊断] 构建吊车部件映射...")
    crane_map = build_crane_part_map(stage)
    
    # 设置场景光照（添加环境光，解决背光面纯黑问题）
    print(f"\n[诊断] 设置场景光照...")
    setup_scene_lighting(stage)
    
    # 修复树木和交通锥材质（OmniPBR）
    print(f"\n[诊断] 修复场景材质...")
    material_results = fix_scene_materials(stage)
    
    """1. 设置相机"""
    print(f"\n[步骤1] 初始化相机...")
    camera = Camera(prim_path=camera_path, resolution=(img_width, img_height))
    
    # 检查相机是否存在
    prim_camera = stage.GetPrimAtPath(camera_path)
    if not prim_camera.IsValid():
        print(f"  相机 {camera_path} 不存在，正在创建...")
        stage.DefinePrim(camera_path, "Camera")
        prim_camera = stage.GetPrimAtPath(camera_path)
    
    if not prim_camera.IsActive():
        prim_camera.SetActive(True)
        print(f"  激活相机: {camera_path}")
    
    # 设置相机clipping范围（根据场景尺寸）
    try:
        cam_usd = UsdGeom.Camera(prim_camera)
        cam_usd.GetClippingRangeAttr().Set((0.5, 250.0))  # near=0.5m, far=250m
        
        # 设置更大的视场角（FOV）
        # 通过减小focal length或增大horizontal aperture来增大FOV
        # 默认focal length约为18mm，减小到12mm可以获得更大视场
        cam_usd.GetFocalLengthAttr().Set(12.0)  # 更小的焦距 = 更大的FOV
        cam_usd.GetHorizontalApertureAttr().Set(25.0)  # 增大光圈尺寸
        
        print(f"  设置相机clipping范围: (0.5, 250.0)")
        print(f"  设置相机FOV: focal_length=12mm, aperture=25mm (广角)")
    except Exception as e:
        print(f"  ⚠ 无法设置相机参数: {e}")
    
    await asyncio.sleep(1)
    camera.initialize()
    await asyncio.sleep(1)
    print("  相机初始化完成")
    
    """2. 附加Annotators"""
    print(f"\n[步骤2] 附加数据采集器...")
    
    # RGB图像（内置）
    # 深度图
    depth_annotator = AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annotator.attach(camera.get_render_product_path())
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(0.5)
    
    # 点云（增加初始化配置）
    pcd_annotator = AnnotatorRegistry.get_annotator("pointcloud")
    pcd_annotator.attach(camera.get_render_product_path())
    # 等待点云annotator完全初始化（点云需要更多时间）
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(1.0)
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(0.5)
    
    # 实例分割
    instance_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
    instance_annotator.attach(camera.get_render_product_path())
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(0.5)
    
    # 3D边界框
    bbox_3d_annotator = AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox_3d_annotator.attach(camera.get_render_product_path())
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(0.5)
    
    print("  所有采集器已附加")
    
    """3. 启动时间线"""
    print(f"\n[步骤3] 启动仿真时间线...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await asyncio.sleep(3)  # 增加等待让timeline稳定
    
    camera.add_motion_vectors_to_frame()
    await asyncio.sleep(2)  # 增加等待让渲染管线准备好
    print("  时间线已启动，等待渲染管线稳定...")
    
    # 额外等待确保第一帧渲染正常
    await asyncio.sleep(2)
    print("  渲染管线已就绪")
    
    # 预热所有annotator（测试采集）
    print("\n[预热] 测试所有数据采集器...")
    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(1.0)
    
    test_pcd = pcd_annotator.get_data()
    test_rgb = camera.get_rgba()
    test_depth = depth_annotator.get_data()
    
    print(f"  点云采集器: {'✓' if test_pcd is not None else '✗'}")
    print(f"  RGB采集器: {'✓' if test_rgb is not None else '✗'}")
    print(f"  深度采集器: {'✓' if test_depth is not None else '✗'}")
    
    if test_pcd is not None and 'data' in test_pcd:
        xyz = test_pcd['data']
        if xyz is not None and len(xyz) > 0:
            print(f"  预热点云: {len(xyz)} 个点 ✓")
        else:
            print(f"  预热点云: 数据为空 ⚠")
    else:
        print(f"  预热点云: annotator返回None ⚠")
    
    print("  预热完成\n")
    
    """4. 数据采集循环"""
    print(f"\n[步骤4] 开始数据采集循环 (目标: {max_iterations} 帧)...")
    print("="*60)
    
    # 预先生成所有相机位置（系统化采样）
    all_camera_positions = get_systematic_camera_positions(max_iterations)
    actual_frames = len(all_camera_positions)
    print(f"\n已生成 {actual_frames} 个系统化采样位置（请求: {max_iterations}）")
    print("采样策略: 同心圆，从近到远覆盖场景")
    print("="*60)
    
    # 重置帧计数器（修复多次运行时不重置的问题）
    current_iter = 0
    
    while current_iter < actual_frames:
        # 每10帧随机改变物体位置和树木颜色（更频繁以增加数据多样性）
        if current_iter > 0 and current_iter % 10 == 0:
            print(f"\n[场景随机化] 帧 {current_iter} - 重新随机物体位置和树木颜色...")
            
            # 随机化物体位置（含碰撞检测）
            randomized = randomize_object_positions(stage)
            if randomized:
                print(f"  ✓ 已随机移动 {len(randomized)} 个物体:")
                for obj_info in randomized:
                    pos = obj_info['new_pos']
                    rotation = obj_info.get('rotation')
                    obj_type = obj_info.get('type', '?')
                    obj_name = obj_info['path'].split('/')[-1]
                    if rotation is not None:
                        print(f"    - [{obj_type}] {obj_name}: 位置 ({pos[0]:.2f}, {pos[1]:.2f})m, 旋转 {rotation:.1f}°")
                    else:
                        print(f"    - [{obj_type}] {obj_name}: 位置 ({pos[0]:.2f}, {pos[1]:.2f})m")
            else:
                print(f"  ⚠ 未能随机移动任何物体")
            
            # 等待场景更新
            await omni.kit.app.get_app().next_update_async()
            await asyncio.sleep(0.5)
        
        # 使用预定义的相机位置
        cam_position, aimed_point = all_camera_positions[current_iter]
        logger.log_frame_start(current_iter, cam_position)
        
        # 重试循环：如果点云为空则微调相机位置
        retry_count = 0
        frame_valid = False
        
        while not frame_valid and retry_count < max_retry_per_frame:
            if retry_count > 0:
                logger.log_retry(retry_count)
                # 微调相机位置（添加小随机偏移）
                offset = np.random.uniform(-2, 2, size=3)
                offset[2] *= 0.5  # z方向偏移更小
                cam_position_adjusted = cam_position + offset
            else:
                cam_position_adjusted = cam_position
            
            """4.1 设置相机位置"""
            cam_orientation = camPosOri(cam_position_adjusted, aimed_point)
            
            camera.set_world_pose(position=cam_position_adjusted, orientation=cam_orientation)
            if retry_count == 0:
                print(f"  相机位置: {cam_position_adjusted}")
                print(f"  瞄准点: {aimed_point}")
            
            # 等待渲染管线更新（确保所有annotator同步）
            await omni.kit.app.get_app().next_update_async()
            await asyncio.sleep(1.5)
            await omni.kit.app.get_app().next_update_async()
            await asyncio.sleep(0.5)
        
            # 获取相机位姿
            try:
                cam_pose = get_obj_pose(stage, camera_path)
            except:
                cam_pose = list(cam_position_adjusted) + [0, 0, 0, 1]
            
            """4.2 预采集点云验证"""
            # 等待渲染管线完全更新（增加等待时间）
            await asyncio.sleep(0.5)
            await omni.kit.app.get_app().next_update_async()
            await asyncio.sleep(1.0)  # 增加等待时间
            await omni.kit.app.get_app().next_update_async()
            await asyncio.sleep(0.5)
            
            # 多次尝试获取点云数据（解决同步问题）
            pcd_data_check = None
            for attempt in range(5):  # 增加重试次数
                pcd_data_check = pcd_annotator.get_data()
                if pcd_data_check is not None and 'data' in pcd_data_check:
                    xyz_check = pcd_data_check['data']
                    if xyz_check is not None and len(xyz_check) > 0:
                        break  # 成功获取点云
                if current_iter < 3 and attempt == 0:
                    print(f"    [调试] 点云采集尝试 {attempt+1}: {type(pcd_data_check)}")
                await asyncio.sleep(0.5)  # 增加等待时间
            
            pointcloud_valid = False
            point_count = 0
            
            if enable_pointcloud_validation:
                # 启用验证模式
                if pcd_data_check is not None and 'data' in pcd_data_check:
                    xyz_check = pcd_data_check['data']
                    if xyz_check is not None and len(xyz_check) > 0:
                        point_count = len(xyz_check)
                        if point_count >= min_pointcloud_points:
                            pointcloud_valid = True
                            logger.log_pointcloud(True, point_count)
                        else:
                            logger.log_pointcloud(False, point_count, f"少于阈值 {min_pointcloud_points}")
                    else:
                        logger.log_pointcloud(False, 0, "数据为None或空")
                else:
                    logger.log_pointcloud(False, 0, "annotator返回None")
                
                # 如果点云无效，重新采样
                if not pointcloud_valid:
                    retry_count += 1
                    continue
            else:
                # 禁用验证模式 - 直接采集，记录点云状态但不重试
                if pcd_data_check is not None and 'data' in pcd_data_check:
                    xyz_check = pcd_data_check['data']
                    if xyz_check is not None and len(xyz_check) > 0:
                        point_count = len(xyz_check)
                        logger.log_pointcloud(True, point_count)
                    else:
                        logger.log_pointcloud(False, 0, "数据为None或空 (验证已禁用，继续采集)")
                else:
                    logger.log_pointcloud(False, 0, "annotator返回None (验证已禁用，继续采集)")
                pointcloud_valid = True  # 强制通过
            
            # 点云有效（或验证被禁用），继续采集其他数据
            frame_valid = True
            
        # 如果所有重试都失败，记录失败并跳过
        if not frame_valid:
            logger.log_frame_end(False)
            current_iter += 1
            continue
        
        """4.3 采集RGB图像"""
        rgb_image = camera.get_rgba()
        if rgb_image is not None and rgb_image.size > 0:
            bgr_image = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
            rgb_path = f"{path_dir_rgb}/rgb_{current_iter:06d}.png"
            cv2.imwrite(rgb_path, bgr_image)
            logger.log_rgb(True)
        else:
            logger.log_rgb(False, "camera.get_rgba()返回None或空")
        
        await asyncio.sleep(0.3)
        
        """4.4 采集深度图"""
        depth_data = depth_annotator.get_data()
        if depth_data is not None and depth_data.size > 0:
            # 记录深度图质量
            logger.log_depth(True, depth_data)
            
            # 保存原始深度数据（CSV）
            depth_csv_path = f"{path_dir_depth}/depth_{current_iter:06d}.csv"
            np.savetxt(depth_csv_path, depth_data, delimiter=' ', fmt='%.6f')
            
            # 保存可视化深度图（PNG）- 改进可视化逻辑
            depth_valid = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
            if len(depth_valid) > 0:
                depth_min = np.min(depth_valid)
                depth_max = np.max(depth_valid)
                
                # 归一化到0-255
                depth_normalized = np.zeros_like(depth_data, dtype=np.uint8)
                mask = np.isfinite(depth_data) & (depth_data > 0)
                depth_normalized[mask] = ((depth_data[mask] - depth_min) / (depth_max - depth_min + 1e-6) * 255).astype(np.uint8)
                
                # 应用颜色映射
                depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_png_path = f"{path_dir_depth}/depth_{current_iter:06d}.png"
                cv2.imwrite(depth_png_path, depth_color)
            else:
                # 全黑图
                depth_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                depth_png_path = f"{path_dir_depth}/depth_{current_iter:06d}.png"
                cv2.imwrite(depth_png_path, depth_color)
        else:
            logger.log_depth(False, reason="annotator返回None或空")
        
        await asyncio.sleep(0.3)
        
        """4.5 采集点云（带后备方案）"""
        pcd_path = f"{path_dir_pointcloud}/pointcloud_{current_iter:06d}.txt"
        pcd_saved = False
        
        # 方法1: 尝试使用pointcloud annotator
        pcd_data = pcd_annotator.get_data()
        if pcd_data is not None and 'data' in pcd_data:
            xyz = pcd_data['data']
            if xyz is not None and len(xyz) > 0:
                save_pointcloud_with_rgb(pcd_data, pcd_path)
                pcd_saved = True
                if current_iter < 3:
                    print(f"  [点云] 使用annotator: {len(xyz)} 个点")
        
        # 方法2: 如果annotator失败，从深度图生成点云
        if not pcd_saved and depth_data is not None and rgb_image is not None:
            try:
                # 获取相机参数
                try:
                    horizontal_aperture = camera.get_horizontal_aperture()
                    focal_length = camera.get_focal_length()
                    vertical_aperture = horizontal_aperture * (img_height / img_width)
                    cam_params = {
                        "horizontal_aperture": horizontal_aperture,
                        "vertical_aperture": vertical_aperture,
                        "focal_length": focal_length,
                        "width": img_width,
                        "height": img_height
                    }
                except:
                    cam_params = {
                        "horizontal_aperture": 20.955,
                        "vertical_aperture": 15.2908,
                        "focal_length": 18.14,
                        "width": img_width,
                        "height": img_height
                    }
                
                # 从深度图生成点云
                xyzrgb = depth_to_pointcloud_with_rgb(depth_data, rgb_image, cam_params, cam_pose)
                if xyzrgb is not None and len(xyzrgb) > 0:
                    np.savetxt(pcd_path, xyzrgb, fmt='%.6f', delimiter=' ', 
                              header='x y z r g b', comments='')
                    pcd_saved = True
                    logger.log_pointcloud(True, len(xyzrgb))
                    if current_iter < 3:
                        print(f"  [点云] 使用深度图后备方案: {len(xyzrgb)} 个点")
            except Exception as e:
                if current_iter < 3:
                    print(f"  [点云] 深度图后备方案失败: {e}")
        
        if not pcd_saved:
            if current_iter < 3:
                print(f"  [点云] 所有方法都失败，未保存点云")
        
        await asyncio.sleep(0.3)
        
        """4.6 采集物体信息（只识别视野内的物体）"""
        # 策略：优先使用bbox_3d_annotator，失败时遍历场景
        
        # 等待bbox_3d_annotator同步（类似点云）
        await omni.kit.app.get_app().next_update_async()
        await asyncio.sleep(0.5)
        
        object_list = []
        bbox_data = bbox_3d_annotator.get_data()
        
        # 调试：前3帧总是输出，后续帧如果没有检测到物体也输出
        should_debug = (current_iter < 3)
        
        visible_prim_paths = []
        
        # 方法1: 尝试从bbox_3d_annotator获取可见物体
        if bbox_data is not None and 'info' in bbox_data:
            if 'primPaths' in bbox_data['info']:
                visible_prim_paths = bbox_data['info']['primPaths']
                if should_debug or len(visible_prim_paths) == 0:
                    print(f"  [物体检测] bbox_3d检测到 {len(visible_prim_paths)} 个可见物体")
                    if len(visible_prim_paths) > 0:
                        print(f"  [物体检测] 所有prim路径 (前10个):")
                        for i, path in enumerate(visible_prim_paths[:10]):
                            path_lower = path.lower()
                            matched = "未匹配"
                            for key in construction_class.keys():
                                if key in path_lower:
                                    matched = f"✓ {key}"
                                    break
                            print(f"    {i+1}. {path} [{matched}]")
                        if len(visible_prim_paths) > 10:
                            print(f"    ... 还有 {len(visible_prim_paths)-10} 个")
            else:
                if should_debug:
                    print(f"  [物体检测] bbox_3d没有'primPaths'字段: {list(bbox_data['info'].keys())}")
        else:
            if should_debug:
                if bbox_data is None:
                    print(f"  [物体检测] bbox_3d返回None")
                elif 'info' not in bbox_data:
                    print(f"  [物体检测] bbox_3d没有'info'字段: {list(bbox_data.keys())}")
        
        # 方法2: 使用instance_segmentation查看实际可见的物体（用于调试）
        if should_debug:
            try:
                instance_data = instance_annotator.get_data()
                if instance_data is not None and 'info' in instance_data:
                    print(f"  [实例分割] info字段: {list(instance_data['info'].keys())}")
                    
                    # 检查idToLabels
                    if 'idToLabels' in instance_data['info']:
                        id_to_labels = instance_data['info']['idToLabels']
                        print(f"  [实例分割] idToLabels检测到 {len(id_to_labels)} 个实例:")
                        for instance_id, label_info in list(id_to_labels.items())[:10]:
                            # label_info可能是字符串或字典
                            if isinstance(label_info, dict):
                                class_name = label_info.get('class', 'unknown')
                                print(f"    ID {instance_id}: {class_name} | {label_info}")
                            else:
                                print(f"    ID {instance_id}: {label_info}")
                    
                    # 检查idToSemantics（更详细的语义信息）
                    if 'idToSemantics' in instance_data['info']:
                        id_to_semantics = instance_data['info']['idToSemantics']
                        print(f"  [实例分割] idToSemantics检测到 {len(id_to_semantics)} 个语义:")
                        for sem_id, sem_info in list(id_to_semantics.items())[:10]:
                            if isinstance(sem_info, dict):
                                print(f"    语义ID {sem_id}: {sem_info}")
                            else:
                                print(f"    语义ID {sem_id}: {sem_info}")
                            
            except Exception as e:
                print(f"  [实例分割] 检查失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 如果bbox_3d未返回物体，输出可能原因
        if len(visible_prim_paths) == 0 and should_debug:
            print(f"  [物体检测] ⚠ bbox_3d未返回任何物体")
            print(f"  [物体检测] 可能原因：")
            print(f"    1. 物体没有设置Semantic Label")
            print(f"    2. 物体的Mesh没有启用包围盒计算")
            print(f"    3. 物体被标记为不可见或禁用")
        
        # 处理找到的prim路径 - 聚合Mesh组件为完整物体
        if len(visible_prim_paths) > 0:
            # 使用字典聚合：object_root -> {class_id, class_name, mesh_paths}
            object_roots = {}
            unmatched_count = 0
            
            for prim_path in visible_prim_paths:
                # 获取物体根路径和类别
                object_root, class_name, class_id = get_object_root(prim_path)
                
                if object_root is not None:
                    if object_root not in object_roots:
                        object_roots[object_root] = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "mesh_paths": []
                        }
                    object_roots[object_root]["mesh_paths"].append(prim_path)
                else:
                    unmatched_count += 1
                    if should_debug and unmatched_count <= 3:
                        print(f"    ✗ 未匹配: {prim_path}")
            
            # 将聚合后的物体添加到列表
            inst_idx = 0
            for object_root, obj_info in object_roots.items():
                object_list.append({
                    "inst_idx": inst_idx,
                    "class_id": obj_info["class_id"],
                    "class_name": obj_info["class_name"],
                    "prim_path": object_root,  # 使用物体根路径（吊车部件含#分隔符）
                    "mesh_count": len(obj_info["mesh_paths"]),
                    "mesh_paths": obj_info["mesh_paths"],  # 保留mesh路径供位姿计算使用
                })
                inst_idx += 1
            
            if should_debug:
                print(f"  [物体检测] 聚合统计: {len(visible_prim_paths)} 个Mesh -> {len(object_roots)} 个物体")
                print(f"  [物体检测] 物体列表:")
                for obj in object_list[:10]:
                    print(f"    - {obj['class_name']}: {obj['prim_path']} ({obj.get('mesh_count', 1)} meshes)")
                if len(object_list) > 10:
                    print(f"    ... 还有 {len(object_list)-10} 个")
                if unmatched_count > 0:
                    print(f"  [物体检测] 未匹配: {unmatched_count} 个Mesh")
        
        if len(object_list) > 0:
            print(f"  ✓ 标签: {len(object_list)} 个物体")
        else:
            print(f"  ⚠ 标签: 0 个物体（视野外或未匹配类别）")
        
        # 创建空的实例掩码（因为没有语义分割数据）
        instance_mask = np.zeros((img_height, img_width), dtype=np.int32)
        instance_mask.fill(-1)
        
        """4.7 采集3D边界框位姿"""
        pose_list = []
        
        # 尝试使用bbox_3d_annotator
        bbox_data = bbox_3d_annotator.get_data()
        use_annotator = False
        
        if bbox_data is not None and 'data' in bbox_data and len(bbox_data['data']) > 0:
            use_annotator = True
            bbox_dict_list = bbox_data['data']
            bbox_prim_paths = bbox_data['info']['primPaths']
        
        for obj in object_list:
            prim_path = obj['prim_path']
            
            # 处理吊车部件的虚拟聚合路径（含#分隔符）
            # 例如: /World/.../crane_root#cranebase -> 使用其mesh_paths获取位姿
            actual_prim_path = prim_path.split("#")[0] if "#" in prim_path else prim_path
            
            # 方法1：尝试从bbox_3d_annotator获取
            if use_annotator:
                try:
                    bbox_index = bbox_prim_paths.index(actual_prim_path)
                    bbox_dict = bbox_dict_list[bbox_index]
                    center, size, euler = bboxDict_to_transform(bbox_dict)
                    
                    pose_info = {
                        "inst_idx": obj['inst_idx'],
                        "class_id": obj['class_id'],
                        "class_name": obj['class_name'],
                        "center": center,
                        "size": size,
                        "rotation": euler,
                        "prim_path": prim_path
                    }
                    pose_list.append(pose_info)
                    continue
                except:
                    pass
                
                # 对于吊车部件：尝试用该部件的任一mesh路径查找bbox
                if "#" in prim_path and "mesh_paths" in obj:
                    found = False
                    for mesh_path in obj.get("mesh_paths", []):
                        try:
                            bbox_index = bbox_prim_paths.index(mesh_path)
                            bbox_dict = bbox_dict_list[bbox_index]
                            center, size, euler = bboxDict_to_transform(bbox_dict)
                            pose_info = {
                                "inst_idx": obj['inst_idx'],
                                "class_id": obj['class_id'],
                                "class_name": obj['class_name'],
                                "center": center,
                                "size": size,
                                "rotation": euler,
                                "prim_path": prim_path
                            }
                            pose_list.append(pose_info)
                            found = True
                            break
                        except:
                            continue
                    if found:
                        continue
            
            # 方法2：从USD prim直接获取位姿
            try:
                prim = stage.GetPrimAtPath(actual_prim_path)
                if not prim or not prim.IsValid():
                    # 吊车部件：尝试从mesh_paths获取第一个有效prim
                    for mp in obj.get("mesh_paths", []):
                        p = stage.GetPrimAtPath(mp)
                        if p and p.IsValid():
                            prim = p
                            break
                
                if prim and prim.IsValid():
                    xform = UsdGeom.Xformable(prim)
                    if xform:
                        # 获取世界坐标变换
                        world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        translation = world_matrix.ExtractTranslation()
                        
                        # 获取旋转
                        rotation_matrix = world_matrix.ExtractRotationMatrix()
                        rot_np = np.array(rotation_matrix.GetTranspose())
                        r_obj = R.from_matrix(rot_np)
                        euler = r_obj.as_euler('xyz', degrees=True)
                        
                        # 计算包围盒尺寸
                        try:
                            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
                            bbox = bbox_cache.ComputeWorldBound(prim)
                            bbox_range = bbox.ComputeAlignedRange()
                            size = [
                                bbox_range.GetMax()[0] - bbox_range.GetMin()[0],
                                bbox_range.GetMax()[1] - bbox_range.GetMin()[1],
                                bbox_range.GetMax()[2] - bbox_range.GetMin()[2]
                            ]
                        except:
                            size = [1.0, 1.0, 1.0]
                        
                        pose_info = {
                            "inst_idx": obj['inst_idx'],
                            "class_id": obj['class_id'],
                            "class_name": obj['class_name'],
                            "center": [translation[0], translation[1], translation[2]],
                            "size": size,
                            "rotation": euler.tolist(),
                            "prim_path": prim_path
                        }
                        pose_list.append(pose_info)
            except Exception as e:
                if current_iter < 3:
                    print(f"    警告: 无法获取 {prim_path} 的位姿: {e}")
                continue
        
        if len(pose_list) > 0:
            print(f"  ✓ 位姿估计: {len(pose_list)} 个物体")
        else:
            print(f"  ⚠ 位姿估计: 0 个物体")
        
        """4.8 相机参数"""
        try:
            horizontal_aperture = camera.get_horizontal_aperture()
            focal_length = camera.get_focal_length()
            vertical_aperture = horizontal_aperture * (img_height / img_width)
            cam_params = {
                "horizontal_aperture": horizontal_aperture,
                "vertical_aperture": vertical_aperture,
                "focal_length": focal_length,
                "width": img_width,
                "height": img_height
            }
        except:
            cam_params = {
                "horizontal_aperture": 20.955,
                "vertical_aperture": 15.2908,
                "focal_length": 18.14,
                "width": img_width,
                "height": img_height
            }
        
        """4.9 保存标注文件"""
        label_data = {
            "frame_id": current_iter,
            "camera_pose": cam_pose,  # [x, y, z, qx, qy, qz, qw]
            "camera_params": cam_params,
            "objects": pose_list,
            "instance_mask_shape": [img_height, img_width],
            "num_objects": len(pose_list),
            "class_mapping": construction_class
        }
        
        # 保存实例掩码（可选，因为文件较大）
        if instance_mask is not None:
            mask_path = f"{path_dir_label}/instance_mask_{current_iter:06d}.npy"
            np.save(mask_path, instance_mask)
        
        label_path = f"{path_dir_label}/label_{current_iter:06d}.json"
        save_label_json(label_data, label_path)
        
        # 记录标签状态
        logger.log_labels(len(pose_list))
        
        # 记录帧完成
        logger.log_frame_end(True)
        
        current_iter += 1
        await asyncio.sleep(0.5)
    
    """5. 清理与报告"""
    print("\n" + "="*60)
    print("数据采集完成！")
    print(f"数据集路径: {path_dir_dataset}")
    print("="*60)
    
    # 保存日志汇总
    report = logger.save_summary()
    print("\n" + report)
    
    timeline.stop()
    await asyncio.sleep(1)


# 启动异步任务
asyncio.ensure_future(generate_data())

print("\n脚本加载完成，数据生成任务已启动...")
print("提示: 在Isaac Sim Script Editor中运行此脚本")

