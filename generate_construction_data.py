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
taskNum = 'construction_world2'
path_dir_root = "/home/xander/partime/ConstructionScenePoseEstimation"
path_dir_dataset = f"{path_dir_root}/dataset_{taskNum}"
path_dir_rgb = f"{path_dir_dataset}/rgb"
path_dir_depth = f"{path_dir_dataset}/depth"
path_dir_pointcloud = f"{path_dir_dataset}/pointcloud"
path_dir_label = f"{path_dir_dataset}/labels"

"""场景参数"""
# 相机参数
camera_path = "/World/Camera_0"
img_width = 1280
img_height = 720

# 相机位置范围（用于随机采样）
cam_distance_range = [8.0, 15.0]  # 距离中心的距离范围
cam_height_range = [2.0, 6.0]     # 相机高度范围
cam_angle_range = [0, 360]        # 水平角度范围

# 目标瞄准点（场景中心）
aimed_point = np.array([0, 0, 1.5])

"""数据集参数"""
max_iterations = 100  # 生成的总帧数
current_iter = 0

"""物体类别定义（根据world2.usd场景中的物体调整）"""
construction_class = {
    "crane": 0,
    "dumper": 1,
    "fence": 2,
    "tree": 3,
    "trafficcone": 4,
    "people": 5,
    # 可根据实际场景添加更多类别
}

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
    使用 task3.py 中验证过的方法
    Input: 
        target_point: 相机位置 (3,)
        aimed_point: 目标瞄准点 (3,)
    Output: 
        q: 四元数 (w, x, y, z)
    """
    x2 = (aimed_point - target_point) / np.linalg.norm(aimed_point - target_point)
    x1 = np.array([-1, 0, 0])
    y1 = np.array([0, -1, 0])
    z1 = np.array([0, 0, 1])
    y2 = np.array([- x1[1]/(np.sqrt(x1[0]**2 + x1[1]**2)), x1[0]/(np.sqrt(x1[0]**2 + x1[1]**2)), 0])
    z2 = np.cross(x2, y2)
    R_1to2 = np.linalg.inv(np.vstack((x2, y2, z2)).T) @ (np.vstack((x1, y1, z1)).T)
    R_0to1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R = R_1to2 @ R_0to1
    q = rotMtx2quaternion(R)
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
        
        print(f"  ✓ 点云: {xyzrgb.shape[0]} 个点 -> {filename}")
        
    except Exception as e:
        print(f"  ✗ 保存点云失败: {e}")
        import traceback
        traceback.print_exc()


def sample_camera_position(distance_range, height_range, angle_range):
    """
    随机采样相机位置
    """
    distance = np.random.uniform(distance_range[0], distance_range[1])
    height = np.random.uniform(height_range[0], height_range[1])
    angle = np.deg2rad(np.random.uniform(angle_range[0], angle_range[1]))
    
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    z = height
    
    return np.array([x, y, z])


"""========== 准备工作 =========="""

# 创建输出目录
for dir_path in [path_dir_dataset, path_dir_rgb, path_dir_depth, 
                 path_dir_pointcloud, path_dir_label]:
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
    await asyncio.sleep(0.5)
    
    # 点云
    pcd_annotator = AnnotatorRegistry.get_annotator("pointcloud")
    pcd_annotator.attach(camera.get_render_product_path())
    await asyncio.sleep(0.5)
    
    # 实例分割
    instance_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
    instance_annotator.attach(camera.get_render_product_path())
    await asyncio.sleep(0.5)
    
    # 3D边界框
    bbox_3d_annotator = AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox_3d_annotator.attach(camera.get_render_product_path())
    await asyncio.sleep(0.5)
    
    print("  所有采集器已附加")
    
    """3. 启动时间线"""
    print(f"\n[步骤3] 启动仿真时间线...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await asyncio.sleep(2)
    
    camera.add_motion_vectors_to_frame()
    await asyncio.sleep(1)
    print("  时间线已启动")
    
    """4. 数据采集循环"""
    print(f"\n[步骤4] 开始数据采集循环 (目标: {max_iterations} 帧)...")
    print("="*60)
    
    while current_iter < max_iterations:
        print(f"\n>>> 采集帧 {current_iter}/{max_iterations} <<<")
        
        """4.1 随机相机位置"""
        cam_position = sample_camera_position(cam_distance_range, cam_height_range, cam_angle_range)
        cam_orientation = camPosOri(cam_position, aimed_point)
        
        camera.set_world_pose(position=cam_position, orientation=cam_orientation)
        print(f"  相机位置: {cam_position}")
        await asyncio.sleep(2)
        
        # 获取相机位姿
        try:
            cam_pose = get_obj_pose(stage, camera_path)
        except:
            cam_pose = list(cam_position) + [0, 0, 0, 1]
        
        """4.2 采集RGB图像"""
        rgb_image = camera.get_rgba()
        if rgb_image is not None:
            bgr_image = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
            rgb_path = f"{path_dir_rgb}/rgb_{current_iter:06d}.png"
            cv2.imwrite(rgb_path, bgr_image)
            print(f"  ✓ RGB图像: {rgb_path}")
        else:
            print(f"  ✗ RGB图像采集失败")
        
        await asyncio.sleep(0.5)
        
        """4.3 采集深度图"""
        depth_data = depth_annotator.get_data()
        if depth_data is not None:
            # 保存原始深度数据（CSV）
            depth_csv_path = f"{path_dir_depth}/depth_{current_iter:06d}.csv"
            np.savetxt(depth_csv_path, depth_data, delimiter=' ', fmt='%.6f')
            
            # 保存可视化深度图（PNG）
            depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_norm = 255 - depth_norm
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_png_path = f"{path_dir_depth}/depth_{current_iter:06d}.png"
            cv2.imwrite(depth_png_path, depth_color)
            print(f"  ✓ 深度图: {depth_csv_path}")
        else:
            print(f"  ✗ 深度图采集失败")
        
        await asyncio.sleep(0.5)
        
        """4.4 采集点云"""
        pcd_data = pcd_annotator.get_data()
        if pcd_data is not None and 'data' in pcd_data:
            # 检查点云数据是否有效
            xyz = pcd_data['data']
            if xyz is not None and len(xyz) > 0:
                pcd_path = f"{path_dir_pointcloud}/pointcloud_{current_iter:06d}.txt"
                save_pointcloud_with_rgb(pcd_data, pcd_path)
            else:
                print(f"  ✗ 点云数据为空")
        else:
            print(f"  ✗ 点云采集失败")
        
        await asyncio.sleep(0.5)
        
        """4.5 采集实例分割"""
        instance_data = instance_annotator.get_data()
        instance_mask = None
        object_list = []
        
        if instance_data is not None:
            id_to_semantics = instance_data['info']['idToSemantics']
            id_to_labels = instance_data['info']['idToLabels']
            
            # 创建实例掩码
            instance_mask = np.zeros((img_height, img_width), dtype=np.int32)
            instance_mask.fill(-1)
            
            inst_idx = 0
            for semantic_id, semantic_info in id_to_semantics.items():
                class_name = semantic_info.get("class", "").lower()
                
                # 检查是否在目标类别中
                class_id = None
                for key, val in construction_class.items():
                    if key in class_name:
                        class_id = val
                        break
                
                if class_id is not None:
                    prim_path = id_to_labels.get(semantic_id, None)
                    if prim_path:
                        instance_mask[instance_data['data'] == semantic_id] = inst_idx
                        object_list.append({
                            "inst_idx": inst_idx,
                            "class_id": class_id,
                            "class_name": class_name,
                            "prim_path": prim_path
                        })
                        inst_idx += 1
            
            print(f"  ✓ 实例分割: {len(object_list)} 个物体")
        else:
            print(f"  ✗ 实例分割采集失败")
        
        """4.6 采集3D边界框位姿"""
        bbox_data = bbox_3d_annotator.get_data()
        pose_list = []
        
        if bbox_data is not None and len(object_list) > 0:
            bbox_dict_list = bbox_data['data']
            bbox_prim_paths = bbox_data['info']['primPaths']
            
            for obj in object_list:
                prim_path = obj['prim_path']
                try:
                    bbox_index = bbox_prim_paths.index(prim_path)
                    bbox_dict = bbox_dict_list[bbox_index]
                    
                    center, size, euler = bboxDict_to_transform(bbox_dict)
                    
                    pose_info = {
                        "inst_idx": obj['inst_idx'],
                        "class_id": obj['class_id'],
                        "class_name": obj['class_name'],
                        "center": center,      # [x, y, z]
                        "size": size,          # [width, height, depth]
                        "rotation": euler,     # [roll, pitch, yaw] in degrees
                        "prim_path": prim_path
                    }
                    pose_list.append(pose_info)
                except:
                    print(f"    警告: 无法获取 {prim_path} 的边界框")
                    continue
            
            print(f"  ✓ 位姿估计: {len(pose_list)} 个物体")
        else:
            print(f"  ✗ 3D边界框采集失败")
        
        """4.7 相机参数"""
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
        
        """4.8 保存标注文件"""
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
        print(f"  ✓ 标注文件: {label_path}")
        
        print(f">>> 帧 {current_iter} 完成 ✓\n")
        
        current_iter += 1
        await asyncio.sleep(1)
    
    """5. 清理"""
    print("\n" + "="*60)
    print("数据采集完成！")
    print(f"总共生成: {current_iter} 帧")
    print(f"数据集路径: {path_dir_dataset}")
    print("="*60)
    
    timeline.stop()
    await asyncio.sleep(1)


# 启动异步任务
asyncio.ensure_future(generate_data())

print("\n脚本加载完成，数据生成任务已启动...")
print("提示: 在Isaac Sim Script Editor中运行此脚本")

