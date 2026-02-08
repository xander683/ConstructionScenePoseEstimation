"""
替换场景中的树模型 → NVIDIA American_Beech
在Isaac Sim Script Editor中运行
"""

import omni.usd as usd
from pxr import Usd, UsdGeom, Sdf, Gf
import random

print("=" * 60)
print("替换树模型 → NVIDIA American_Beech")
print("=" * 60)

stage = usd.get_context().get_stage()
if stage is None:
    print("❌ 无法获取Stage")
    raise RuntimeError("No stage")

# 相对路径（相对于world2.usd所在的cad_models/目录）
# world2.usd 在 cad_models/，American_Beech.usd 在 cad_models/tree/
nvidia_tree_path = "./tree/American_Beech.usd"
scale_factor = 5.0 / 597.4  # 目标5m高

# 12棵树的世界坐标（之前用ComputeLocalToWorldTransform提取的）
# Z≈0 是地面，树从地面向上生长
tree_world_positions = [
    ("Tree",    ( 11.5,  -0.2,  0.0)),
    ("Tree_01", ( 11.5,  -7.4,  0.1)),
    ("Tree_02", ( 11.5,   6.1,  0.0)),
    ("Tree_03", (-13.1,   6.1,  0.0)),
    ("Tree_04", (-13.1,  -0.2,  0.1)),
    ("Tree_05", (-13.1,  -7.4,  0.1)),
    ("Tree_06", (  6.9, -13.3, -0.1)),
    ("Tree_07", ( -0.1, -13.3, -0.1)),
    ("Tree_08", ( -7.5, -13.3, -0.1)),
    ("Tree_09", ( -7.5,  12.9, -0.3)),
    ("Tree_10", ( -0.1,  12.9, -0.3)),
    ("Tree_11", (  6.9,  12.9, -0.3)),
]

print(f"  模型: American_Beech.usd")
print(f"  缩放: {scale_factor:.6f} (~5m高)")

# 第1步: 彻底删除 /World/Tree 并重建
print(f"\n  [1] 清除旧树...")
if stage.GetPrimAtPath("/World/Tree"):
    stage.RemovePrim("/World/Tree")
    print(f"      已删除 /World/Tree")

# 重建空的 /World/Tree 容器（无变换）
stage.DefinePrim("/World/Tree", "Xform")
print(f"      重建 /World/Tree")

# 第2步: 创建新树 - 直接用世界坐标
print(f"\n  [2] 创建新树...")

try:
    from pxr import Semantics
    has_sem = True
except ImportError:
    has_sem = False

for name, (wx, wy, wz) in tree_world_positions:
    path = f"/World/Tree/{name}"
    
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(nvidia_tree_path)
    
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    
    # 位置（世界坐标，Z≈0地面）
    xf.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))
    
    # 只绕Z轴旋转（树干保持朝上）
    rz = random.uniform(0, 360)
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, rz))
    
    # 缩放
    xf.AddScaleOp().Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
    
    # 语义标签
    if has_sem:
        try:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set("Tree")
        except:
            pass
    
    print(f"    ✓ {name:8s} ({wx:6.1f}, {wy:6.1f}, {wz:5.1f}) rotZ={rz:.0f}°")

# 第3步: 验证
print(f"\n  [3] 验证...")
for name, (wx, wy, wz) in tree_world_positions:
    prim = stage.GetPrimAtPath(f"/World/Tree/{name}")
    xf = UsdGeom.Xformable(prim)
    mat = xf.ComputeLocalToWorldTransform(0)
    world_z = mat[3][2]
    status = "✓" if abs(world_z) < 1.0 else "⚠ Z偏高!"
    print(f"    {status} {name:8s} worldZ={world_z:.2f}")

print(f"\n  ✅ 完成! 请 Ctrl+S 保存")
print("=" * 60)
