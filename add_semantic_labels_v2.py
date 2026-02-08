"""
为场景物体添加语义标签 - 使用Isaac Sim正确的Semantics API
Add Semantic Labels using Isaac Sim's proper Semantics Schema API

在Isaac Sim Script Editor中运行此脚本
"""

import omni.usd as usd
from pxr import Usd, UsdGeom, Sdf

print("="*80)
print("为场景物体添加语义标签 (v2 - 使用正确的API)")
print("="*80)

# 获取当前场景
stage = usd.get_context().get_stage()
if stage is None:
    print("❌ 错误: 无法获取Stage，请先加载场景文件")
else:
    print(f"✓ 场景文件: {stage.GetRootLayer().identifier}")
    print("")
    
    # 尝试导入Isaac Sim的Semantics模块
    try:
        from pxr import Semantics
        has_semantics_module = True
        print("✓ 已加载 Semantics 模块")
    except ImportError:
        has_semantics_module = False
        print("⚠ 无法导入 Semantics 模块，使用备用方法")
    
    # 需要添加标签的物体
    objects_to_label = [
        # 交通锥
        ("/World/GroundPlane/Cone001", "class", "TrafficCone"),
        ("/World/GroundPlane/Cone001_01", "class", "TrafficCone"),
        ("/World/GroundPlane/Cone001_02", "class", "TrafficCone"),
        
        # 吊车
        ("/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM", "class", "Crane"),
        
        # 卡车
        ("/World/GroundPlane/tn__09684481_", "class", "Dumper"),
        
        # 人物
        ("/World/GroundPlane/DHGen", "class", "Human"),
        ("/World/GroundPlane/DHGen/SkelRoot", "class", "Human"),
        
        # 树木
        ("/World/Tree", "class", "Tree"),
    ]
    
    # 添加围栏
    fence_base = "/World/GroundPlane/Construction_Site_Construction_Zeppelin_Rental_GmbH_Metal_Construction_Site_Fencing_height_"
    for i in [2] + list(range(3, 26)):
        suffix = str(i) if i == 2 else f"{i:02d}"
        objects_to_label.append((f"{fence_base}{suffix}", "class", "Fence"))
    
    # 添加树木子对象
    for i in range(12):
        suffix = "" if i == 0 else f"_{i:02d}"
        objects_to_label.append((f"/World/Tree/Tree{suffix}", "class", "Tree"))
    
    labeled_count = 0
    failed_count = 0
    
    print("\n📋 添加语义标签...")
    print("-" * 80)
    
    def add_semantic_label(prim, semantic_type, semantic_data):
        """使用多种方法尝试添加语义标签"""
        success = False
        
        # 方法1: 使用Semantics Schema API (Isaac Sim推荐)
        if has_semantics_module:
            try:
                # 检查是否已有Semantics API
                sem_api = Semantics.SemanticsAPI.Get(prim, "Semantics")
                if not sem_api:
                    # 应用Semantics Schema
                    sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                
                if sem_api:
                    sem_api.CreateSemanticTypeAttr().Set(semantic_type)
                    sem_api.CreateSemanticDataAttr().Set(semantic_data)
                    success = True
            except Exception as e:
                print(f"    方法1失败: {e}")
        
        # 方法2: 直接设置属性 (备用方法)
        if not success:
            try:
                # Isaac Sim使用的属性路径
                type_attr = prim.GetAttribute("semantic:Semantics:params:semanticType")
                if not type_attr:
                    type_attr = prim.CreateAttribute(
                        "semantic:Semantics:params:semanticType",
                        Sdf.ValueTypeNames.String
                    )
                type_attr.Set(semantic_type)
                
                data_attr = prim.GetAttribute("semantic:Semantics:params:semanticData")
                if not data_attr:
                    data_attr = prim.CreateAttribute(
                        "semantic:Semantics:params:semanticData",
                        Sdf.ValueTypeNames.String
                    )
                data_attr.Set(semantic_data)
                success = True
            except Exception as e:
                print(f"    方法2失败: {e}")
        
        # 方法3: 使用另一种属性格式
        if not success:
            try:
                type_attr = prim.CreateAttribute(
                    "semantics:Semantics:params:semanticType",
                    Sdf.ValueTypeNames.String
                )
                type_attr.Set(semantic_type)
                
                data_attr = prim.CreateAttribute(
                    "semantics:Semantics:params:semanticData", 
                    Sdf.ValueTypeNames.String
                )
                data_attr.Set(semantic_data)
                success = True
            except Exception as e:
                print(f"    方法3失败: {e}")
        
        return success
    
    for prim_path, semantic_type, semantic_data in objects_to_label:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        
        if add_semantic_label(prim, semantic_type, semantic_data):
            labeled_count += 1
            print(f"  ✓ {prim_path} -> {semantic_data}")
        else:
            failed_count += 1
            print(f"  ✗ {prim_path} - 添加失败")
    
    # 额外：为所有Mesh子对象也添加标签（某些annotator需要Mesh级别的标签）
    print("\n📋 为Mesh子对象添加标签...")
    print("-" * 80)
    
    mesh_labeled = 0
    for prim_path, semantic_type, semantic_data in objects_to_label:
        parent_prim = stage.GetPrimAtPath(prim_path)
        if not parent_prim:
            continue
        
        # 遍历所有子Mesh
        for descendant in Usd.PrimRange(parent_prim):
            if descendant.GetTypeName() == "Mesh":
                if add_semantic_label(descendant, semantic_type, semantic_data):
                    mesh_labeled += 1
                    # 只显示前几个
                    if mesh_labeled <= 10:
                        print(f"  ✓ {descendant.GetPath()} -> {semantic_data}")
    
    if mesh_labeled > 10:
        print(f"  ... 还有 {mesh_labeled - 10} 个Mesh")
    
    print("\n" + "="*80)
    print(f"总结:")
    print(f"  父对象标签: {labeled_count} 个")
    print(f"  Mesh标签: {mesh_labeled} 个")
    print(f"  失败: {failed_count} 个")
    print("="*80)
    
    if labeled_count > 0 or mesh_labeled > 0:
        print("\n✅ 语义标签已添加！")
        print("\n⚠️  重要步骤：")
        print("  1. 保存场景: Ctrl+S")
        print("  2. 关闭Isaac Sim，重新打开场景")
        print("  3. 重新运行数据生成脚本")

print("\n脚本执行完成")

