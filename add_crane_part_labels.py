"""
为吊车部件添加语义标签（修复版）
在Isaac Sim Script Editor中运行
在运行 add_semantic_labels_v2.py 之后运行此脚本

修复说明：
- 使用与 add_semantic_labels_v2.py 一致的 API (semanticType + semanticData)
- 同时为子Mesh节点也添加标签（bbox_3d annotator需要Mesh级别标签）
- 三种备用方法确保兼容不同版本的Isaac Sim
"""

import omni.usd as usd
from pxr import Usd, UsdGeom, Sdf

print("="*80)
print("为吊车部件添加语义标签（修复版）")
print("="*80)

stage = usd.get_context().get_stage()
if stage is None:
    print("错误: 无法获取Stage")
else:
    print(f"✓ 场景: {stage.GetRootLayer().identifier}\n")

    # 尝试导入Semantics模块
    try:
        from pxr import Semantics
        has_semantics_module = True
        print("✓ 已加载 Semantics 模块")
    except ImportError:
        has_semantics_module = False
        print("⚠ 无法导入 Semantics 模块，使用备用方法")

    # 吊车部件分类（根据空间位置分析的结果）
    CRANE_PARTS = {
        'CraneBase': [
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104GG03A_SW',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104S01KB_SW',
        ],
        'CraneColumn': [
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104HZ01KA_SW',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104H01KB_SW',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104HZ02KA_SW',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104KZ01KA_SW',
        ],
        'CraneBoom': [
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/tn__S104EKB_AS_SW_jJ7',
        ],
        'CraneTelescopic': [
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/S104KZ02KA_SW',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/tn__HHK320KA_SW_lG',
            '/World/GroundPlane/tn__Pk7501SLD_PNR3879_fPM/tn__HHK319_SW_oD',
        ],
    }

    def add_semantic_label(prim, semantic_type, semantic_data):
        """
        使用多种方法尝试添加语义标签
        与 add_semantic_labels_v2.py 完全一致的API
        """
        success = False

        # 方法1: 使用Semantics Schema API (Isaac Sim推荐)
        if has_semantics_module:
            try:
                sem_api = Semantics.SemanticsAPI.Get(prim, "Semantics")
                if not sem_api:
                    sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")

                if sem_api:
                    sem_api.CreateSemanticTypeAttr().Set(semantic_type)
                    sem_api.CreateSemanticDataAttr().Set(semantic_data)
                    success = True
            except Exception as e:
                pass  # 尝试下一种方法

        # 方法2: 直接设置属性 (备用方法1)
        if not success:
            try:
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
                pass  # 尝试下一种方法

        # 方法3: 另一种属性格式 (备用方法2)
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
                pass

        return success

    # ===== 第1步: 为部件根节点添加标签 =====
    print("\n📋 第1步: 为吊车部件根节点添加标签...")
    print("-" * 80)

    total_count = 0
    success_count = 0

    for category, paths in CRANE_PARTS.items():
        print(f"\n[{category}] 添加标签中...")
        category_count = 0

        for prim_path in paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                if add_semantic_label(prim, "class", category):
                    success_count += 1
                    category_count += 1
                    print(f"  ✓ {prim.GetName()}")
                else:
                    print(f"  ✗ 失败: {prim.GetName()}")
                total_count += 1
            else:
                print(f"  ⚠ 不存在: {prim_path}")

        print(f"  完成: {category_count}/{len(paths)}")

    # ===== 第2步: 为所有子Mesh节点也添加标签 =====
    print("\n📋 第2步: 为子Mesh节点添加标签...")
    print("-" * 80)

    mesh_labeled = 0
    mesh_failed = 0

    for category, paths in CRANE_PARTS.items():
        category_mesh_count = 0

        for prim_path in paths:
            parent_prim = stage.GetPrimAtPath(prim_path)
            if not parent_prim or not parent_prim.IsValid():
                continue

            # 遍历所有子节点（包括Mesh和其他类型）
            for descendant in Usd.PrimRange(parent_prim):
                if add_semantic_label(descendant, "class", category):
                    mesh_labeled += 1
                    category_mesh_count += 1
                else:
                    mesh_failed += 1

        if category_mesh_count > 0:
            print(f"  [{category}] {category_mesh_count} 个子节点已标记")

    # ===== 汇总 =====
    print("\n" + "="*80)
    print(f"✓ 语义标签添加完成！")
    print(f"  部件根节点: {success_count}/{total_count} 成功")
    print(f"  子节点标记: {mesh_labeled} 成功, {mesh_failed} 失败")
    print(f"\n  部件分类:")
    for category, paths in CRANE_PARTS.items():
        print(f"    {category}: {len(paths)} 个根节点")
    print(f"\n⚠ 重要: 请保存场景 File -> Save (Ctrl+S)")
    print("="*80)
