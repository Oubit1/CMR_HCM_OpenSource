#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
模块一：数据预处理与质控 - 患者队列清单构建与数据校验 (build_manifest.py)
================================================================================
功能描述:
    本脚本负责将临床指标数据表 (开发集 train.xlsx / 外部测试集 external.xlsx) 与对应的
    CMR 原始影像目录进行对齐，校验必填临床变量与结局指标，并生成统一的患者清单 (manifest.csv)。

输入:
    - 开发队列临床表格: 包含 ID, event, Rscore, MWT, LAs, LVGRS, LVGLS 等字段
    - 外部验证队列临床表格: 格式同上
    - 原始影像存放根目录

输出:
    - patient_manifest.csv (包含每位患者的基本信息、临床特征、标签及影像路径)
================================================================================
"""

import argparse
from pathlib import Path
import pandas as pd


# 必填核心特征与结局标签
REQUIRED_COLUMNS = [
    "ID",       # 患者唯一识别编号
    "event",    # 临床预后终点 (0: 未发生, 1: 发生不良预后事件)
    "Rscore",   # 影像组学 R-score 评分
    "MWT",      # 最大室间隔厚度 (Maximal Wall Thickness)
    "LAs",      # 左心房收缩末期面积 (Left Atrial Systolic Area)
    "LVGRS",    # 左室整体放射状应变 (Global Radial Strain)
    "LVGLS",    # 左室整体纵向应变 (Global Longitudinal Strain)
]


def parse_args():
    """解析命令行参数，支持用户传入自定义路径"""
    parser = argparse.ArgumentParser(description="构建并校验 HCM 研究多中心队列的患者清单 (Manifest)")
    
    # 路径参数设置 (默认采用相对路径，方便开源用户直接运行)
    parser.add_argument(
        "--train-table",
        type=Path,
        default=Path("./data/CMR_parameter/train.xlsx"),
        help="开发集临床特征与标签 Excel 表格路径 (例如: ./data/CMR_parameter/train.xlsx)",
    )
    parser.add_argument(
        "--external-table",
        type=Path,
        default=Path("./data/CMR_parameter/external.xlsx"),
        help="外部测试集临床特征与标签 Excel 表格路径 (例如: ./data/CMR_parameter/external.xlsx)",
    )
    parser.add_argument(
        "--train-image-dir",
        type=Path,
        default=Path("./data/raw_dicom/train"),
        help="开发集原始 DICOM 影像目录路径",
    )
    parser.add_argument(
        "--external-image-dir",
        type=Path,
        default=Path("./data/raw_dicom/external"),
        help="外部测试集原始 DICOM 影像目录路径",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("./data/patient_manifest.csv"),
        help="生成的患者清单输出保存路径 (默认: ./data/patient_manifest.csv)",
    )
    
    return parser.parse_args()


def get_patient_dir(image_root: Path, patient_id: int, zero_pad: int = 0) -> Path:
    """根据患者 ID 与补零规则获取对应影像子目录"""
    numeric_id = int(patient_id)
    dir_name = str(numeric_id).zfill(zero_pad) if zero_pad > 0 else str(numeric_id)
    return image_root / dir_name


def load_and_validate_cohort(
    cohort_name: str,
    table_path: Path,
    image_root: Path,
    zero_pad: int = 0
) -> pd.DataFrame:
    """
    加载并校验单个队列的临床数据与影像目录有效性。
    """
    if not table_path.exists():
        raise FileNotFoundError(f"【错误】找不到队列 [{cohort_name}] 的临床表格: {table_path}")
    
    # 读取 Excel
    df = pd.read_excel(table_path)
    
    # 字段兼容性处理 (如果表格历史列名为 maxLAV，自动转换为 LAs)
    if "maxLAV" in df.columns and "LAs" not in df.columns:
        print(f"[{cohort_name}] 检测到历史列名 maxLAV，已自动更名为 LAs")
        df = df.rename(columns={"maxLAV": "LAs"})
        
    # 检查必填字段
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{cohort_name}] 表格缺少必填字段: {missing_cols}")
        
    # 检查 ID 唯一性
    if df["ID"].duplicated().any():
        raise ValueError(f"[{cohort_name}] 表格中存在重复的患者 ID！")
        
    # 校验二分类标签
    valid_events = {0, 1}
    actual_events = set(df["event"].dropna().unique())
    if not actual_events.issubset(valid_events):
        raise ValueError(f"[{cohort_name}] event 标签中存在非 0/1 的无效值: {actual_events}")

    # 提取所需列
    clean_df = df[REQUIRED_COLUMNS].copy()
    clean_df["cohort"] = cohort_name
    
    # 检查患者影像目录是否存在
    image_paths = []
    exists_flags = []
    for pid in clean_df["ID"]:
        p_dir = get_patient_dir(image_root, pid, zero_pad)
        image_paths.append(str(p_dir))
        exists_flags.append(p_dir.exists())
        
    clean_df["image_path"] = image_paths
    clean_df["image_exists"] = exists_flags
    
    # 输出统计信息
    n_total = len(clean_df)
    n_img_found = sum(exists_flags)
    n_pos = int(clean_df["event"].sum())
    print(f"[{cohort_name}] 成功加载 {n_total} 例患者 | 阳性事件: {n_pos} ({n_pos/n_total:.1%}) | 影像存在: {n_img_found}/{n_total}")
    
    return clean_df


def main():
    args = parse_args()
    
    print("=" * 60)
    print("开始构建患者清单 (Patient Manifest)...")
    print("=" * 60)
    
    # 1. 加载开发集
    dev_df = load_and_validate_cohort(
        cohort_name="development",
        table_path=args.train_table,
        image_root=args.train_image_dir,
        zero_pad=0
    )
    
    # 2. 加载外部测试集
    ext_df = load_and_validate_cohort(
        cohort_name="external_test",
        table_path=args.external_table,
        image_root=args.external_image_dir,
        zero_pad=3 # 部分中心外部验证 ID 采用 3 位补零命名 (如 001, 002)
    )
    
    # 3. 合并清单
    manifest = pd.concat([dev_df, ext_df], ignore_index=True)
    
    # 调整列顺序
    cols_order = ["ID", "cohort", "event", "Rscore", "MWT", "LAs", "LVGRS", "LVGLS", "image_path", "image_exists"]
    manifest = manifest[cols_order]
    
    # 保存输出
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False, encoding="utf-8")
    
    print("=" * 60)
    print(f"清单构建完成！共收录 {len(manifest)} 例患者。")
    print(f"文件已保存至: {args.output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
