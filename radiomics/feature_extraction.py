#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# 模块二：影像组学全流程 - 舒张末期短轴 Cine 序列左心室心肌 (LV Myocardium) PyRadiomics 特征提取 (feature_extraction.py)
# ==============================================================================
# 功能描述:
#     本脚本基于 SimpleITK 与 PyRadiomics 库，实现对非增强短轴动态 Cine CMR
#     在舒张末期 (End-diastolic, ED) 的左心室心肌 (LV Myocardium) ROI 进行高维影像组学特征提取。
#     包含完整的图像预处理：
#       1. 多通道/单通道维度转换 (Vector to Scalar)
#       2. 统一物理空间分辨率重采样 (Resampling: 1.0 x 1.0 x 8.0 mm)
#       3. 像素灰度强度分位数裁剪与归一化 (Intensity Normalization: 1%-99% -> 0-255)
#       4. 提取原始特征 (Original)、拉普拉斯-高斯滤波 (LoG: sigma 1.0, 2.0, 3.0 mm)
#          以及小波变换 (Wavelet) 多尺度特征。
#
# 输入:
#     - image_path: 患者舒张末期短轴 Cine (ED SAX Cine) NIfTI 图像路径
#     - mask_path: 医生/模型勾画的左室心肌 (LV Myocardium) ROI 掩膜 NIfTI 路径
#
# 输出:
#     - 包含 1000+ 维影像组学特征字典 (形状、一阶统计、GLCM, GLRLM, GLSZM, GLDM, NGTDM 等)
# ==============================================================================

import argparse
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
from tqdm import tqdm


def convert_vector_to_scalar(image: sitk.Image) -> sitk.Image:
    """
    将多组件/多通道 (Vector) 图像转换为单通道 (Scalar) 灰度图像。
    若图像每个像素包含多个组件 (如 RGB)，则提取第 0 个通道。
    """
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)
    return image


def preprocess_image(image: sitk.Image, spacing=(1.0, 1.0, 8.0)) -> sitk.Image:
    """
    预处理图像：空间重采样 (Resampling)。
    将图像的体素间距统一为指定的物理间距 (默认 x=1.0 mm, y=1.0 mm, z=8.0 mm)。
    这一步保证了不同患者、不同中心采集的图像具有统一的物理空间分辨率。
    """
    image = sitk.Resample(
        image,
        image.GetSize(),
        sitk.Transform(),
        sitk.sitkLinear,       # 线性插值
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID()
    )
    return image


def normalize_intensity(image: sitk.Image) -> sitk.Image:
    """
    对图像像素灰度强度进行归一化处理：
      1. 计算 1% 和 99% 的灰度百分位数。
      2. 将极端离群值裁剪 (Clip) 到 [P1, P99] 范围内，有效消除高亮伪影与噪声。
      3. 将像素灰度线性映射到 [0, 255] 范围。
    """
    import numpy as np

    array = sitk.GetArrayFromImage(image).astype(float)
    lower = np.percentile(array, 1)
    upper = np.percentile(array, 99)

    array = np.clip(array, lower, upper)

    if upper > lower:
        array = (array - lower) / (upper - lower) * 255.0

    output = sitk.GetImageFromArray(array)
    output.CopyInformation(image)
    return output


def create_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """
    创建并配置 PyRadiomics 影像组学特征提取器。
    按 IBSI (Image Biomarker Standardisation Initiative) 标准进行参数配置。
    """
    settings = {
        "binWidth": 5,           # 灰度离散化固定 bin 宽度
        "force2D": True,         # 心脏短轴图像按 2D 切片方式提取
        "force2Ddimension": 0    # 指定沿 Z 轴切片 (0 表示切片轴)
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # 禁用所有默认特征，后面手动按需启用以保证特征完备性
    extractor.disableAllFeatures()

    # 启用全部 7 大类影像组学特征
    extractor.enableFeatureClassByName("firstorder") # 一阶直方图统计特征
    extractor.enableFeatureClassByName("shape")      # 形状与形态学特征
    extractor.enableFeatureClassByName("glcm")       # 灰度共生矩阵 (GLCM, 纹理)
    extractor.enableFeatureClassByName("glrlm")      # 灰度游程矩阵 (GLRLM)
    extractor.enableFeatureClassByName("glszm")      # 灰度区域大小矩阵 (GLSZM)
    extractor.enableFeatureClassByName("gldm")       # 灰度相关矩阵 (GLDM)
    extractor.enableFeatureClassByName("ngtdm")      # 邻域灰度差分矩阵 (NGTDM)

    # 启用原图特征
    extractor.enableImageTypeByName("Original")

    # 启用高斯-拉普拉斯滤波多尺度特征 (LoG, sigma = 1.0, 2.0, 3.0 mm)
    extractor.enableImageTypeByName(
        "LoG",
        customArgs={"sigma": [1.0, 2.0, 3.0]}
    )

    # 启用小波变换滤波特征 (Wavelet, 包含 LL, LH, HL, HH 各子带)
    extractor.enableImageTypeByName("Wavelet")

    return extractor


def extract_features_single(image_path: Path, mask_path: Path) -> dict:
    """
    对单例病例提取特征：给定舒张末期短轴 cine 图像与左室心肌 (LV Myocardium) ROI 掩膜，提取全部组学特征。
    """
    # 1. 读取 NIfTI 图像与掩膜
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    # 2. 执行标准预处理
    image = convert_vector_to_scalar(image)
    image = preprocess_image(image)
    image = normalize_intensity(image)

    # 3. 创建提取器并执行提取
    extractor = create_extractor()
    result = extractor.execute(image, mask)

    # 4. 过滤非特征的诊断/元数据信息
    features = {}
    for key, value in result.items():
        if not key.startswith("diagnostics"):
            features[key] = float(value) if hasattr(value, "__float__") else value

    return features


def parse_args():
    parser = argparse.ArgumentParser(description="批量执行舒张末期短轴 Cine 左室心肌影像组学特征提取")
    parser.add_argument(
        "--data-list",
        type=Path,
        default=Path("./data/radiomics_cases.csv"),
        help="包含 ID, image_path, mask_path 的 CSV 列表文件路径"
    )

    parser.add_argument(
        "--output-excel",
        type=Path,
        default=Path("./data/Radiomics_extracted.xlsx"),
        help="提取出的特征表输出 Excel 路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("PyRadiomics 特征提取器初始化完成")
    print(f"数据输入列表: {args.data_list}")
    print(f"特征输出路径: {args.output_excel}")
    print("=" * 60)
    
    if not args.data_list.exists():
        print(f"[提示] 未找到输入文件 {args.data_list}。此处提供单例提取接口 `extract_features_single()` 供调用。")
        return

    df = pd.read_csv(args.data_list)
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取组学特征"):
        pid = row["ID"]
        img_p = Path(row["image_path"])
        mask_p = Path(row["mask_path"])
        
        try:
            feats = extract_features_single(img_p, mask_p)
            feats["ID"] = pid
            records.append(feats)
        except Exception as e:
            print(f"【警告】患者 {pid} 特征提取失败: {e}")
            
    if records:
        out_df = pd.DataFrame(records)
        # 将 ID 列移至首列
        cols = ["ID"] + [c for c in out_df.columns if c != "ID"]
        out_df = out_df[cols]
        args.output_excel.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_excel(args.output_excel, index=False)
        print(f"提取完成！共保存 {len(out_df)} 例患者的 {len(cols)-1} 个影像组学特征。")


if __name__ == "__main__":
    main()
