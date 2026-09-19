# Non-contrast Cine CMR for Predicting LGE Presence in Hypertrophic Cardiomyopathy (CMR-HCM)
## 基于非增强 cine CMR 预测肥厚型心肌病 LGE presence (心肌晚期钆增强状态)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 1.11+](https://img.shields.io/badge/PyTorch-1.11%2B-red.svg)](https://pytorch.org/)
[![R: 4.0+](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)]()

---

## 1. 项目简介 (Overview)

晚期钆增强 (Late Gadolinium Enhancement, LGE) 是评估肥厚型心肌病 (HCM) 心肌纤维化程度的临床金标准，但需要注射钆对比剂，在肾功能受损等特殊人群中应用受限。

本开源项目旨在构建一套基于**非增强短轴动态 cine CMR** 的多模态融合预测体系，实现无需钆对比剂直接预测 **LGE presence（晚期钆增强阳性状态, `LGE_status`）**。

项目核心构成包括：
1. **非增强 cine 影像组学全流程 (Radiomics & R-score)**：
   - 从**舒张末期 (End-diastolic, ED) 短轴 cine 的 LV myocardium ROI (左心室心肌感兴趣区)** 中提取高维影像组学特征；
   - 经过观察者内与观察者间双盲 ICC(2,1) 一致性筛选 (ICC ≥ 0.80)；
   - 采用**重复嵌套交叉验证弹性网络 (Repeated Nested CV Elastic Net)** 在训练折内无偏调优混合参数 $\alpha$ 与惩罚参数 $\lambda$，构建影像组学评分 (R-score)。
2. **临床 CMR 参数筛选 (LASSO)**：
   - 基于 10 折交叉验证惩罚逻辑回归 (LASSO, $\lambda_{1\text{se}}$) 从临床 CMR 候选参数中筛选核心表征，包括最大室间隔厚度 (`MWT`)、**左心房储藏期应变 (`LAs`, left atrial reservoir strain)**、左室整体放射状应变 (`LVGRS`) 以及左室整体纵向应变 (`LVGLS`)。
3. **深度时序基础模型 (CMR Transformer / MViT)**：
   - 从短轴全时序动态 cine 序列中直接提取深层时空心功能表征，并通过切片级注意力多示例学习 (Attention-MIL) 完成端到端微调。
4. **经典机器学习基线与多模态融合 (Multimodal Fusion)**：
   - 涵盖 SVM、Bagging、XGBoost、LightGBM、CatBoost 五大经典分类器，融合临床 CMR 参数、弹性网络 R-score 及深度基础表征。
5. **完备的统计检验与可解释性分析**：
   - 包含 Bootstrap 95% 置信区间、DeLong 检验、净重新分类指数 (NRI)、综合判别改善指数 (IDI)、校准曲线 (Calibration Curve)、临床决策曲线 (DCA) 以及 SHAP 全局与局部特征归因。

---

## 2. 核心架构与目录组织 (Repository Structure)

```text
CMR_HCM_OpenSource/
├── README.md                           # 开源项目主文档 (中英文使用指南)
├── requirements.txt                    # Python 依赖清单
├── environment.yaml                    # Conda 环境复现配置
├── LICENSE                             # MIT 开源协议
│
├── configs/                            # 配置文件
│   └── default_config.yaml             # 包含超参数、训练配置及相对路径模板
│
├── data_preparation/                   # 【模块一：数据预处理与质控】
│   ├── build_manifest.py               # 构建队列患者清单 (开发集 & 外部验证集)
│   ├── audit_sax_retrieval.py          # 短轴 cine 序列检索、多序列冲突判定与审计
│   ├── prepare_sax_nifti.py            # DICOM 转 NIfTI 序列与切片空间几何校验
│   ├── merge_sax_qc.py                 # 汇总质控状态与切片质量控制 (Slice QC)
│   ├── audit_auto_myo_masks.py         # 自动心肌分割掩膜完整性校验
│   └── extract_acquisition_metadata.py # 提取厂商、机型、场强等扫描技术参数
│
├── radiomics/                          # 【模块二：影像组学全流程】
│   ├── feature_extraction.py           # 舒张末期短轴 cine 左室心肌 (LV myocardium) ROI 特征提取
│   ├── icc_reproducibility.R           # ICC(2,1) 双人重测特征可重复性与稳定性筛选
│   ├── compare_icc_thresholds.py       # ICC 截断值 (0.80 vs 0.75) 敏感性比较
│   ├── rscore_nested_oof_elasticnet.R  # 10折重复嵌套交叉验证弹性网络 (Elastic Net) R-score 构建
│   ├── build_rscore_oof.py             # Python 版严控信息泄露 R-score 构建
│   ├── cmr_parameter_lasso.R           # 10折交叉验证 LASSO 筛选核心 CMR 临床参数 (含 LAs, MWT 等)
│   └── cmr_lasso_vif.R                 # 临床参数 LASSO 筛选与多重共线性 VIF 分析
│
├── models/                             # 【模块三：模型构建与训练】
│   ├── classical_ml/                   # 经典机器学习分类器
│   │   ├── run_tabular_benchmark.py    # 5 大分类器 (SVM, Bagging, XGBoost, LightGBM, CatBoost) 重复嵌套CV与独立测试
│   │   └── catboost_training.py        # CatBoost 独立训练与最优超参数搜索
│   └── deep_foundation/                # 基础模型与深度学习
│       ├── arch/                       # 核心网络骨干架构
│       │   ├── mvit.py                 # 多尺度时序 Transformer (MViT / CMR Transformer)
│       │   └── resnet.py               # 3D/2D CNN 骨干与注意力模块
│       ├── model_factory.py            # 预训练模型加载与权重初始化适配器
│       ├── dataset.py                  # PyTorch 时序短轴 cine 数据集与时空数据增广
│       ├── extract_embeddings.py       # 冻结基础模型特征提取 (支持多GPU分片)
│       ├── train_attention_mil.py      # 全时序端到端微调与切片注意力多示例学习 (Attention-MIL)
│       └── compare_temporal_ablation.py# 动态全时序 vs 舒张末期 (ED) 单帧时序消融对比
│
├── multimodal_fusion/                  # 【模块四：多模态融合与综合统计评估】
│   ├── run_foundation_comparison.py    # 基础模型表征 + 临床参数 + 影像组学 R-score 融合消融
│   ├── build_complete_comparison.py    # 整合所有模型预测值 (统一输出与指标对齐)
│   ├── run_additional_statistics.py    # 深度统计评估 (AUC 95% CI, DeLong, NRI, IDI, 校准曲线, DCA)
│   ├── analyze_primary_xgboost_shap.py # 主模型 SHAP 全局重要性与局部依赖特征解释
│   ├── analyze_subgroup_performance.py # 亚组分层分析 (中心、场强、设备品牌等)
│   ├── analyze_technical_robustness.py # 扫描参数与技术稳健性检验
│   └── analyze_clinical_impact.py      # 预设高灵敏度操作点下的临床净获益评估
│
├── scripts/                            # 【模块五：批处理与自动化流水线】
│   └── run_pipeline.sh                 # 端到端一键运行全流程脚本
│
└── data_template/                      # 【模块六：输入模板与格式说明】
    ├── patient_manifest_template.csv   # 患者信息、临床变量与 LGE_status 标签模板
    └── slice_qc_template.csv           # 切片质控与位置信息模板
```

---

## 3. 环境配置与安装 (Installation)

### 方式一：Conda 环境一键安装 (推荐)

```bash
# 1. 克隆代码仓库
git clone https://github.com/Oubit1/CMR_HCM_OpenSource.git
cd CMR_HCM_OpenSource

# 2. 通过 environment.yaml 创建独立 Conda 环境
conda env create -f environment.yaml
conda activate cmr_hcm
```

### 方式二：Pip 安装核心依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*若需运行 R 语言脚本（`rscore_nested_oof_elasticnet.R`、`cmr_parameter_lasso.R`、`icc_reproducibility.R`），需在 R 环境中安装：*
```R
install.packages(c("readxl", "irr", "dplyr", "glmnet", "caret", "car", "pROC", "openxlsx"))
```

---

## 4. 模型权重下载与配置 (Model Weights & Checkpoints)

本项目提供预训练的基础模型骨干权重及锁定训练好的最终模型。所有权重文件均已托管发布至 GitHub Release：

👉 **[下载地址：GitHub Release v1.0.0 页面](https://github.com/Oubit1/CMR_HCM_OpenSource/releases/tag/v1.0.0)**

| 权重文件名 | 文件大小 | 推荐存放目录 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| **`pretrained_cmr_backbone.ckpt`** | ~1.32 GB | `./checkpoints/` | 基础 CMR 时序 Transformer 预训练编码器骨干权重（用于微调训练 `train_attention_mil.py` 的初始化，模型骨干架构源自 *A Generalizable Deep Learning System for Cardiac MRI*） |
| **`CatBoost_final.cbm`** | 267 KB | `./checkpoints/` | 锁定的最终 CatBoost 机器学习分类器模型（输入特征：`MWT`, `LAs`, `LVGRS`, `LVGLS`, `Rscore`） |
| **`Rscore_locked_model.rds`** | 1.39 MB | `./checkpoints/` | 锁定的最终弹性网络影像组学评分模型 (R-score Model Object) |

#### 快速下载与部署指引：
```bash
# 在项目根目录下创建 checkpoints 目录
mkdir -p checkpoints

# 使用 wget 命令行直接下载到指定目录
wget -O checkpoints/pretrained_cmr_backbone.ckpt https://github.com/Oubit1/CMR_HCM_OpenSource/releases/download/v1.0.0/pretrained_cmr_backbone.ckpt
wget -O checkpoints/CatBoost_final.cbm https://github.com/Oubit1/CMR_HCM_OpenSource/releases/download/v1.0.0/CatBoost_final.cbm
wget -O checkpoints/Rscore_locked_model.rds https://github.com/Oubit1/CMR_HCM_OpenSource/releases/download/v1.0.0/Rscore_locked_model.rds
```

---

## 5. 数据准备与输入模板 (Data Preparation)

由于患者医疗数据隐私合规要求，原始影像不随代码公开发布。您只需按照 `data_template/` 中的样例格式组织本地数据：

### 5.1 患者总清单 (`patient_manifest.csv`)
格式参考 `data_template/patient_manifest_template.csv`：
- `ID`: 患者唯一识别编号
- `cohort`: 所属队列 (`development` 或 `external_test`)
- `LGE_status`: **目标预测终点**（`1`: LGE presence / 阳性, `0`: LGE 阴性 / Absence）
- `Rscore`: 舒张末期左室心肌 (LV myocardium) 提取并经弹性网络构建的影像组学评分 (连续值)
- `MWT`: 最大室间隔厚度 (Maximal Wall Thickness, mm)
- `LAs`: **左心房储藏期应变 (left atrial reservoir strain, %)**
- `LVGRS`: 左室整体放射状应变 (Global Radial Strain, %)
- `LVGLS`: 左室整体纵向应变 (Global Longitudinal Strain, %)
- `image_path`: 患者原始 DICOM 目录或对应 NIfTI 路径

### 5.2 切片质控与位置清单 (`slice_qc.csv`)
格式参考 `data_template/slice_qc_template.csv`，记录短轴 cine 序列中各有效切片（Slice）的空间位置与对应文件路径。

---

## 6. 分步复现操作指南 (Step-by-Step Guide)

### 阶段 1: 数据预处理与短轴质控
```bash
# 1. 构建患者清单
python data_preparation/build_manifest.py \
  --train-table ./data/CMR_parameter/train.xlsx \
  --external-table ./data/CMR_parameter/external.xlsx \
  --output-csv ./data/patient_manifest.csv

# 2. 转换 DICOM 为短轴 Cine NIfTI
python data_preparation/prepare_sax_nifti.py \
  --manifest ./data/patient_manifest.csv \
  --output-dir ./data/processed_sax_cine
```

### 阶段 2: 影像组学全流程与 CMR 参数筛选
```bash
# 1. (可选) 从舒张末期 (ED) 短轴 cine 左室心肌 (LV myocardium) ROI 提取高维影像组学特征
python radiomics/feature_extraction.py \
  --data-list ./data/radiomics_cases.csv \
  --output-excel ./data/Radiomics_extracted.xlsx

# 2. 执行双人 ICC(2,1) 重测一致性筛选 (R 语言)
Rscript radiomics/icc_reproducibility.R

# 3. 运行 10 折重复嵌套交叉验证弹性网络 (Elastic Net) 构建 R-score (R 语言)
Rscript radiomics/rscore_nested_oof_elasticnet.R

# 4. 运行 10 折交叉验证 LASSO 筛选核心 CMR 临床参数 (R 语言)
Rscript radiomics/cmr_parameter_lasso.R
```

### 阶段 3: 经典机器学习 5 大分类器评估
```bash
# 运行 SVM, Bagging, XGBoost, LightGBM, CatBoost 重复嵌套交叉验证
python models/classical_ml/run_tabular_benchmark.py \
  --manifest ./data/patient_manifest.csv \
  --output-dir ./results/tabular_benchmark \
  --outer-folds 5 \
  --outer-repeats 5 \
  --bootstraps 2000
```

### 阶段 4: 基础模型微调与 Attention-MIL
```bash
# 全时序 Attention-MIL 端到端微调训练
python models/deep_foundation/train_attention_mil.py \
  --manifest ./data/patient_manifest.csv \
  --slice-qc ./data/processed_sax_cine/slice_qc.csv \
  --checkpoint ./checkpoints/pretrained_cmr_backbone.ckpt \
  --output-dir ./results/foundation_mil \
  --arms Foundation \
  --device cuda
```

### 阶段 5: 多模态融合对比与时序消融
```bash
# 运行 CMR / Rscore+CMR / Foundation / Foundation+CMR / Foundation+Rscore+CMR 全面消融
python multimodal_fusion/run_foundation_comparison.py \
  --manifest ./data/patient_manifest.csv \
  --embeddings ./results/foundation_embeddings/patient_embeddings.npz \
  --output-dir ./results/fusion_comparison
```

### 阶段 6: 顶级期刊级统计评估与 SHAP 可解释性
```bash
# 1. 计算 DeLong 检验、NRI、IDI、校准曲线与 DCA
python multimodal_fusion/run_additional_statistics.py \
  --development-predictions ./results/fusion_comparison/development_oof_predictions.csv \
  --external-predictions ./results/fusion_comparison/external_test_predictions.csv \
  --output-dir ./results/statistical_evaluation

# 2. 计算主模型 SHAP 特征重要性蜂群图与依赖图
python multimodal_fusion/analyze_primary_xgboost_shap.py \
  --output-dir ./results/shap_analysis
```

---

## 7. 一键运行自动化流水线 (One-Click Pipeline)

本项目提供了预先封装好的端到端运行脚本：

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

---

## 8. 开源协议与引用 (License & Citation)

本项目采用 [MIT 许可证](LICENSE)。

如果您在学术研究中使用了本项目代码或思路，请在论文发表后引用我们的研究工作：
```bibtex

```
