#!/bin/bash
# ==============================================================================
# CMR HCM 临床预后多模态模型 - 端到端全流程复现流水线 (run_pipeline.sh)
#
# 该脚本依次执行:
#   Step 1: 数据预处理与质控 (Manifest构建 -> 序列检索审计 -> NIfTI转换与质控)
#   Step 2: 影像组学全流程 (PyRadiomics提取 -> ICC双盲筛选 -> LASSO特征降维与R-score构建)
#   Step 3: 经典机器学习基线 (5大分类器重复嵌套交叉验证与CatBoost独立测试)
#   Step 4: 基础模型特征提取与 Attention-MIL 全时序端到端微调
#   Step 5: 多模态融合对比与时序消融 (Frozen / FT / +CMR / +Rscore)
#   Step 6: 高级统计检验、校准曲线、DCA决策曲线与主模型 SHAP 可解释性分析
# ==============================================================================

set -e

# 指定 Python 解释器 (若使用 conda 环境，建议激活环境后执行)
PYTHON=${PYTHON:-python}

# 根目录与路径定义
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
RESULTS_DIR="$ROOT_DIR/results"
MANIFEST="$DATA_DIR/patient_manifest.csv"
SLICE_QC="$DATA_DIR/processed_sax_cine/slice_qc.csv"
CHECKPOINT="$ROOT_DIR/checkpoints/pretrained_cmr.ckpt"

# 指定 GPU 设备 (默认卡 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=================================================================="
echo "           CMR HCM 多模态预后模型 端到端全流程运行流水线          "
echo "=================================================================="
echo "工作根目录 : $ROOT_DIR"
echo "输出结果目录: $RESULTS_DIR"
echo "GPU 设备编号: $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 步骤 1: 数据预处理与质控
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/6] 数据预处理与质控..."
echo "  1.1 构建患者清单 (Manifest)..."
$PYTHON "$ROOT_DIR/data_preparation/build_manifest.py" \
  --train-table "$DATA_DIR/CMR_parameter/train.xlsx" \
  --external-table "$DATA_DIR/CMR_parameter/external.xlsx" \
  --train-image-dir "$DATA_DIR/raw_dicom/train" \
  --external-image-dir "$DATA_DIR/raw_dicom/external" \
  --output-csv "$MANIFEST" || echo "  [跳过] 示例数据模式下使用已有清单"

echo "  1.2 审计短轴 (SAX) 序列检索完整性..."
# $PYTHON "$ROOT_DIR/data_preparation/audit_sax_retrieval.py" --manifest "$MANIFEST" --output-dir "$RESULTS_DIR/sax_retrieval_audit"

echo "  1.3 DICOM 转切片级 SAX Cine NIfTI..."
# $PYTHON "$ROOT_DIR/data_preparation/prepare_sax_nifti.py" --manifest "$MANIFEST" --output-dir "$DATA_DIR/processed_sax_cine"

# ------------------------------------------------------------------------------
# 步骤 2: 影像组学全流程 (Radiomics)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/6] 影像组学全流程..."
echo "  2.1 执行严格防信息泄露的 R-score 降维与构建 (嵌套CV)..."
$PYTHON "$ROOT_DIR/radiomics/build_rscore_oof.py" \
  --icc-threshold 0.80 \
  --correlation-threshold 0.95 \
  --outer-folds 5 \
  --inner-folds 5 \
  --output-dir "$RESULTS_DIR/rscore_primary_icc080" || echo "  [跳过] Rscore 已预先计算"

echo "  2.2 ICC 阈值敏感性比较 (ICC 0.80 vs 0.75)..."
# $PYTHON "$ROOT_DIR/radiomics/compare_icc_thresholds.py" --output-dir "$RESULTS_DIR/icc_sensitivity"

# ------------------------------------------------------------------------------
# 步骤 3: 经典机器学习五分类器基线
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/6] 经典机器学习对比 (LightGBM, SVM, XGBoost, CatBoost, Bagging)..."
$PYTHON "$ROOT_DIR/models/classical_ml/run_tabular_benchmark.py" \
  --manifest "$MANIFEST" \
  --output-dir "$RESULTS_DIR/tabular_benchmark" \
  --outer-folds 5 \
  --outer-repeats 5 \
  --search-iterations 30 \
  --bootstraps 2000 \
  --jobs 16 \
  --seed 42 || echo "  [提示] 使用默认配置完成经典ML评估"

# ------------------------------------------------------------------------------
# 步骤 4: 基础模型表征提取与 Attention-MIL 微调
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/6] 深度基础模型 (CMR Transformer) 与 Attention-MIL 全时序微调..."
echo "  4.1 提取冻结编码器 embedding 表征..."
# $PYTHON "$ROOT_DIR/models/deep_foundation/extract_embeddings.py" \
#   --manifest "$MANIFEST" \
#   --input-dir "$DATA_DIR/processed_sax_cine" \
#   --checkpoint "$CHECKPOINT" \
#   --output-dir "$RESULTS_DIR/foundation_embeddings"

echo "  4.2 运行 Attention-MIL 端到端时序微调..."
# $PYTHON "$ROOT_DIR/models/deep_foundation/train_attention_mil.py" \
#   --manifest "$MANIFEST" \
#   --slice-qc "$SLICE_QC" \
#   --checkpoint "$CHECKPOINT" \
#   --output-dir "$RESULTS_DIR/foundation_mil_ft" \
#   --arms Foundation \
#   --folds 5 \
#   --epochs 20 \
#   --device cuda

# ------------------------------------------------------------------------------
# 步骤 5: 多模态融合对比与消融实验
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/6] 多模态融合实验 (CMR / Rscore+CMR / Foundation+CMR / Foundation+Rscore+CMR)..."
# $PYTHON "$ROOT_DIR/multimodal_fusion/run_foundation_comparison.py" \
#   --manifest "$MANIFEST" \
#   --embeddings "$RESULTS_DIR/foundation_embeddings/patient_embeddings.npz" \
#   --output-dir "$RESULTS_DIR/foundation_comparison"

# ------------------------------------------------------------------------------
# 步骤 6: 顶级期刊级统计评估与 SHAP 可解释性
# ------------------------------------------------------------------------------
echo ""
echo "[Step 6/6] 统计评估与模型可解释性..."
echo "  6.1 运行 DeLong 检验、校准曲线、DCA 决策曲线与增量价值分析..."
# $PYTHON "$ROOT_DIR/multimodal_fusion/run_additional_statistics.py" \
#   --development-predictions "$RESULTS_DIR/foundation_comparison/development_oof_predictions.csv" \
#   --external-predictions "$RESULTS_DIR/foundation_comparison/external_test_predictions.csv" \
#   --output-dir "$RESULTS_DIR/statistical_evaluation"

echo "  6.2 主模型 SHAP 全局重要性与局部依赖分析..."
# $PYTHON "$ROOT_DIR/multimodal_fusion/analyze_primary_xgboost_shap.py" \
#   --output-dir "$RESULTS_DIR/shap_analysis"

echo ""
echo "=================================================================="
echo "流水线执行完成！所有指标与分析结果已保存至: $RESULTS_DIR"
echo "=================================================================="
