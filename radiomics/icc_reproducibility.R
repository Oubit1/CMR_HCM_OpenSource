# ==============================================================================
# 模块二：影像组学全流程 - 特征可重复性与稳定性筛选 (ICC.R)
# ==============================================================================
# 模型类型: ICC(2,1), 双向随机效应模型 (Two-way random effects), 绝对一致性 (Absolute agreement)
# 目的:
#   评估观察者内 (Intra-observer) 与观察者间 (Inter-observer) 的特征提取一致性。
#   仅保留组内相关系数 ICC >= 0.80 (或敏感性阈值 0.75) 的高稳定性特征。
# ==============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(irr)
  library(dplyr)
})

# ----------------- 1. 文件与路径配置 (请修改为本地路径) -----------------
base_dir <- "./data/Radiomics/ICC"

# 输入文件:
# 1. 原始全量提取特征表
file_original <- file.path(base_dir, "Radiomics.xlsx")
# 2. 观察者 1 重复勾画提取的特征表 (用于计算观察者内 ICC)
file_reader1  <- file.path(base_dir, "Radiomics_ICC_reader1.xlsx")
# 3. 观察者 2 独立勾画提取的特征表 (用于计算观察者间 ICC)
file_reader2  <- file.path(base_dir, "Radiomics_ICC_reader2.xlsx")

# 输出文件:
# 1. 所有特征的 ICC 详细计算结果 CSV
output_detail   <- file.path(base_dir, "ICC_analysis_results.csv")
# 2. 经 ICC 阈值筛选后保留的高稳定性特征数据集 CSV
output_selected <- file.path(base_dir, "Radiomics_ICC_selected.csv")

# 研究方案设定的 ICC 截断阈值 (默认 >= 0.80 为极佳可重复性)
icc_threshold <- 0.80

# ----------------- 2. 检查并读取输入数据 -----------------
input_files <- c(file_original, file_reader1, file_reader2)
for (f in input_files) {
  if (!file.exists(f)) {
    warning(paste0("【提示】未找到文件: ", f, "。请确保数据放置在指定目录下。"))
  }
}

if (all(file.exists(input_files))) {
  original <- read_excel(file_original, sheet = 1, .name_repair = "unique")
  reader1  <- read_excel(file_reader1,  sheet = 1, .name_repair = "unique")
  reader2  <- read_excel(file_reader2,  sheet = 1, .name_repair = "unique")

  cat("原始数据集患者数:", nrow(original), "，特征数:", ncol(original) - 1, "\n")
  cat("观察者 1 复测样本数:", nrow(reader1), "，特征数:", ncol(reader1) - 1, "\n")
  cat("观察者 2 独立测试样本数:", nrow(reader2), "，特征数:", ncol(reader2) - 1, "\n")

  # 确定患者 ID 识别列
  get_id_col <- function(dat) {
    if ("Patient" %in% names(dat)) "Patient" else if ("ID" %in% names(dat)) "ID" else names(dat)[1]
  }

  id_original <- get_id_col(original)
  id_reader1  <- get_id_col(reader1)
  id_reader2  <- get_id_col(reader2)

  # 匹配共同患者 ID
  common_ids_intra <- intersect(original[[id_original]], reader1[[id_reader1]])
  common_ids_inter <- intersect(original[[id_original]], reader2[[id_reader2]])

  cat("观察者内匹配患者数:", length(common_ids_intra), "\n")
  cat("观察者间匹配患者数:", length(common_ids_inter), "\n")

  # 提取共有特征列表 (排除 ID 与标签列)
  feature_cols <- setdiff(names(original), c(id_original, "label", "event", "Label", "Event"))

  # ----------------- 3. 循环计算 ICC(2,1) -----------------
  results <- data.frame(
    Feature = character(),
    ICC_Intra = numeric(),
    ICC_Inter = numeric(),
    stringsAsFactors = FALSE
  )

  for (feat in feature_cols) {
    # 1) 计算观察者内 ICC
    sub_orig_intra <- original[match(common_ids_intra, original[[id_original]]), feat][[1]]
    sub_r1         <- reader1[match(common_ids_intra, reader1[[id_reader1]]), feat][[1]]
    
    mat_intra <- cbind(as.numeric(sub_orig_intra), as.numeric(sub_r1))
    icc_intra_val <- tryCatch({
      icc_res <- icc(mat_intra, model = "twoway", type = "agreement", unit = "single")
      icc_res$value
    }, error = function(e) NA)

    # 2) 计算观察者间 ICC
    sub_orig_inter <- original[match(common_ids_inter, original[[id_original]]), feat][[1]]
    sub_r2         <- reader2[match(common_ids_inter, reader2[[id_reader2]]), feat][[1]]
    
    mat_inter <- cbind(as.numeric(sub_orig_inter), as.numeric(sub_r2))
    icc_inter_val <- tryCatch({
      icc_res <- icc(mat_inter, model = "twoway", type = "agreement", unit = "single")
      icc_res$value
    }, error = function(e) NA)

    results <- rbind(results, data.frame(
      Feature = feat,
      ICC_Intra = icc_intra_val,
      ICC_Inter = icc_inter_val,
      stringsAsFactors = FALSE
    ))
  }

  # ----------------- 4. 筛选并导出结果 -----------------
  results$Passed <- (!is.na(results$ICC_Intra) & results$ICC_Intra >= icc_threshold) &
                    (!is.na(results$ICC_Inter) & results$ICC_Inter >= icc_threshold)

  dir.create(dirname(output_detail), recursive = TRUE, showWarnings = FALSE)
  write.csv(results, output_detail, row.names = FALSE)

  selected_features <- results$Feature[results$Passed]
  cat("\n============================================================\n")
  cat("ICC 计算完成！总特征数:", length(feature_cols), "\n")
  cat("双重 ICC >=", icc_threshold, "筛选通过的稳定特征数:", length(selected_features), "\n")
  cat("结果已保存至:\n  - 详情:", output_detail, "\n")

  # 导出筛选后的原数据集
  selected_data <- original[, c(id_original, selected_features)]
  write.csv(selected_data, output_selected, row.names = FALSE)
  cat("  - 筛选后特征表:", output_selected, "\n")
  cat("============================================================\n")
}
