# ==============================================================================
# 模块二：影像组学全流程 - 重复嵌套交叉验证弹性网络 R-score 构建 (rscore_nested_oof_elasticnet.R)
# ==============================================================================
# 模型: 弹性网络 (Elastic Net) 惩罚逻辑回归
# 核心机制:
#   1. 外层评估: 10 折交叉验证 x 5 次重复 (共 50 个验证折)，生成患者级 Out-of-Fold (OOF) 预测。
#   2. 内层调优: 10 折交叉验证网格搜索最优混合参数 alpha (0.1 ~ 0.9) 与惩罚参数 lambda (lambda.1se)。
#   3. 严控信息泄露 (Leakage-controlled):
#      - 缺失率过滤 (missing cutoff = 0.20) 仅在训练折内统计；
#      - 特征相关性去冗余 (pairwise correlation cutoff = 0.95) 仅在训练折内计算；
#      - Z-score 均值与方差标准化仅基于训练折，再应用到验证折与外部测试集。
#   4. 锁定最终模型: 在全量开发集上拟合并导出 R-score 权重公式与模型包，供独立外部测试集评估。
#
# 结局目标: LGE_status (或 event): 1 表示 LGE presence (阳性), 0 表示 LGE 阴性
# 特征输入: 舒张末期 (ED) 短轴 cine 左心室心肌 ROI 提取并经 ICC >= 0.80 筛选后的稳定组学特征
# ==============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(openxlsx)
  library(caret)
  library(glmnet)
  library(pROC)
  library(dplyr)
})

# ------------------------- 1. 路径与参数配置 (请根据本地环境修改) -------------------------
base_dir <- "./data/Radiomics"

train_file <- file.path(base_dir, "Radiomics_train.xlsx")
external_file <- file.path(base_dir, "Radiomics_external.xlsx")
icc_selected_file <- file.path(base_dir, "ICC", "Radiomics_ICC_selected.csv")

output_dir <- "./results/Rscore_OOF_results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 20260731
outer_folds <- 10
outer_repeats <- 5
inner_folds <- 10
alpha_grid <- seq(0.1, 0.9, by = 0.1) # 弹性网络混合参数搜索网格
missing_threshold <- 0.20             # 缺失率容忍上限
correlation_threshold <- 0.95         # 特征共线性去冗余相关系数截断值
bootstrap_iterations <- 1000

# ------------------------- 2. 辅助工具函数 -------------------------
stop_if_missing <- function(paths) {
  absent <- paths[!file.exists(paths)]
  if (length(absent) > 0) {
    stop("【错误】未找到以下输入文件:\n",
         paste(absent, collapse = "\n"))
  }
}

make_foldid <- function(y, k, seed) {
  counts <- table(y)
  if (length(counts) != 2 || min(counts) < k) {
    stop("分层划分失败: 阳性与阴性样本数必须分别不少于 ", k, " 例。")
  }
  set.seed(seed)
  folds <- caret::createFolds(factor(y, levels = c(0, 1)),
                              k = k, returnTrain = FALSE)
  foldid <- integer(length(y))
  for (i in seq_along(folds)) foldid[folds[[i]]] <- i
  if (any(foldid == 0)) stop("Fold 分配异常。")
  foldid
}

to_numeric_frame <- function(dat, feature_names, dataset_name) {
  out <- lapply(dat[, feature_names, drop = FALSE], function(x) {
    suppressWarnings(as.numeric(x))
  })
  out <- as.data.frame(out, check.names = FALSE)
  names(out) <- feature_names

  completely_missing <- names(out)[vapply(out, function(x) all(is.na(x)), logical(1))]
  if (length(completely_missing) > 0) {
    stop(dataset_name, " 数据集中存在非数值或全缺失特征: ",
         paste(completely_missing, collapse = ", "))
  }
  out
}

# 仅在训练折内执行的预处理器拟合 (严格杜绝测试集信息穿越)
fit_preprocessor <- function(x,
                             missing_cutoff = 0.20,
                             correlation_cutoff = 0.95) {
  p_missing <- colMeans(is.na(x))
  kept <- names(p_missing)[p_missing <= missing_cutoff]
  x_sub <- x[, kept, drop = FALSE]

  center_val <- vapply(x_sub, function(col) mean(col, na.rm = TRUE), numeric(1))
  scale_val  <- vapply(x_sub, function(col) {
    s <- sd(col, na.rm = TRUE)
    if (is.na(s) || s < 1e-12) 0 else s
  }, numeric(1))

  valid_scale <- names(scale_val)[scale_val > 0]
  x_sub <- x_sub[, valid_scale, drop = FALSE]
  center_val <- center_val[valid_scale]
  scale_val  <- scale_val[valid_scale]

  imputed <- as.data.frame(lapply(names(x_sub), function(col_name) {
    col <- x_sub[[col_name]]
    col[is.na(col)] <- center_val[[col_name]]
    col
  }), check.names = FALSE)
  names(imputed) <- names(x_sub)

  x_scaled <- scale(as.matrix(imputed),
                    center = center_val,
                    scale = scale_val)

  cor_mat <- suppressWarnings(cor(x_scaled, use = "pairwise.complete.obs"))
  cor_mat[is.na(cor_mat)] <- 0
  diag(cor_mat) <- 1

  high_cor <- caret::findCorrelation(cor_mat,
                                     cutoff = correlation_cutoff,
                                     names = TRUE,
                                     exact = TRUE)
  final_features <- setdiff(colnames(x_scaled), high_cor)

  list(
    features = final_features,
    center = center_val[final_features],
    scale = scale_val[final_features],
    missing_dropped = setdiff(colnames(x), kept),
    zero_sd_dropped = setdiff(kept, valid_scale),
    correlated_dropped = high_cor
  )
}

transform_preprocessor <- function(x, preprocessor) {
  feats <- preprocessor$features
  mat <- matrix(0, nrow = nrow(x), ncol = length(feats),
                dimnames = list(rownames(x), feats))

  for (f in feats) {
    val <- suppressWarnings(as.numeric(x[[f]]))
    m <- preprocessor$center[[f]]
    s <- preprocessor$scale[[f]]
    val[is.na(val)] <- m
    mat[, f] <- (val - m) / s
  }
  mat
}

# 内层交叉验证调优 Elastic Net (寻找最优 alpha 和 lambda.1se)
tune_elastic_net_inner <- function(x_mat, y_vec, foldid_inner, alphas) {
  best_alpha <- NA_real_
  best_lambda <- NA_real_
  best_cvm <- -Inf
  best_fit <- NULL

  records <- vector("list", length(alphas))

  for (idx in seq_along(alphas)) {
    a <- alphas[idx]
    fit <- cv.glmnet(
      x = x_mat,
      y = y_vec,
      family = "binomial",
      alpha = a,
      foldid = foldid_inner,
      type.measure = "auc",
      standardize = FALSE,
      intercept = TRUE
    )

    lambda_1se <- fit$lambda.1se
    idx_1se <- which(fit$lambda == lambda_1se)[1]
    cv_auc_1se <- fit$cvm[idx_1se]

    records[[idx]] <- data.frame(
      alpha = a,
      lambda_1se = lambda_1se,
      cv_auc_1se = cv_auc_1se,
      nzero_1se = fit$nzero[idx_1se],
      cv_auc_min = max(fit$cvm, na.rm = TRUE),
      lambda_min = fit$lambda.min
    )

    if (cv_auc_1se > best_cvm) {
      best_cvm <- cv_auc_1se
      best_alpha <- a
      best_lambda <- lambda_1se
      best_fit <- fit
    }
  }

  list(
    alpha = best_alpha,
    lambda = best_lambda,
    cv_auc = best_cvm,
    fit = best_fit,
    tuning = do.call(rbind, records)
  )
}

extract_coefficients <- function(fit, lambda, feature_names) {
  cm <- as.matrix(coef(fit, s = lambda))
  intercept <- as.numeric(cm["(Intercept)", 1])
  cm_no_int <- cm[rownames(cm) != "(Intercept)", 1, drop = FALSE]
  beta <- as.numeric(cm_no_int[, 1])
  names(beta) <- rownames(cm_no_int)

  nonzero_beta <- beta[beta != 0]
  list(
    intercept = intercept,
    beta_all = beta,
    nonzero_features = names(nonzero_beta),
    nonzero_beta = nonzero_beta
  )
}

compute_auc_ci <- function(y, score, seed, n_boot = 1000) {
  r <- roc(y, score, levels = c(0, 1), direction = "<", quiet = TRUE)
  set.seed(seed)
  ci_val <- ci.auc(r, method = "bootstrap", boot.n = n_boot, stratified = TRUE)
  c(
    AUC = as.numeric(auc(r)),
    CI_lower = as.numeric(ci_val[1]),
    CI_upper = as.numeric(ci_val[3])
  )
}

draw_roc_panel <- function(y, score, title_text, col_line) {
  r <- roc(y, score, levels = c(0, 1), direction = "<", quiet = TRUE)
  ci_val <- ci.auc(r, method = "bootstrap", boot.n = 1000, stratified = TRUE)
  plot(r, col = col_line, lwd = 2.5,
       main = paste0(title_text, "\nAUC = ",
                     format(round(auc(r), 3), nsmall = 3),
                     " (", format(round(ci_val[1], 3), nsmall = 3),
                     "-", format(round(ci_val[3], 3), nsmall = 3), ")"),
       cex.main = 0.95)
  grid(col = "gray85")
  abline(a = 0, b = 1, lty = 2, col = "gray50")
}

# ------------------------- 3. 数据载入与校验 -------------------------
cat("============================================================\n")
cat("正在启动基于非增强短轴 Cine 的影像组学 R-score 构建流程...\n")
cat("模型: 重复嵌套交叉验证弹性网络 (Repeated Nested CV Elastic Net)\n")
cat("============================================================\n")

stop_if_missing(c(train_file, external_file, icc_selected_file))

train <- as.data.frame(readxl::read_excel(train_file, sheet = 1))
external <- as.data.frame(readxl::read_excel(external_file, sheet = 1))
icc_selected <- read.csv(icc_selected_file, stringsAsFactors = FALSE, check.names = FALSE)

feature_names <- icc_selected$Feature
if (is.null(feature_names) || length(feature_names) == 0) {
  stop("未能从 ICC 筛选结果表中读取有效特征列表。")
}

# 统一结局标签列名 (优先 LGE_status, 兼容 event)
get_target_col <- function(df) {
  if ("LGE_status" %in% names(df)) return("LGE_status")
  if ("event" %in% names(df)) return("event")
  stop("【错误】数据集中未找到 LGE_status 或 event 结局标签列！")
}

target_train <- get_target_col(train)
target_ext   <- get_target_col(external)

y_train <- as.numeric(train[[target_train]])
y_external <- as.numeric(external[[target_ext]])

cat("经 ICC 稳定性筛选保留的特征数:", length(feature_names), "\n")
cat("开发集样本数:", nrow(train), " (LGE 阳性数:", sum(y_train == 1), ")\n")
cat("外部测试集样本数:", nrow(external), " (LGE 阳性数:", sum(y_external == 1), ")\n")

X_train_raw <- to_numeric_frame(train, feature_names, "Development cohort")
X_external_raw <- to_numeric_frame(external, feature_names, "External cohort")

# ------------------------- 4. 重复嵌套交叉验证 (Nested CV) -------------------------
n_train <- nrow(train)
oof_matrix_raw <- matrix(NA_real_, nrow = n_train, ncol = outer_repeats)

outer_records <- vector("list", outer_repeats * outer_folds)
outer_tuning_records <- vector("list", outer_repeats * outer_folds)
selection_counts <- setNames(integer(length(feature_names)), feature_names)

cat("\n开始执行 5 次重复 x 10 折嵌套交叉验证 (共 50 折外层迭代)...\n")

fold_counter <- 0
for (rep_i in seq_len(outer_repeats)) {
  rep_seed <- random_seed + rep_i * 1000
  outer_foldid <- make_foldid(y_train, k = outer_folds, seed = rep_seed)

  for (fold_i in seq_len(outer_folds)) {
    fold_counter <- fold_counter + 1

    train_idx <- which(outer_foldid != fold_i)
    val_idx   <- which(outer_foldid == fold_i)

    # 1. 仅在训练折内进行预处理器拟合
    prep_fold <- fit_preprocessor(
      X_train_raw[train_idx, , drop = FALSE],
      missing_cutoff = missing_threshold,
      correlation_cutoff = correlation_threshold
    )

    X_tr_mat  <- transform_preprocessor(X_train_raw[train_idx, , drop = FALSE], prep_fold)
    X_val_mat <- transform_preprocessor(X_train_raw[val_idx, , drop = FALSE], prep_fold)

    # 2. 内层交叉验证分层划分
    inner_seed <- rep_seed + fold_i * 100
    inner_foldid <- make_foldid(y_train[train_idx], k = inner_folds, seed = inner_seed)

    # 3. 调优弹性网络超参数 (alpha, lambda)
    tuned <- tune_elastic_net_inner(X_tr_mat, y_train[train_idx], inner_foldid, alpha_grid)

    coef_info <- extract_coefficients(tuned$fit, tuned$lambda, prep_fold$features)
    selection_counts[coef_info$nonzero_features] <-
      selection_counts[coef_info$nonzero_features] + 1

    # 4. 在外层验证折上生成无偏预测
    val_pred_raw <- as.numeric(predict(
      tuned$fit,
      newx = X_val_mat,
      s = tuned$lambda,
      type = "link"
    ))

    oof_matrix_raw[val_idx, rep_i] <- val_pred_raw

    outer_records[[fold_counter]] <- data.frame(
      Repeat = rep_i,
      Fold = fold_i,
      Selected_alpha = tuned$alpha,
      Selected_lambda = tuned$lambda,
      Inner_CV_AUC = tuned$cv_auc,
      Nonzero_count = length(coef_info$nonzero_features)
    )

    t_df <- tuned$tuning
    t_df$Repeat <- rep_i
    t_df$Fold <- fold_i
    outer_tuning_records[[fold_counter]] <- t_df
  }
  cat(sprintf("  -> 成功完成第 %d / %d 次外层重复评估\n", rep_i, outer_repeats))
}

# 患者级 OOF Rscore 平均值
oof_rscore_raw <- rowMeans(oof_matrix_raw, na.rm = TRUE)
oof_mean <- mean(oof_rscore_raw, na.rm = TRUE)
oof_sd   <- sd(oof_rscore_raw, na.rm = TRUE)
oof_rscore_std <- (oof_rscore_raw - oof_mean) / oof_sd

# ------------------------- 5. 全量开发集上拟合最终锁定模型 -------------------------
cat("\n正在开发集全量样本上拟合并锁定最终 R-score 弹性网络模型...\n")

final_prep <- fit_preprocessor(
  X_train_raw,
  missing_cutoff = missing_threshold,
  correlation_cutoff = correlation_threshold
)

X_train_final <- transform_preprocessor(X_train_raw, final_prep)
X_ext_final   <- transform_preprocessor(X_external_raw, final_prep)

final_inner_foldid <- make_foldid(y_train, k = inner_folds, seed = random_seed + 9999)
final_tuning <- tune_elastic_net_inner(X_train_final, y_train, final_inner_foldid, alpha_grid)

final_fit <- final_tuning$fit
final_coef <- extract_coefficients(final_fit, final_tuning$lambda, final_prep$features)

# 外部独立验证集预测
external_rscore_raw <- as.numeric(predict(
  final_fit,
  newx = X_ext_final,
  s = final_tuning$lambda,
  type = "link"
))
external_rscore_std <- (external_rscore_raw - oof_mean) / oof_sd

# ------------------------- 6. 提取数学公式与系数 -------------------------
nonzero_names <- final_coef$nonzero_features
beta_std <- final_coef$nonzero_beta

center_sub <- final_prep$center[nonzero_names]
scale_sub  <- final_prep$scale[nonzero_names]

beta_orig <- beta_std / scale_sub
intercept_orig <- final_coef$intercept - sum(beta_std * center_sub / scale_sub)

coef_original <- data.frame(
  Feature = c("(Intercept)", nonzero_names),
  Coefficient_original_scale = c(intercept_orig, beta_orig),
  Center = c(NA_real_, center_sub),
  Scale = c(NA_real_, scale_sub),
  stringsAsFactors = FALSE
)

coef_standardized <- data.frame(
  Feature = c("(Intercept)", nonzero_names),
  Coefficient_standardized_scale = c(final_coef$intercept, beta_std),
  stringsAsFactors = FALSE
)

formula_terms <- vapply(seq_along(nonzero_names), function(i) {
  sprintf("(%16.14f * %s)", beta_orig[i], nonzero_names[i])
}, character(1))
formula_text <- paste0(
  "Rscore = ", sprintf("%.10f", intercept_orig), "\n  + ",
  paste(formula_terms, collapse = "\n  + ")
)

# ------------------------- 7. 性能统计与结果保存 -------------------------
perf_dev_oof <- compute_auc_ci(y_train, oof_rscore_raw, random_seed + 11, bootstrap_iterations)
perf_ext     <- compute_auc_ci(y_external, external_rscore_raw, random_seed + 12, bootstrap_iterations)

performance <- data.frame(
  Dataset = c("Development repeated nested OOF", "External locked-model validation"),
  N = c(length(y_train), length(y_external)),
  LGE_positive = c(sum(y_train == 1), sum(y_external == 1)),
  LGE_rate = c(mean(y_train == 1), mean(y_external == 1)),
  AUC = c(perf_dev_oof["AUC"], perf_ext["AUC"]),
  AUC_CI_lower = c(perf_dev_oof["CI_lower"], perf_ext["CI_lower"]),
  AUC_CI_upper = c(perf_dev_oof["CI_upper"], perf_ext["CI_upper"])
)

print(performance)

# 保存预测表格
oof_predictions <- data.frame(
  ID = train$ID,
  LGE_status = y_train,
  Rscore = oof_rscore_raw,
  Rscore_standardized = oof_rscore_std
)

external_predictions <- data.frame(
  ID = external$ID,
  LGE_status = y_external,
  Rscore = external_rscore_raw,
  Rscore_standardized = external_rscore_std
)

write.csv(oof_predictions, file.path(output_dir, "Rscore_OOF_predictions.csv"), row.names = FALSE)
write.csv(external_predictions, file.path(output_dir, "Rscore_external_predictions.csv"), row.names = FALSE)
writeLines(formula_text, con = file.path(output_dir, "Rscore_formula.txt"), useBytes = TRUE)

# 导出模型 RDS 对象
model_bundle <- list(
  model = final_fit,
  preprocessor = final_prep,
  alpha = final_tuning$alpha,
  lambda = final_tuning$lambda,
  event_levels = c(0, 1),
  input_icc_features = feature_names,
  oof_standardization_mean = oof_mean,
  oof_standardization_sd = oof_sd,
  random_seed = random_seed,
  formula_original_scale = formula_text
)
saveRDS(model_bundle, file.path(output_dir, "Rscore_locked_model.rds"))

# 绘制 ROC 曲线
pdf(file.path(output_dir, "Rscore_ROC_curves.pdf"), width = 10, height = 5)
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2) + 0.1)
draw_roc_panel(y_train, oof_rscore_raw, "Development: repeated nested OOF", "#0072B2")
draw_roc_panel(y_external, external_rscore_raw, "External validation: locked model", "#D55E00")
dev.off()

cat("\n============================================================\n")
cat("弹性网络 R-score 构建全部完成！\n")
cat("最终最优 alpha:", final_tuning$alpha, " | lambda.1se:", final_tuning$lambda, "\n")
cat("入选非零组学特征数:", length(nonzero_names), "\n")
cat(sprintf("开发集 OOF AUC: %.3f (95%% CI: %.3f - %.3f)\n", 
    perf_dev_oof["AUC"], perf_dev_oof["CI_lower"], perf_dev_oof["CI_upper"]))
cat(sprintf("外部测试集 AUC: %.3f (95%% CI: %.3f - %.3f)\n", 
    perf_ext["AUC"], perf_ext["CI_lower"], perf_ext["CI_upper"]))
cat("结果保存至目录:", output_dir, "\n")
cat("============================================================\n")
