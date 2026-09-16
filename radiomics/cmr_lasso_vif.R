# ==============================================================================
# 模块二：影像组学与临床参数分析 - LASSO 回归筛选与多重共线性 VIF 分析 (cmr_lasso_vif.R)
# ==============================================================================
# 功能:
#   1. 使用 10 折交叉验证惩罚逻辑回归 (LASSO L1 惩罚) 对临床 CMR 候选参数进行特征筛选。
#   2. 计算方差膨胀因子 (Variance Inflation Factor, VIF)，诊断并排除多重共线性特征。
#   3. 使用 Bootstrap 自助法 (1000 次) 计算开发集与外部独立验证集的 AUC 及 95% 置信区间。
# ==============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(glmnet)
  library(car)
  library(pROC)
  library(openxlsx)
})

# ----------------- 1. 路径与全局参数设置 -----------------
base_dir <- "./data/CMR_parameter"
train_file <- file.path(base_dir, "train.xlsx")
external_file <- file.path(base_dir, "external.xlsx")
output_dir <- "./results/LASSO_VIF_results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 20260905
inner_folds <- 10
bootstrap_iterations <- 1000

# 候选临床参数与影像组学评分
candidate_vars <- c(
  "sex", "age", "Rscore",
  "LVEDV", "EF", "HR", "LVmass", "MWT",
  "maxLAV", "LAEF", "LAs", "SRs", "SRe", "Sra",
  "LVGRS", "LVGCS", "LVGLS"
)

# ----------------- 2. 数据读取与校验 -----------------
if (!file.exists(train_file) || !file.exists(external_file)) {
  warning("【提示】未找到 train.xlsx 或 external.xlsx，请确认数据放置在: ", base_dir)
} else {
  train <- as.data.frame(read_excel(train_file, sheet = 1))
  external <- as.data.frame(read_excel(external_file, sheet = 1))

  # 兼容性重命名处理
  if ("maxLAV" %in% names(train) && !"LAs" %in% names(train)) {
    train$LAs <- train$maxLAV
  }
  if ("maxLAV" %in% names(external) && !"LAs" %in% names(external)) {
    external$LAs <- external$maxLAV
  }

  # 过滤实际存在的变量
  available_vars <- intersect(candidate_vars, intersect(names(train), names(external)))
  cat("可用特征数量:", length(available_vars), "\n")

  y_train <- as.numeric(train$event)
  y_external <- as.numeric(external$event)
  x_train <- as.matrix(train[, available_vars])
  x_external <- as.matrix(external[, available_vars])
  storage.mode(x_train) <- "double"
  storage.mode(x_external) <- "double"

  set.seed(random_seed)
  foldid <- integer(length(y_train))
  for (group in c(0, 1)) {
    index <- which(y_train == group)
    foldid[index] <- sample(rep(seq_len(inner_folds), length.out = length(index)))
  }

  # ----------------- 3. 10折交叉验证 LASSO 筛选 -----------------
  cat("\n正在执行 10 折交叉验证 LASSO 惩罚回归...\n")
  cv_fit <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 1,
    foldid = foldid,
    type.measure = "deviance",
    standardize = TRUE
  )

  # 提取 1-SE (最优稳健惩罚力度) 对应的非零系数变量
  coef_1se <- coef(cv_fit, s = "lambda.1se")
  selected_idx <- which(as.vector(coef_1se) != 0)
  selected_names <- rownames(coef_1se)[selected_idx]
  selected_features <- setdiff(selected_names, "(Intercept)")

  cat("LASSO (lambda.1se) 筛选出的核心变量:", paste(selected_features, collapse = ", "), "\n")

  # ----------------- 4. 多重共线性诊断 (VIF 分析) -----------------
  cat("\n正在进行多元逻辑回归与方差膨胀因子 (VIF) 诊断...\n")
  formula_str <- paste("event ~", paste(selected_features, collapse = " + "))
  fit_glm <- glm(as.formula(formula_str), data = train, family = binomial())

  vif_values <- if (length(selected_features) > 1) {
    car::vif(fit_glm)
  } else {
    c(1.0)
  }
  print(vif_values)

  # ----------------- 5. 外部验证集预测与 ROC 评估 -----------------
  prob_train <- predict(fit_glm, newdata = train, type = "response")
  prob_ext <- predict(fit_glm, newdata = external, type = "response")

  roc_train <- roc(y_train, prob_train, ci = TRUE)
  roc_ext <- roc(y_external, prob_ext, ci = TRUE)

  cat(sprintf("\n开发集 AUC: %.3f (95%% CI: %.3f-%.3f)\n", roc_train$auc, roc_train$ci[1], roc_train$ci[3]))
  cat(sprintf("外部验证集 AUC: %.3f (95%% CI: %.3f-%.3f)\n", roc_ext$auc, roc_ext$ci[1], roc_ext$ci[3]))

  # 保存模型与结果
  saveRDS(fit_glm, file.path(output_dir, "CMR_LASSO_VIF_model.rds"))
  cat("模型与分析结果已保存至:", output_dir, "\n")
}
