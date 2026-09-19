# ==============================================================================
# 模块二：临床 CMR 参数筛选 - 10折交叉验证 LASSO 惩罚回归 (cmr_parameter_lasso.R)
# ==============================================================================
# 功能:
#   1. 使用 10 折交叉验证惩罚逻辑回归 (LASSO L1 惩罚, alpha = 1) 筛选核心 CMR 参数。
#   2. 以 lambda.1se (最优稳健惩罚力度) 锁定非零系数变量。
#   3. 构建多元逻辑回归模型，计算变量优势比 (OR, 95% CI) 与显著性 p 值。
#   4. 使用 Bootstrap 自助法 (1000 次) 评估开发集表观与外部独立测试集的 AUC (95% CI)。
#
# 结局终点: LGE_status (或 event): 1 表示 LGE 阳性 (Presence), 0 表示 LGE 阴性
# 关键变量: LAs (Left Atrial Reservoir Strain, 左心房储藏期应变)
# ==============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(glmnet)
  library(pROC)
  library(openxlsx)
})

# ----------------- 1. 路径与全局配置 (请根据本地环境修改) -----------------
base_dir <- "./data/CMR_parameter"
train_file <- file.path(base_dir, "train.xlsx")
external_file <- file.path(base_dir, "external.xlsx")
output_dir <- "./results/CMR_parameter_LASSO_results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 20260905
inner_folds <- 10
bootstrap_iterations <- 1000

# 候选临床与 CMR 参数
# 注: LAs 为 Left Atrial Reservoir Strain (左心房储藏期应变)
candidate_vars <- c(
  "sex", "age",
  "LVEDV", "EF", "HR", "LVmass", "MWT",
  "maxLAV", "LAEF", "LAs", "SRs", "SRe", "Sra",
  "LVGRS", "LVGCS", "LVGLS"
)

# ----------------- 2. 数据读取与预处理 -----------------
if (!file.exists(train_file) || !file.exists(external_file)) {
  stop("【错误】未找到输入文件: ", train_file, " 或 ", external_file, 
       "\n请确保将临床数据表格放置在指定路径。")
}

train <- as.data.frame(read_excel(train_file, sheet = 1))
external <- as.data.frame(read_excel(external_file, sheet = 1))

# 获取目标变量列名 (优先识别 LGE_status，同时兼容 event)
get_target_col <- function(df) {
  if ("LGE_status" %in% names(df)) return("LGE_status")
  if ("event" %in% names(df)) return("event")
  stop("【错误】数据表中未找到 LGE_status 或 event 结局标签列！")
}

target_col_train <- get_target_col(train)
target_col_ext   <- get_target_col(external)

y_train <- as.numeric(train[[target_col_train]])
y_external <- as.numeric(external[[target_col_ext]])

# 检查候选变量是否存在
available_vars <- intersect(candidate_vars, intersect(names(train), names(external)))
cat("候选 CMR 参数总数:", length(candidate_vars), " | 实际匹配可用数:", length(available_vars), "\n")
cat("开发集样本数:", nrow(train), " (LGE 阳性数:", sum(y_train == 1), ")\n")
cat("外部测试集样本数:", nrow(external), " (LGE 阳性数:", sum(y_external == 1), ")\n")

x_train <- as.matrix(train[, available_vars])
x_external <- as.matrix(external[, available_vars])
storage.mode(x_train) <- "double"
storage.mode(x_external) <- "double"

# ----------------- 3. 10折交叉验证分层划分与 LASSO 拟合 -----------------
set.seed(random_seed)
foldid <- integer(length(y_train))
for (group in c(0, 1)) {
  index <- which(y_train == group)
  foldid[index] <- sample(rep(seq_len(inner_folds), length.out = length(index)))
}

cat("\n正在运行 10 折交叉验证 LASSO 特征筛选...\n")
cv_fit <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1,                 # LASSO (L1 惩罚)
  foldid = foldid,
  type.measure = "auc",      # 以 AUC 为调优指标
  standardize = TRUE,
  intercept = TRUE
)

lambda_selected <- cv_fit$lambda.1se
coef_matrix <- as.matrix(coef(cv_fit$glmnet.fit, s = lambda_selected))
selected_vars <- setdiff(
  rownames(coef_matrix)[coef_matrix[, 1] != 0],
  "(Intercept)"
)

cat("LASSO (lambda.1se) 筛选保留的核心变量:\n")
print(selected_vars)

lasso_coefficients <- data.frame(
  Variable = rownames(coef_matrix),
  Coefficient = as.numeric(coef_matrix[, 1])
)
lasso_coefficients <- lasso_coefficients[lasso_coefficients$Coefficient != 0, ]

cv_table <- data.frame(
  Lambda = cv_fit$lambda,
  Mean_CV_AUC = cv_fit$cvm,
  SE = cv_fit$cvsd,
  Nonzero_variables = cv_fit$nzero,
  Selected_lambda_1se = cv_fit$lambda == lambda_selected
)

# ----------------- 4. 多元逻辑回归与效应量评估 -----------------
final_formula <- as.formula(paste(target_col_train, "~", paste(selected_vars, collapse = " + ")))
final_logit <- glm(final_formula, data = train, family = binomial())

beta <- coef(final_logit)
se <- sqrt(diag(vcov(final_logit)))
logistic_coefficients <- data.frame(
  Variable = names(beta),
  Beta = as.numeric(beta),
  OR = exp(beta),
  CI_lower = exp(beta - 1.96 * se),
  CI_upper = exp(beta + 1.96 * se),
  P_value = summary(final_logit)$coefficients[, 4]
)

train_probability <- as.numeric(predict(final_logit, newdata = train, type = "response"))
external_probability <- as.numeric(predict(final_logit, newdata = external, type = "response"))

performance_metrics <- function(y, probability, dataset, seed) {
  roc_obj <- roc(y, probability, levels = c(0, 1), direction = "<", quiet = TRUE)
  set.seed(seed)
  auc_ci <- ci.auc(
    roc_obj,
    method = "bootstrap",
    boot.n = bootstrap_iterations,
    stratified = TRUE
  )
  probability <- pmin(pmax(probability, 1e-15), 1 - 1e-15)
  data.frame(
    Dataset = dataset,
    N = length(y),
    LGE_positive = sum(y == 1),
    LGE_rate = mean(y == 1),
    AUC = as.numeric(auc(roc_obj)),
    AUC_CI_lower = as.numeric(auc_ci[1]),
    AUC_CI_upper = as.numeric(auc_ci[3]),
    Brier_score = mean((probability - y)^2),
    Log_loss = -mean(y * log(probability) + (1 - y) * log(1 - probability))
  )
}

performance <- rbind(
  performance_metrics(
    y_train, train_probability,
    "Development cohort", random_seed + 1
  ),
  performance_metrics(
    y_external, external_probability,
    "External validation cohort", random_seed + 2
  )
)

print(performance)

# ----------------- 5. 输出与结果保存 -----------------
model_summary <- data.frame(
  Item = c(
    "Target_endpoint", "Development_N", "Development_LGE_positive",
    "External_N", "External_LGE_positive",
    "Candidate_variables", "Selected_variables",
    "Alpha", "Lambda_1se", "CV_folds", "Random_seed"
  ),
  Value = c(
    "LGE_status", nrow(train), sum(y_train), nrow(external), sum(y_external),
    length(available_vars), paste(selected_vars, collapse = ", "),
    1, lambda_selected, inner_folds, random_seed
  )
)

train_predictions <- data.frame(
  ID = train$ID,
  LGE_status = y_train,
  Probability = train_probability
)

external_predictions <- data.frame(
  ID = external$ID,
  LGE_status = y_external,
  Probability = external_probability
)

write.xlsx(
  list(
    Model_summary = model_summary,
    Performance = performance,
    Candidate_variables = data.frame(Variable = available_vars),
    LASSO_coefficients = lasso_coefficients,
    Logistic_coefficients = logistic_coefficients,
    CV_path = cv_table,
    Development_predictions = train_predictions,
    External_predictions = external_predictions
  ),
  file.path(output_dir, "CMR_LASSO_results.xlsx"),
  overwrite = TRUE
)

saveRDS(
  list(
    lasso_model = cv_fit$glmnet.fit,
    lambda = lambda_selected,
    selected_variables = selected_vars,
    logistic_model = final_logit,
    candidate_variables = available_vars
  ),
  file.path(output_dir, "CMR_LASSO_model.rds")
)

pdf(file.path(output_dir, "CMR_LASSO_CV.pdf"), width = 7, height = 6)
plot(cv_fit)
abline(v = log(lambda_selected), lty = 2, col = "red")
dev.off()

cat("\n============================================================\n")
cat("CMR 参数 LASSO 分析完成！\n")
cat("筛选变量:", paste(selected_vars, collapse = ", "), "\n")
cat("最优 lambda.1se:", lambda_selected, "\n")
cat("结果已保存至目录:", output_dir, "\n")
cat("============================================================\n")
