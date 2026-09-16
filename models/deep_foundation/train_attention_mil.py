#!/usr/bin/env python3
"""Fine-tune the pretrained cardiac mViT with patient-level attention MIL."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from tqdm import tqdm


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "epoch=599-step=124200.ckpt"
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "patient_manifest.csv"
# 默认切片质控文件路径 (记录各患者短轴切片 NIfTI 路径)
DEFAULT_SLICE_QC = Path("./data/processed_sax_cine/slice_qc.csv")
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/foundation_attention_mil_full_transfer_v1"
DEFAULT_RSCORE_DIR = EXPERIMENT_ROOT / "results/rscore_oof"
CMR_COLUMNS = ["MWT", "LAs", "LVGRS", "LVGLS"]
ARMS = {
    "Foundation": [],
    "Foundation+CMR": CMR_COLUMNS,
    "Foundation+Rscore+CMR": ["Rscore", *CMR_COLUMNS],
}


def parse_args():
    """解析并返回命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slice-qc", type=Path, default=DEFAULT_SLICE_QC)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rscore-dir", type=Path, default=DEFAULT_RSCORE_DIR)
    parser.add_argument(
        "--rscore-mode", choices=["rebuilt", "provided"], default="rebuilt",
        help="provided: use Rscore from manifest directly; rebuilt: load from rscore-dir"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=["Foundation"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-slices", type=int, default=8)
    parser.add_argument(
        "--temporal-mode",
        choices=["full", "ed_repeat"],
        default="full",
        help="Use the full cine or repeat the first/ED phase to isolate temporal value.",
    )
    parser.add_argument("--encoder-microbatch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-folds", nargs="+", type=int)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-full-transfer", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_cine(cine):
    """对心脏磁共振电影(cine)数据进行归一化处理。"""
    finite = cine[np.isfinite(cine)]
    if not finite.size:
        raise ValueError("cine contains no finite voxels")
    lower, upper = np.percentile(finite, [1, 99])
    cine = np.nan_to_num(cine, nan=lower, posinf=upper, neginf=lower)
    cine = np.clip(cine, lower, upper)
    cine = (cine - lower) / max(float(upper - lower), 1e-6)
    return torch.from_numpy(cine.astype(np.float32, copy=False))


def prepare_cine(path, augment=False, temporal_mode="full"):
    """加载、预处理(如插值、裁剪等)并可选地增强心脏磁共振数据。"""
    cine = np.asarray(nib.load(path).dataobj, dtype=np.float32)#加载cine数据
    if cine.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {cine.shape}: {path}")
    cine = normalize_cine(cine)
    if temporal_mode == "full" and augment and cine.shape[0] > 16:
        phase = int(torch.randint(0, cine.shape[0], ()).item())#随机选择相位
        cine = cine.roll(phase, dims=0)
    if temporal_mode == "full":
        frame_indices = torch.linspace(0, cine.shape[0] - 1, 16).round().long()#等间距选择16个相位
        cine = cine[frame_indices]
    elif temporal_mode == "ed_repeat":
        cine = cine[:1].repeat(16, 1, 1)
    else:
        raise ValueError(f"unsupported temporal mode: {temporal_mode}")
    height, width = cine.shape[-2:]
    resize_short = 244
    scale = resize_short / min(height, width)
    resized_height = max(224, round(height * scale))
    resized_width = max(224, round(width * scale))#缩放cine到244x244
    cine = F.interpolate(
        cine[:, None],
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    if augment:
        top = int(torch.randint(0, resized_height - 224 + 1, ()).item())
        left = int(torch.randint(0, resized_width - 224 + 1, ()).item())
    else:
        top = (resized_height - 224) // 2
        left = (resized_width - 224) // 2
    cine = cine[:, top : top + 224, left : left + 224]
    if augment and torch.rand(()) < 0.5:
        cine = cine.flip(-1)
    if augment:
        cine = (cine * (0.9 + 0.2 * torch.rand(()))).clamp(0.0, 1.0)
    return cine[None].repeat(3, 1, 1, 1)


def choose_slice_paths(paths, max_slices):
    paths = list(paths)
    if len(paths) <= max_slices:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_slices).round().astype(int)
    return [paths[index] for index in np.unique(indices)]


def load_encoder(checkpoint_path):
    from model_factory import Cardiac_MRI_Encoder, load_contrastive_pretrained_weights

    encoder = Cardiac_MRI_Encoder(
        "mvit", 50, 512, 1, None, 16, "transfer", False
    )
    return load_contrastive_pretrained_weights(encoder, str(checkpoint_path))#加载对比学习的预训练权重


class GatedAttention(nn.Module):
    """基于门控机制的注意力模块(Gated Attention)，用于多实例学习(MIL)。"""
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.value = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_a = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Sigmoid())
        self.attention_c = nn.Linear(128, 1)

    def forward(self, embeddings):
        values = self.value(embeddings)
        scores = self.attention_c(
            self.attention_a(values) * self.attention_b(values)
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(weights[:, None] * values, dim=0), weights


class FoundationAttentionMIL(nn.Module):
    """结合基础预训练编码器和门控注意力机制的多实例学习网络。"""
    def __init__(self, encoder, tabular_dim, dropout, encoder_microbatch):
        super().__init__()
        self.encoder = encoder
        self.encoder_microbatch = encoder_microbatch
        self.aggregator = GatedAttention(dropout=dropout)
        input_dim = 256 + tabular_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, videos):
        outputs = []
        for chunk in videos.split(self.encoder_microbatch):
            if self.training and any(p.requires_grad for p in self.encoder.parameters()):
                outputs.append(
                    gradient_checkpoint(self.encoder, chunk, use_reentrant=False)#梯度检查点
                )
            else:
                outputs.append(self.encoder(chunk))
        return torch.cat(outputs, dim=0)

    def forward(self, videos, tabular, return_features=False):
        embeddings = self.encode(videos)
        image_features, attention = self.aggregator(embeddings)#注意力加权
        pooled = image_features
        if tabular.numel():
            pooled = torch.cat([pooled, tabular], dim=0)
        output = (self.classifier(pooled).squeeze(0), attention)
        if return_features:
            return (*output, image_features)
        return output


def set_encoder_trainable(model, trainable):
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable
    model.encoder.train(trainable and model.training)#是否微调encoder权重


def build_dataset(manifest_path, slice_qc_path, rscore_dir, rscore_mode="rebuilt"):
    """整合病人元数据、切片质控路径和各种临床评分，构建训练使用的完整数据集。
    rscore_mode='provided': 直接使用 manifest 中的 Rscore 列（已是 repeated OOF 平均值）。
    rscore_mode='rebuilt': 从 rscore_dir 加载 OOF 和 locked Rscore。
    """
    manifest = pd.read_csv(manifest_path)
    slice_qc = pd.read_csv(slice_qc_path).sort_values(
        ["cohort", "ID", "slice_coordinate", "slice_index"]
    )
    paths = (
        slice_qc.groupby(["cohort", "ID"])["output_path"]
        .apply(list)
        .rename("slice_paths")
        .reset_index()
    )
    table = manifest.merge(paths, on=["cohort", "ID"], how="inner", validate="one_to_one")
    if rscore_mode == "provided":
        # Rscore 列已在 manifest 中，直接用作 oof 和 locked 值
        if "Rscore" not in table.columns:
            raise ValueError("manifest 中缺少 Rscore 列，无法使用 provided 模式")
        table["Rscore_oof"] = table["Rscore"]
        table["Rscore_locked"] = table["Rscore"]
        if table["Rscore_oof"].isna().any():
            raise ValueError("manifest Rscore 列存在缺失值")
    else:
        development_oof = pd.read_csv(rscore_dir / "development_rscore_oof.csv")
        external = pd.read_csv(rscore_dir / "external_rscore.csv")
        model_bundle = joblib.load(rscore_dir / "final_rscore_model.joblib")
        radiomics = pd.read_excel(PROJECT_ROOT / "Radiomics/Radiomics_train.xlsx")
        locked = model_bundle["model"].decision_function(radiomics[model_bundle["features"]])
        locked_map = pd.Series(locked, index=radiomics["ID"].astype(int))
        oof_map = development_oof.set_index("ID")["Rscore_OOF"]
        external_map = external.set_index("ID")["Rscore"]
        table["Rscore_oof"] = np.nan
        table["Rscore_locked"] = np.nan
        development_rows = table["cohort"] == "development"
        external_rows = table["cohort"] == "external_test"
        table.loc[development_rows, "Rscore_oof"] = table.loc[
            development_rows, "ID"
        ].map(oof_map)
        table.loc[development_rows, "Rscore_locked"] = table.loc[
            development_rows, "ID"
        ].map(locked_map)
        table.loc[external_rows, "Rscore_locked"] = table.loc[
            external_rows, "ID"
        ].map(external_map)
    return table  # 包含病人的元数据以及对应切片的路径


def tabular_statistics(table, indices, columns, rscore_column):
    if not columns:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    resolved = [rscore_column if column == "Rscore" else column for column in columns]
    values = table.iloc[indices][resolved].to_numpy(dtype=np.float32)
    return values.mean(axis=0), np.clip(values.std(axis=0), 1e-6, None)#计算表格特征的均值和标准差


def patient_input(
    table,
    index,
    columns,
    mean,
    scale,
    max_slices,
    temporal_mode,
    augment,
    device,
    rscore_column,
):
    """从给定的数据表中提取指定病人的影像、临床表格特征及标签，并转移到指定设备。"""
    row = table.iloc[index]
    paths = choose_slice_paths(row["slice_paths"], max_slices)
    videos = torch.stack(
        [
            prepare_cine(
                path,
                augment=augment,
                temporal_mode=temporal_mode,
            )
            for path in paths
        ]
    ).to(device)
    resolved = [rscore_column if column == "Rscore" else column for column in columns]
    if resolved:
        values = row[resolved].to_numpy(dtype=np.float32)
        tabular = torch.from_numpy((values - mean) / scale).to(device)
    else:
        tabular = torch.empty(0, device=device)
    label = torch.tensor(float(row["event"]), device=device)
    return videos, tabular, label#返回处理好的影像、表格特征和标签


def optimizer_for(model, args):
    encoder_parameters = list(model.encoder.parameters())#获取encoder的参数
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]#获取head的参数
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    history,
    best_auc=None,
    best_epoch=None,
    best_state=None,
    stale=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "stale": stale,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "last_completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_auc": best_auc,
                "stale_epochs": stale,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_training_checkpoint(path, model, optimizer, scaler, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if device.type == "cuda" and payload["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    return payload


@torch.no_grad()
def predict(model, table, indices, columns, mean, scale, args, device, rscore_column, description):
    model.eval()
    set_encoder_trainable(model, False)
    probabilities, attentions, features = [], [], []
    for index in tqdm(indices, desc=description, leave=False):
        videos, tabular, _ = patient_input(
            table,
            index,
            columns,
            mean,
            scale,
            args.max_slices,
            args.temporal_mode,
            False,
            device,
            rscore_column,
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logit, attention, image_features = model(
                videos, tabular, return_features=True
            )
        probabilities.append(float(torch.sigmoid(logit).cpu()))
        attentions.append(attention.float().cpu().numpy().tolist())
        features.append(image_features.float().cpu().numpy())
    return np.asarray(probabilities), attentions, np.stack(features)


def train_epoch(model, table, indices, columns, mean, scale, optimizer, scaler, args, device, epoch, rscore_column):
    """在给定的训练集数据上训练模型一个epoch(全量遍历一次)。"""
    model.train()
    encoder_trainable = epoch > args.warmup_epochs #根据parser定义判断是否解冻编码器权重
    set_encoder_trainable(model, encoder_trainable)
    order = np.random.permutation(indices)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    progress = tqdm(order, desc=f"epoch {epoch}", leave=False)
    for step, index in enumerate(progress, start=1):
        videos, tabular, label = patient_input(
            table,
            index,
            columns,
            mean,
            scale,
            args.max_slices,
            args.temporal_mode,
            True,
            device,
            rscore_column,
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logit, _ = model(videos, tabular)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        if step % args.gradient_accumulation == 0 or step == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
    return float(np.mean(losses))


def new_model(args, tabular_dim, device):
    model = FoundationAttentionMIL(
        load_encoder(args.checkpoint), tabular_dim, args.dropout, args.encoder_microbatch
    ).to(device)
    return model


def fit_with_early_stopping(
    table,
    train_indices,
    columns,
    args,
    device,
    seed,
    rscore_column,
    checkpoint_path,
):
    """结合早停机制(Early Stopping)在内部拆分的训练集和验证集上训练网络，防止过拟合。"""
    labels = table.iloc[train_indices]["event"].to_numpy()
    inner_train, inner_validation = train_test_split(
        train_indices,
        test_size=args.validation_fraction,
        stratify=labels,
        random_state=seed,
    )
    mean, scale = tabular_statistics(
        table, inner_train, columns, rscore_column
    )
    model = new_model(args, len(columns), device)
    optimizer = optimizer_for(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    start_epoch = 1
    if checkpoint_path.exists():
        payload = load_training_checkpoint(
            checkpoint_path, model, optimizer, scaler, device
        )
        start_epoch = int(payload["epoch"]) + 1
        history = payload["history"]
        best_auc = float(payload["best_auc"])
        best_epoch = int(payload["best_epoch"])
        best_state = payload["best_state"]
        stale = int(payload["stale"])
        print(
            f"resuming {checkpoint_path.name} after epoch {start_epoch - 1}; "
            f"best epoch={best_epoch}, AUC={best_auc:.4f}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        loss = train_epoch(
            model,
            table,
            inner_train,
            columns,
            mean,
            scale,
            optimizer,
            scaler,
            args,
            device,
            epoch,
            rscore_column,
        )
        probabilities, _, _ = predict(
            model,
            table,
            inner_validation,
            columns,
            mean,
            scale,
            args,
            device,
            rscore_column,
            "inner validation",
        )
        auc = roc_auc_score(table.iloc[inner_validation]["event"], probabilities)
        history.append({"epoch": epoch, "train_loss": loss, "validation_auc": auc})
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch}: train loss={loss:.4f}, inner validation AUC={auc:.4f}, "
            f"best epoch={best_epoch}, stale={stale}/{args.patience}",
            flush=True,
        )
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            history,
            best_auc,
            best_epoch,
            best_state,
            stale,
        )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid early-stopping model state was produced")
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, history


def fit_fixed_epochs(
    table,
    indices,
    columns,
    args,
    device,
    epochs,
    rscore_column,
    checkpoint_path,
):
    mean, scale = tabular_statistics(table, indices, columns, rscore_column)
    model = new_model(args, len(columns), device)
    optimizer = optimizer_for(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    start_epoch = 1
    if checkpoint_path.exists():
        payload = load_training_checkpoint(
            checkpoint_path, model, optimizer, scaler, device
        )
        start_epoch = int(payload["epoch"]) + 1
        history = payload["history"]
        print(
            f"resuming final model after epoch {start_epoch - 1}/{epochs}",
            flush=True,
        )
    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(
            model,
            table,
            indices,
            columns,
            mean,
            scale,
            optimizer,
            scaler,
            args,
            device,
            epoch,
            rscore_column,
        )
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"final epoch {epoch}/{epochs}: train loss={loss:.4f}", flush=True)
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            history,
        )
    return model, mean, scale, history


def select_threshold(labels, probabilities, target_sensitivity=0.95):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    eligible = np.flatnonzero(true_positive_rate >= target_sensitivity)
    specificity = 1.0 - false_positive_rate
    best = eligible[np.argmax(specificity[eligible])]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    divide = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {
        "n": len(labels),
        "auc": roc_auc_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "threshold": threshold,
        "sensitivity": divide(true_positive, true_positive + false_negative),
        "specificity": divide(true_negative, true_negative + false_positive),
        "ppv": divide(true_positive, true_positive + false_positive),
        "npv": divide(true_negative, true_negative + false_negative),
    }


def tune_fusion(features, labels, seed):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        model,
        {
            "model__C": loguniform(1e-3, 100),
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        n_iter=30,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
        random_state=seed,
        refit=True,
    )
    search.fit(features, labels)
    parameters = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, parameters


def fusion_features(table, indices, image_features, columns, rscore_column):
    resolved = [rscore_column if column == "Rscore" else column for column in columns]
    clinical = table.iloc[indices][resolved].to_numpy(dtype=np.float32)
    return np.concatenate([image_features, clinical], axis=1)


def run_arm(arm, table, args, device):
    """运行并评估指定的实验分支(arm)，执行外层K折交叉验证，保存相应的预测概率和历史记录。"""
    columns = ARMS[arm]
    development_indices = np.flatnonzero(table["cohort"].to_numpy() == "development")
    external_indices = np.flatnonzero(table["cohort"].to_numpy() == "external_test")
    labels = table.iloc[development_indices]["event"].to_numpy()
    outer_cv = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    oof_probability = np.full(len(development_indices), np.nan)
    fusion_columns = {
        "Foundation+CMR": CMR_COLUMNS,
        "Foundation+Rscore+CMR": ["Rscore", *CMR_COLUMNS],
    } if arm == "Foundation" else {}
    fusion_oof = {
        name: np.full(len(development_indices), np.nan)
        for name in fusion_columns
    }
    fold_rows, histories = [], {}
    arm_dir = args.output_dir / arm.replace("+", "_plus_")
    progress_dir = arm_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_local, validation_local) in enumerate(
        outer_cv.split(development_indices, labels), start=1
    ):
        if args.only_folds and fold not in args.only_folds:
            continue
        print(f"{arm}: outer fold {fold}/{args.folds}", flush=True)
        train_indices = development_indices[train_local]
        validation_indices = development_indices[validation_local]
        completed_path = progress_dir / f"fold_{fold}_complete.joblib"
        if completed_path.exists():
            completed = joblib.load(completed_path)
            if not np.array_equal(completed["validation_local"], validation_local):
                raise RuntimeError(f"Fold definition changed for {completed_path}")
            oof_probability[validation_local] = completed["probabilities"]
            for fusion_name, values in completed["fusion_probabilities"].items():
                fusion_oof[fusion_name][validation_local] = values
            fold_rows.append(completed["fold_row"])
            histories[str(fold)] = completed["history"]
            for fusion_name, parameters in completed["fusion_parameters"].items():
                histories.setdefault("fusion_parameters", {}).setdefault(
                    fusion_name, {}
                )[str(fold)] = parameters
            print(f"loaded completed fold {fold} from disk", flush=True)
            continue
        epoch_checkpoint = progress_dir / f"fold_{fold}_training.pt"
        model, mean, scale, best_epoch, history = fit_with_early_stopping(
            table,
            train_indices,
            columns,
            args,
            device,
            args.seed + fold,
            "Rscore_oof",
            epoch_checkpoint,
        )
        _, _, train_features = predict(
            model,
            table,
            train_indices,
            columns,
            mean,
            scale,
            args,
            device,
            "Rscore_oof",
            f"outer fold {fold} training representations",
        )
        probabilities, _, validation_features = predict(
            model,
            table,
            validation_indices,
            columns,
            mean,
            scale,
            args,
            device,
            "Rscore_oof",
            f"outer fold {fold}",
        )
        oof_probability[validation_local] = probabilities
        fold_fusion_probabilities = {}
        fold_fusion_parameters = {}
        for fusion_name, extra_columns in fusion_columns.items():
            fusion_model, best_parameters = tune_fusion(
                fusion_features(
                    table,
                    train_indices,
                    train_features,
                    extra_columns,
                    "Rscore_oof",
                ),
                table.iloc[train_indices]["event"].to_numpy(),
                args.seed + fold,
            )
            fusion_probability = fusion_model.predict_proba(
                fusion_features(
                    table,
                    validation_indices,
                    validation_features,
                    extra_columns,
                    "Rscore_oof",
                )
            )[:, 1]
            fusion_oof[fusion_name][validation_local] = fusion_probability
            fold_fusion_probabilities[fusion_name] = fusion_probability
            fold_fusion_parameters[fusion_name] = best_parameters
            histories.setdefault("fusion_parameters", {}).setdefault(
                fusion_name, {}
            )[str(fold)] = best_parameters
        fold_row = {
            "fold": fold,
            "training_n": len(train_indices),
            "validation_n": len(validation_indices),
            "best_epoch": best_epoch,
            "validation_auc": roc_auc_score(labels[validation_local], probabilities),
        }
        fold_rows.append(fold_row)
        histories[str(fold)] = history
        joblib.dump(
            {
                "validation_local": validation_local,
                "probabilities": probabilities,
                "fusion_probabilities": fold_fusion_probabilities,
                "fusion_parameters": fold_fusion_parameters,
                "fold_row": fold_row,
                "history": history,
            },
            completed_path,
        )
        epoch_checkpoint.unlink(missing_ok=True)
        epoch_checkpoint.with_suffix(".json").unlink(missing_ok=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.skip_final:
        return []
    if len(fold_rows) != args.folds or np.isnan(oof_probability).any():
        raise RuntimeError(
            "All outer folds must be available before final locked-model training"
        )
    locked_epochs = max(args.warmup_epochs + 1, int(np.median([x["best_epoch"] for x in fold_rows])))
    final_model, mean, scale, final_history = fit_fixed_epochs(
        table,
        development_indices,
        columns,
        args,
        device,
        locked_epochs,
        "Rscore_locked",
        progress_dir / "final_training.pt",
    )
    development_probability_final, _, development_features_final = predict(
        final_model,
        table,
        development_indices,
        columns,
        mean,
        scale,
        args,
        device,
        "Rscore_locked",
        "final development representations",
    )
    external_probability, external_attention, external_features = predict(
        final_model,
        table,
        external_indices,
        columns,
        mean,
        scale,
        args,
        device,
        "Rscore_locked",
        "external locked test",
    )
    threshold = select_threshold(labels, oof_probability)
    arm_dir.mkdir(parents=True, exist_ok=True)
    development_output = table.iloc[development_indices][["ID", "event"]].copy()
    development_output["probability"] = oof_probability
    for fusion_name, values in fusion_oof.items():
        development_output[f"{fusion_name}_probability"] = values
    development_output.to_csv(arm_dir / "development_oof_predictions.csv", index=False)
    external_output = table.iloc[external_indices][["ID", "event"]].copy()
    external_output["probability"] = external_probability
    external_output["slice_attention"] = [json.dumps(x) for x in external_attention]
    final_fusion_models = {}
    for fusion_name, extra_columns in fusion_columns.items():
        fusion_model, best_parameters = tune_fusion(
            fusion_features(
                table,
                development_indices,
                development_features_final,
                extra_columns,
                "Rscore_locked",
            ),
            labels,
            args.seed + 1000,
        )
        external_output[f"{fusion_name}_probability"] = fusion_model.predict_proba(
            fusion_features(
                table,
                external_indices,
                external_features,
                extra_columns,
                "Rscore_locked",
            )
        )[:, 1]
        final_fusion_models[fusion_name] = {
            "model": fusion_model,
            "columns": extra_columns,
            "best_parameters": best_parameters,
        }
    external_output.to_csv(arm_dir / "external_test_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(arm_dir / "fold_metrics.csv", index=False)
    (arm_dir / "training_history.json").write_text(
        json.dumps(
            {"outer_folds": histories, "final": final_history}, indent=2
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in final_model.state_dict().items()},
            "arm": arm,
            "tabular_columns": columns,
            "tabular_mean": mean,
            "tabular_scale": scale,
            "locked_epochs": locked_epochs,
            "threshold": threshold,
        },
        arm_dir / "locked_final_model.pt",
    )
    if final_fusion_models:
        joblib.dump(final_fusion_models, arm_dir / "locked_fusion_models.joblib")
    results = [{
        "arm": arm,
        "locked_epochs": locked_epochs,
        **{f"development_{key}": value for key, value in metrics(labels, oof_probability, threshold).items()},
        **{
            f"external_{key}": value
            for key, value in metrics(
                table.iloc[external_indices]["event"].to_numpy(),
                external_probability,
                threshold,
            ).items()
        },
    }]
    for fusion_name, values in fusion_oof.items():
        fusion_threshold = select_threshold(labels, values)
        external_values = external_output[f"{fusion_name}_probability"].to_numpy()
        results.append(
            {
                "arm": fusion_name,
                "locked_epochs": locked_epochs,
                **{
                    f"development_{key}": value
                    for key, value in metrics(labels, values, fusion_threshold).items()
                },
                **{
                    f"external_{key}": value
                    for key, value in metrics(
                        table.iloc[external_indices]["event"].to_numpy(),
                        external_values,
                        fusion_threshold,
                    ).items()
                },
            }
        )
    return results


def smoke_test(table, args, device):
    arm = args.arms[0]
    columns = ARMS[arm]
    indices = np.flatnonzero(table["cohort"].to_numpy() == "development")[:2]
    mean, scale = tabular_statistics(table, indices, columns, "Rscore_oof")
    model = new_model(args, len(columns), device)
    set_encoder_trainable(model, args.smoke_full_transfer)
    model.train()
    videos, tabular, label = patient_input(
        table,
        indices[0],
        columns,
        mean,
        scale,
        args.max_slices,
        args.temporal_mode,
        True,
        device,
        "Rscore_oof",
    )
    logit, attention = model(videos, tabular)
    loss = F.binary_cross_entropy_with_logits(logit, label)
    loss.backward()
    payload = {
        "arm": arm,
        "device": str(device),
        "video_shape": list(videos.shape),
        "logit": float(logit.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "attention_sum": float(attention.detach().sum().cpu()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_test.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def main():
    """主函数：解析命令行参数，构建数据集，执行核心评估及训练流程。"""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the formal full-transfer run but is unavailable. "
            "Use --device cpu --smoke only for pipeline validation."
        )
    table = build_dataset(args.manifest, args.slice_qc, args.rscore_dir, rscore_mode=args.rscore_mode)
    if args.smoke:
        smoke_test(table, args, device)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "checkpoint_sha256": sha256(args.checkpoint),
            "development_complete_cases": int((table["cohort"] == "development").sum()),
            "external_complete_cases": int((table["cohort"] == "external_test").sum()),
            "selection_policy": (
                "Architecture and hyperparameters prespecified; outer 5-fold OOF; "
                "median inner-validation best epoch locked; external evaluated once."
            ),
            "fine_tuning": "head warm-up followed by complete encoder unfreezing",
            "analysis_role": (
                "primary full-cine benchmark"
                if args.temporal_mode == "full"
                else "post hoc single-phase temporal ablation"
            ),
        }
    )
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config.items()
    }
    config_name = (
        "run_config.json"
        if not args.only_folds
        else "run_config_folds_" + "_".join(map(str, args.only_folds)) + ".json"
    )
    (args.output_dir / config_name).write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = []
    for arm in args.arms:
        results.extend(run_arm(arm, table, args, device))
    if results:
        pd.DataFrame(results).to_csv(args.output_dir / "model_comparison.csv", index=False)
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
