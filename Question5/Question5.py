# =============================================================================
# CENG 463 – Machine Learning | Take-Home Midterm
# Question 5

# Dataset : CIFAR-10
# =============================================================================

import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 22
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")

CLASSES = ("plane","car","bird","cat","deer",
           "dog","frog","horse","ship","truck")
NUM_CLASSES = 10
DATA_DIR = "./data"
OUT_DIR  = "./q5_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {name}.png")

def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# 1. DATA LOADING & AUGMENTATION
# =============================================================================

section("1. DATA LOADING & AUGMENTATION")


# Pre-computed average pixel values and standard deviations for the Red, Green, and Blue channels of the entire CIFAR-10 dataset
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

full_train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
test_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)

# 90 / 10 split for train / val
val_size  = int(0.1 * len(full_train))
train_idx = list(range(len(full_train) - val_size))
val_idx   = list(range(len(full_train) - val_size, len(full_train)))

train_set = Subset(full_train, train_idx)
val_set_raw = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=test_transform)
val_set   = Subset(val_set_raw, val_idx)

BATCH = 128
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

print(f"  Train : {len(train_set):,}  |  Val : {len(val_set):,}  |  Test : {len(test_set):,}")


# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

section("2. MODEL DEFINITIONS")

# ── 2a. Deep MLP ─────────────────────────────────────────────────────────────
class DeepMLP(nn.Module):
    """4-hidden-layer MLP with BatchNorm and Dropout."""
    def __init__(self, input_dim=3*32*32, hidden=512,
                 num_classes=10, dropout=0.4):
        super().__init__()
        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.net = nn.Sequential(
            block(input_dim, hidden),
            block(hidden, hidden),
            block(hidden, hidden // 2),
            block(hidden // 2, hidden // 4),
            nn.Linear(hidden // 4, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ── 2b. CNN ──────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CIFARCNN(nn.Module):
    """3 conv-blocks + global average pool + dropout."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64,  pool=True),   # 32→16
            ConvBlock(64,  128, pool=True),   # 16→8
            ConvBlock(128, 256, pool=True),   # 8→4
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(self.dropout(x))


# ── 2c. ResNet18 Transfer Learning ───────────────────────────────────────────

def build_resnet18(num_classes=10, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze last two layers (layer4 + fc)
        for p in model.layer4.parameters():
            p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =============================================================================
# 3. TRAINING UTILITIES
# =============================================================================

section("3. TRAINING UTILITIES")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out  = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs=80,
                lr=1e-3, weight_decay=1e-4, patience=12,
                label="model", scheduler_type="cosine"):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    es = EarlyStopping(patience=patience)
    use_amp = DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    for epoch in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d} | "
                  f"train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")

        if es.step(vl, model):
            print(f"  [{label}] Early stop at epoch {epoch}")
            break

    es.restore(model)
    print(f"  [{label}] Best val loss: {es.best_loss:.4f}")
    return history


def plot_history(histories, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    for ax_idx, (metric, title, ylabel) in enumerate([
        (("train_loss", "val_loss"), "Loss Curves",     "Loss"),
        (("train_acc",  "val_acc"),  "Accuracy Curves",  "Accuracy"),
    ]):
        ax = axes[ax_idx]
        for (label, hist), col in zip(histories.items(), colors):
            ax.plot(hist[metric[0]], ls="--", color=col, alpha=0.6)
            ax.plot(hist[metric[1]], ls="-",  color=col, label=label)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(title="— val  -- train")
    plt.suptitle("Training & Validation Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(filename)


# =============================================================================
# 4. EVALUATION METRICS
# =============================================================================

section("4. EVALUATION METRICS")

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        all_preds.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())
        all_labels.append(y)
    return (torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
            torch.cat(all_probs).numpy())


def top5_error(probs, labels):
    top5 = np.argsort(probs, axis=1)[:, -5:]
    correct = sum(labels[i] in top5[i] for i in range(len(labels)))
    return 1 - correct / len(labels)


def evaluate_model(model, loader, name):
    y_true, y_pred, y_prob = get_predictions(model, loader)
    acc     = accuracy_score(y_true, y_pred)
    f1_mac  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_mic  = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    top5_e  = top5_error(y_prob, y_true)
    report  = classification_report(y_true, y_pred,
                                    target_names=CLASSES, zero_division=0,
                                    output_dict=True)
    per_cls = {c: report[c]["recall"] for c in CLASSES}
    print(f"\n  ── {name} ──")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Macro-F1    : {f1_mac:.4f}")
    print(f"    Micro-F1    : {f1_mic:.4f}")
    print(f"    Top-5 Error : {top5_e:.4f}")
    print(f"    Per-class recall: {per_cls}")
    return dict(name=name, accuracy=acc, macro_f1=f1_mac,
                micro_f1=f1_mic, top5_error=top5_e,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def plot_confusion_matrix(y_true, y_pred, name, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    save_fig(filename)


# =============================================================================
# 5. HYPERPARAMETER OPTIMISATION WITH OPTUNA
# =============================================================================

section("5. OPTUNA HYPERPARAMETER OPTIMISATION (CNN)")


def objective_cnn(trial):
    lr       = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout  = trial.suggest_float("dropout",      0.2,  0.6)
    batch_sz = trial.suggest_categorical("batch_size", [64, 128, 256])

    t_loader = DataLoader(train_set, batch_size=batch_sz,
                          shuffle=True, num_workers=2, pin_memory=False)
    v_loader = DataLoader(val_set, batch_size=batch_sz,
                          shuffle=False, num_workers=2, pin_memory=False)

    model = CIFARCNN(dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    use_amp = DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\n  Starting Trial {trial.number} (Batch: {batch_sz}, LR: {lr:.5f})")

    for epoch in range(1, 21):
        train_epoch(model, t_loader, optimizer, criterion, scaler)
        scheduler.step()

        print(f"    -> Finished epoch {epoch}/20")

    _, val_acc = eval_epoch(model, v_loader, criterion)
    print(f"  Trial {trial.number} finished with Val Acc: {val_acc:.4f}")

    # Explicitly delete models/loaders to free up GPU memory between trials
    del model, t_loader, v_loader, optimizer
    torch.cuda.empty_cache()

    return val_acc

print("  Running 20-trial Optuna study…")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))

study.optimize(objective_cnn, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print(f"\n  Best trial val-acc : {study.best_value:.4f}")
print(f"  Best params        : {best_params}")

try:
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Optuna — Hyperparameter Importance (CNN)", fontweight="bold")
    plt.tight_layout()
    save_fig("q5_optuna_importance")
except Exception:
    pass


# =============================================================================
# 6. TRAIN FINAL MODELS
# =============================================================================

section("6. TRAINING FINAL MODELS")

# ── 6a. Deep MLP ─────────────────────────────────────────────────────────────
print("\n  Training MLP…")
mlp = DeepMLP(dropout=0.4)
hist_mlp = train_model(mlp, train_loader, val_loader,
                       epochs=80, lr=1e-3, weight_decay=1e-4,
                       patience=15, label="MLP")
torch.save(mlp.state_dict(), os.path.join(OUT_DIR, "mlp.pth"))

# ── 6b. CNN (best Optuna params) ─────────────────────────────────────────────
print("\n  Training CNN…")
cnn = CIFARCNN(dropout=best_params.get("dropout", 0.5))
hist_cnn = train_model(
    cnn, train_loader, val_loader,
    epochs=80,
    lr=best_params.get("lr", 1e-3),
    weight_decay=best_params.get("weight_decay", 1e-4),
    patience=15, label="CNN"
)
torch.save(cnn.state_dict(), os.path.join(OUT_DIR, "cnn.pth"))

# ── 6c. ResNet18 Transfer Learning ───────────────────────────────────────────
print("\n  Training ResNet18 (transfer learning)…")
resnet = build_resnet18()
hist_res = train_model(resnet, train_loader, val_loader,
                       epochs=60, lr=5e-4, weight_decay=1e-4,
                       patience=12, label="ResNet18")
torch.save(resnet.state_dict(), os.path.join(OUT_DIR, "resnet18.pth"))

# Plot training curves
plot_history({"MLP": hist_mlp, "CNN": hist_cnn, "ResNet18": hist_res},
             "q5_training_curves")



# =============================================================================
# 7. EVALUATION
# =============================================================================

section("7. EVALUATION ON TEST SET")

res_mlp    = evaluate_model(mlp,    test_loader, "MLP")
res_cnn    = evaluate_model(cnn,    test_loader, "CNN")
res_resnet = evaluate_model(resnet, test_loader, "ResNet18")

for r in [res_mlp, res_cnn, res_resnet]:
    plot_confusion_matrix(r["y_true"], r["y_pred"],
                          r["name"],
                          f"q5_cm_{r['name'].lower()}")

# Summary table
df_eval = pd.DataFrame([
    {k: v for k, v in r.items()
     if k not in ("y_true","y_pred","y_prob")}
    for r in [res_mlp, res_cnn, res_resnet]
])
print("\n  Summary:")
print(df_eval.to_string(index=False))
df_eval.to_csv(os.path.join(OUT_DIR, "q5_results.csv"), index=False)

# Per-class recall bar chart
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(NUM_CLASSES)
w = 0.25
for i, (r, col) in enumerate(zip([res_mlp, res_cnn, res_resnet],
                                  ["#4e79a7","#f28e2b","#e15759"])):
    per_cls = classification_report(
        r["y_true"], r["y_pred"],
        target_names=CLASSES, zero_division=0, output_dict=True
    )
    recalls = [per_cls[c]["recall"] for c in CLASSES]
    ax.bar(x + i * w, recalls, w, label=r["name"], color=col)
ax.set_xticks(x + w); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
ax.set_title("Per-Class Recall — All Models", fontweight="bold")
ax.legend()
plt.tight_layout()
save_fig("q5_per_class_recall")


# =============================================================================
# 8. GRAD-CAM (CNN)
# =============================================================================

section("8. GRAD-CAM  —  CNN Misclassified Examples")

class GradCAM:
    """Grad-CAM for the last conv block."""
    def __init__(self, model, target_layer):
        self.model   = model
        self.grads   = None
        self.activs  = None
        target_layer.register_forward_hook(self._save_activs)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_activs(self, _, __, output):
        self.activs = output.detach()

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.unsqueeze(0).to(DEVICE).requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activs).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(32, 32),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def unnorm(t):
    """Undo CIFAR normalisation for display."""
    m = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    s = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (t * s + m).permute(1, 2, 0).clamp(0, 1).numpy()


grad_cam = GradCAM(cnn, cnn.features[-1].block[-2])   # last BN before ReLU

for module in cnn.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False

# Collect misclassified examples
cnn.eval()
misclassified = []
test_no_aug = datasets.CIFAR10(DATA_DIR, train=False,
                               download=False, transform=test_transform)

for idx in range(len(test_no_aug)):
    img, label = test_no_aug[idx]
    with torch.no_grad():
        pred = cnn(img.unsqueeze(0).to(DEVICE)).argmax(1).item()
    if pred != label:
        misclassified.append((img, label, pred))
    if len(misclassified) == 10:
        break

# Plot 10 misclassified with Grad-CAM overlay
fig, axes = plt.subplots(10, 3, figsize=(9, 34))
for row, (img, true_lbl, pred_lbl) in enumerate(misclassified):
    cam  = grad_cam(img, class_idx=pred_lbl)
    orig = unnorm(img)

    # Original
    axes[row, 0].imshow(orig)
    axes[row, 0].set_title(f"True: {CLASSES[true_lbl]}", fontsize=8)
    axes[row, 0].axis("off")

    # Grad-CAM heatmap
    axes[row, 1].imshow(cam, cmap="jet")
    axes[row, 1].set_title("Grad-CAM", fontsize=8)
    axes[row, 1].axis("off")

    # Overlay
    axes[row, 2].imshow(orig)
    axes[row, 2].imshow(cam, cmap="jet", alpha=0.45)
    axes[row, 2].set_title(f"Pred: {CLASSES[pred_lbl]}", fontsize=8, color="red")
    axes[row, 2].axis("off")

plt.suptitle("CNN: 10 Misclassified Samples with Grad-CAM\n"
             "(left: original | centre: heatmap | right: overlay)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
save_fig("q5_gradcam_misclassified")



# =============================================================================
# 9. SHAP — MLP Individual Prediction Explanations
# =============================================================================

section("9. SHAP  —  MLP Explanations")

try:
    import shap

    mlp.eval()
    mlp.to("cpu")

    # Use a small background set and 50 test samples
    bg_imgs  = torch.stack([test_no_aug[i][0] for i in range(200)])
    tst_imgs = torch.stack([test_no_aug[i][0] for i in range(200, 250)])
    tst_lbls = [test_no_aug[i][1] for i in range(200, 250)]

    def mlp_predict(x_np):
        t = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return mlp(t).numpy()

    bg_flat  = bg_imgs.view(200, -1).numpy()
    tst_flat = tst_imgs.view(50, -1).numpy()

    explainer   = shap.KernelExplainer(mlp_predict, shap.kmeans(bg_flat, 50))
    shap_values = explainer.shap_values(tst_flat[:5], nsamples=100)

    # Plot summary for one sample per class
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # 1. Extract the raw numbers (newer SHAP versions hide them inside an object)
    if hasattr(shap_values, "values"):
        raw_shap = shap_values.values
    else:
        raw_shap = shap_values


    shap_arr = np.array(raw_shap)

    for i in range(5):
        lbl = tst_lbls[i]


        if shap_arr.shape == (10, 5, 3072):
            sv = shap_arr[lbl, i, :]
        elif shap_arr.shape == (5, 10, 3072):
            sv = shap_arr[i, lbl, :]
        elif shap_arr.shape == (5, 3072, 10):
            sv = shap_arr[i, :, lbl]
        elif shap_arr.shape == (5, 3072):
            sv = shap_arr[i, :]
        elif isinstance(shap_values, list) and len(shap_values) == 10:
            sv = shap_values[lbl][i]
        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")


        sv = np.array(sv)
        if sv.size != 3072:
            raise ValueError(f"Extracted {sv.size} values instead of 3072!")

        img  = sv.reshape(3, 32, 32)
        importance = np.abs(img).mean(axis=0)  # spatial importance map

        axes[i].imshow(unnorm(tst_imgs[i]))
        axes[i].imshow(importance, cmap="hot", alpha=0.5)
        axes[i].set_title(f"True: {CLASSES[lbl]}\n"
                          f"Pred: {CLASSES[mlp(tst_imgs[i:i+1]).argmax(1).item()]}",
                          fontsize=8)
        axes[i].axis("off")

    plt.suptitle("SHAP Feature Importance — MLP (pixel-level)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("q5_shap_mlp")
    mlp.to(DEVICE)

except ImportError:
    print("  SHAP not installed — skipping. Run: pip install shap")
    mlp.to(DEVICE)


# =============================================================================
# 10. ADVERSARIAL ROBUSTNESS  —  FGSM & PGD
# =============================================================================

section("10. ADVERSARIAL ROBUSTNESS  —  FGSM & PGD")

CIFAR_MIN = torch.tensor(
    [(0 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]
).view(3, 1, 1).to(DEVICE)

CIFAR_MAX = torch.tensor(
    [(1 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]
).view(3, 1, 1).to(DEVICE)


def fgsm_attack(model, x, y, eps):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss  = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return torch.clamp(x_adv.detach(), CIFAR_MIN, CIFAR_MAX)


def pgd_attack(model, x, y, eps, alpha=0.01, steps=10):
    model.eval()
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, CIFAR_MIN, CIFAR_MAX)
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        loss  = nn.CrossEntropyLoss()(model(x_adv), y)
        model.zero_grad(); loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, CIFAR_MIN, CIFAR_MAX).detach()
    return x_adv


def adv_accuracy(model, loader, attack_fn, **kwargs):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 1. Temporarily ENABLE gradients just for the attack generation
        with torch.enable_grad():
            x_adv = attack_fn(model, x, y, **kwargs)

        # 2. DISABLE gradients for the actual inference step to save memory
        with torch.no_grad():
            preds = model(x_adv).argmax(1)

        correct += preds.eq(y).sum().item()
        total   += y.size(0)

    return correct / total


# Use a smaller subset for speed
adv_subset = Subset(test_set, list(range(1000)))
adv_loader  = DataLoader(adv_subset, batch_size=100, shuffle=False)

EPS_LIST = [0.01, 0.03, 0.05]

adv_results = []
for eps in EPS_LIST:
    print(f"\n  eps = {eps}")
    for model, name in [(mlp, "MLP"), (cnn, "CNN"), (resnet, "ResNet18")]:
        fgsm_acc = adv_accuracy(model, adv_loader, fgsm_attack, eps=eps)
        pgd_acc  = adv_accuracy(model, adv_loader, pgd_attack,  eps=eps)
        print(f"    {name:8s}  FGSM: {fgsm_acc:.4f}  PGD: {pgd_acc:.4f}")
        adv_results.append(dict(model=name, eps=eps,
                                fgsm=fgsm_acc, pgd=pgd_acc))

df_adv = pd.DataFrame(adv_results)
df_adv.to_csv(os.path.join(OUT_DIR, "q5_adversarial.csv"), index=False)

# Plot adversarial accuracy vs epsilon
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, attack, col_title in zip(axes, ["fgsm", "pgd"],
                                  ["FGSM Attack", "PGD Attack"]):
    for name, col in [("MLP","#4e79a7"),("CNN","#f28e2b"),("ResNet18","#e15759")]:
        sub = df_adv[df_adv["model"] == name]
        ax.plot(sub["eps"], sub[attack], "o-", color=col, label=name)
    ax.set_xlabel("ε (epsilon)"); ax.set_ylabel("Accuracy under attack")
    ax.set_title(col_title, fontweight="bold"); ax.legend()
    ax.set_ylim(0, 1)
plt.suptitle("Adversarial Robustness: Accuracy vs ε",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("q5_adversarial_robustness")

# Visualise a few FGSM adversarial examples
cnn.eval()
x_batch, y_batch = next(iter(adv_loader))
x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
x_fgsm = fgsm_attack(cnn, x_batch, y_batch, eps=0.03)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i in range(8):
    axes[0, i].imshow(unnorm(x_batch[i].cpu()))
    axes[0, i].set_title(CLASSES[y_batch[i].item()], fontsize=7)
    axes[0, i].axis("off")
    pred_adv = cnn(x_fgsm[i:i+1]).argmax(1).item()
    axes[1, i].imshow(unnorm(x_fgsm[i].cpu()))
    axes[1, i].set_title(CLASSES[pred_adv], fontsize=7,
                          color="red" if pred_adv != y_batch[i].item() else "green")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original",    fontsize=9)
axes[1, 0].set_ylabel("FGSM ε=0.03", fontsize=9)
plt.suptitle("FGSM Adversarial Examples (CNN)\nRed title = fooled",
             fontsize=11, fontweight="bold")
plt.tight_layout()
save_fig("q5_fgsm_examples")